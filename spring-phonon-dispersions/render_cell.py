import numpy as np
import vpython as vp

class RenderCell(object):
    """ Pipes the cell object to our render pipeline,
    which runs through a vpython interface.
    """
    # Class globals
    use_two_body_analytical = False
    
    # Forces filled by sliders
    initial_force_factor = 0.
    initial_force_factor_right = 0.
    initial_force_factor_left = 0.
    
    # Text boxes for sliders
    force_wtext           = vp.wtext(text="")
    force_wtext_right     = vp.wtext(text="")
    force_wtext_left      = vp.wtext(text="")
    spring_constant_wtext = vp.wtext(text="")
    
    # Animation parameters
    animation_time_step = 0.01
    rate_of_animation = 1/animation_time_step
    time_step = 0.005
    stop_time = 10.
    time = 0.
    running = True
    
    sphere_list = []
    spring_list = []
    
    def __init__(self, cell_obj, use_two_body_analytical=False):
        self.use_two_body_analytical = use_two_body_analytical
        self.initial_velocity = vp.vector(0.,0.,0.)
        self.cell_obj = cell_obj
        self._initialize()
        
    def _initialize(self):
        # Construct and place all atoms.
        for _, position, force, connected in self.cell_obj.atom_list:
            atom = vp.sphere(pos=position, radius=0.35,
                    velocity=vp.vector(0.,0.,0.), mass=1.0, initial_force=vp.vector(0.,0.,0),
                    make_trail=False, connected=connected, initial_pos=position, force_constant=force)
            self.sphere_list.append(atom)
        
        # Construct and place all springs between the atoms if connected.
        # Grab every other atom.
        first_set_atoms = self.sphere_list[0::2]
        second_set_atoms = self.sphere_list[1::2]
        atoms_tuple_list = list(zip(first_set_atoms, second_set_atoms))
        
        for atom_one, atom_two in atoms_tuple_list:
            # Attach a spring between every atom unless specified otherwise.
            if (atom_one.connected) and (atom_two.connected):
                self.spring_list.append(vp.helix(pos=atom_one.pos, atom_one=atom_one, atom_two=atom_two, eq_len=(atom_one.pos-atom_two.pos).mag,
                                                axis=(atom_two.pos-atom_one.pos), radius=0.25, force_constant=atom_one.force_constant,
                                        coils=atom_one.force_constant+4))
            
    def apply_physics(self, spring, two_body_analytical=False, t=0):
        """ Return the force exerted on the atom based on initial conditions. If using analytical
        two body, then just change the positions of each atom at a given time t.
        """
        # Newton's Method
        if not two_body_analytical:
            force_dir = (spring.atom_two.pos - spring.atom_one.pos).hat
            #for s in self.spring_list:
            s = spring # Tired of typing spring.
            new_len = (s.atom_one.pos-s.atom_two.pos).mag
            spring_force = -s.force_constant*(s.eq_len-new_len)*force_dir
            return spring_force
        
        # Analytical method for two body situation.
        force_dir = (spring.atom_two.pos - spring.atom_one.pos).hat
        s = spring # Tired of typing spring.
        k_over_m_sqrt = np.sqrt(s.force_constant/(s.atom_one.mass))
        omega = k_over_m_sqrt
        phase = 0.
        amplitude_1 = 1.
        s.atom_one.pos = -(amplitude_1 * s.eq_len * np.cos(omega*t+phase))*force_dir + -amplitude_1 * s.eq_len*force_dir + s.atom_one.initial_pos
        s.atom_one.velocity = (amplitude_1 * s.eq_len * omega * np.sin(omega*t+phase))*force_dir
        
        s.atom_two.pos = (amplitude_1 * s.eq_len * np.cos(omega*t+phase))*force_dir + amplitude_1 * s.eq_len*force_dir + s.atom_two.initial_pos
        s.atom_two.velocity = -(amplitude_1 * omega * np.sin(omega*t+phase))*force_dir
    
    def render(self):
        self.initial_force_factor = 0.
        # Simulation loop
        while True:
            # Determine if we're paused or not
            if self.running:
                vp.rate(self.rate_of_animation)
                
                # Reset spring force
                spring_force = vp.vector(0.,0.,0.)
                
                # Iterate over all of the springs.
                for spring in self.spring_list:
                    # Grab direction if an initial force exists. Only apply an initial force on the first atom.
                    force_dir = (spring.atom_two.pos - spring.atom_one.pos).hat
                    if self.initial_force_factor:
                        # Apply force in both possible directions
                        force_in_direction = force_dir*self.initial_force_factor
                        # Apply force as the initial force to every atom acting as if they're oscillating inwards.
                        for s in self.spring_list:
                            s.atom_one.initial_force = force_in_direction
                            s.atom_two.initial_force = -force_in_direction
                        
                    elif self.initial_force_factor_left:
                        force_in_direction = -force_dir*self.initial_force_factor_left
                        spring.atom_one.initial_force = force_in_direction
                        
                    elif self.initial_force_factor_right:
                        force_in_direction = force_dir*self.initial_force_factor_right
                        spring.atom_one.initial_force = force_in_direction
                    
                    # Calculate force on spring's atoms just from displacement
                    if self.use_two_body_analytical:
                        self.apply_physics(spring, True, t=self.time)
                    else:
                        spring_force = self.apply_physics(spring)
                    
                    # First atom bound to spring
                    spring.atom_one.force = spring_force + spring.atom_one.initial_force
                    spring.atom_one.velocity += spring.atom_one.force/spring.atom_one.mass*self.animation_time_step
                    spring.atom_one.pos += spring.atom_one.velocity*self.animation_time_step
                    
                    # Second atom bound to spring
                    spring.atom_two.force = -spring_force + spring.atom_two.initial_force
                    spring.atom_two.velocity += spring.atom_two.force/spring.atom_two.mass*self.animation_time_step
                    spring.atom_two.pos += spring.atom_two.velocity*self.animation_time_step
                    
                    spring.axis = spring.atom_two.pos - spring.atom_one.pos
                    spring.pos  = spring.atom_one.pos
                
                # Are we also plotting information?
                if self.using_plots:
                    # Pick our data. It seems like most of these want three cases...
                    try:
                        spheres_pos = [self.sphere_list[0].pos,self.sphere_list[2].pos,self.sphere_list[4].pos]
                        spheres_vel = [self.sphere_list[0].velocity,self.sphere_list[2].velocity,self.sphere_list[4].velocity]
                        self._update_plots(spheres_pos, spheres_vel)
                    except IndexError:
                        # Should always be at least one case
                        self._update_plots(self.sphere_list[0].pos, self.sphere_list[0].velocity)
                
                # Update time per step size.
                self.time += self.time_step
                self.initial_force_factor = 0.
                self.initial_force_factor_right = 0.
                self.initial_force_factor_left = 0.
                
                # Clear initial forces at the end of the frame.
                spring.atom_one.initial_force = vp.vector(0.,0.,0.)
                spring.atom_two.initial_force = vp.vector(0.,0.,0.)
    
    def plot_supplimentary_information(self):
        self.using_plots = True
        atom_position_graph = vp.graph(title='Absolute Displacement of Atom(s)', 
                                       xtitle=r't [s]', ytitle=r'|r(t)-vector| [m]', fast=False,
                                       legend=True)
        atom_velocity_graph = vp.graph(title='Absolute Speed of Atom(s)', xtitle=r't [s]', ytitle=r'|r(t)-dot-vector| [m/s]', fast=False, 
                                       legend=True)
        if len(self.sphere_list) > 5:
            self.atom_position_points = vp.gcurve(graph=atom_position_graph, color=vp.color.red, size=0.1, 
            label="Spring Constant: {} N/m".format(self.sphere_list[0].force_constant))
            self.atom_position_points_2 = vp.gcurve(graph=atom_position_graph, color=vp.color.blue, size=0.1, 
            label="Spring Constant: {} N/m".format(self.sphere_list[2].force_constant))
            self.atom_position_points_3 = vp.gcurve(graph=atom_position_graph, color=vp.color.green, size=0.1, 
            label="Spring Constant: {} N/m".format(self.sphere_list[4].force_constant))
            
            self.atom_velocity_points = vp.gcurve(graph=atom_velocity_graph, color=vp.color.red, size=0.1, 
                label="Spring Constant: {} N/m".format(self.sphere_list[0].force_constant))
            self.atom_velocity_points_2 = vp.gcurve(graph=atom_velocity_graph, color=vp.color.blue, size=0.1, 
                label="Spring Constant: {} N/m".format(self.sphere_list[2].force_constant))
            self.atom_velocity_points_3 = vp.gcurve(graph=atom_velocity_graph, color=vp.color.green, size=0.1, 
                label="Spring Constant: {} N/m".format(self.sphere_list[4].force_constant))
        else:
            self.atom_position_points = vp.gcurve(graph=atom_position_graph, color=vp.color.red, size=0.1, 
                label="Spring Constant: {} N/m".format(self.sphere_list[0].force_constant))
            self.atom_velocity_points = vp.gcurve(graph=atom_velocity_graph, color=vp.color.red, size=0.1, 
                label="Spring Constant: {} N/m".format(self.sphere_list[0].force_constant))
    
    def _update_plots(self, new_pos, new_vel):
        try:
            magnitude = new_pos[0].mag
            self.atom_position_points.plot(pos=(self.time, magnitude))
            magnitude = new_pos[1].mag
            self.atom_position_points_2.plot(pos=(self.time, magnitude))
            magnitude = new_pos[2].mag
            self.atom_position_points_3.plot(pos=(self.time, magnitude))
        except IndexError:
            # Should have at least one at the basic case.
            magnitude = new_pos.mag
            self.atom_position_points.plot(pos=(self.time, magnitude))
        
        try:
            magnitude = new_vel[0].mag
            self.atom_velocity_points.plot(pos=(self.time, magnitude))
            magnitude = new_vel[1].mag
            self.atom_velocity_points_2.plot(pos=(self.time, magnitude))
            magnitude = new_vel[2].mag
            self.atom_velocity_points_3.plot(pos=(self.time, magnitude))
        except IndexError:
            magnitude = new_vel.mag
            self.atom_velocity_points.plot(pos=(self.time, magnitude))
            
    def menu(self):
        units = " Newtons \n"
        # Slider for controlling inward force
        vp.scene.append_to_caption("Apply a force towards each atom: \n")
        self.force_slider = vp.slider(pos=(0.,2.), text="Set Force Between Atoms", bind=self._update_force, min=0.0, max=1.0, right=15)
        self.force_wtext  = vp.wtext(text='{:1.2f}'.format(self.initial_force_factor))
        vp.scene.append_to_caption(units)
        
        # Slider for force only in the left direction.
        vp.scene.append_to_caption("Apply a force towards the right: \n")
        self.force_slider_left = vp.slider(pos=(0.,2.), text="Set Force to left", bind=self._update_force_left, min=0.0, max=1.0, right=15)
        self.force_wtext_left  = vp.wtext(text='{:1.2f}'.format(self.initial_force_factor_left))
        vp.scene.append_to_caption(units)
        
        # Slider for force only in the right direction.
        vp.scene.append_to_caption("Apply a force towards the left: \n")
        self.force_slider_right = vp.slider(pos=(0.,2.), text="Set Force to right", bind=self._update_force_right, min=0.0, max=1.0, right=15)
        self.force_wtext_right  = vp.wtext(text='{:1.2f}'.format(self.initial_force_factor_right))
        vp.scene.append_to_caption(units)
        
        # Slider to change the spring constant.
        vp.scene.append_to_caption("Change the Spring Constant: \n")
        self.spring_constant_slider = vp.slider(pos=(0.,2.), text="Set spring constant", bind=self._update_spring_constant, min=1.0, max=50.0, right=15)
        self.spring_constant_wtext  = vp.wtext(text='{:1.2f}'.format(self.spring_constant_slider.value))
        vp.scene.append_to_caption(" Newtons / meter \n")
        
        # Add a Pause/Play button.
        vp.button(pos=vp.scene.title_anchor, text="Pause", bind=self._pause)
    
    def _pause(self, b):
        # Global bool to pause or play simulation
        if self.running:
            self.running = False
            b.text = "Play"
        else:
            self.running = True
            b.text = "Pause"
        
    def _update_force(self, s):
        # Update the initial force, then set the slider back to zero.
        self.initial_force_factor = s.value
        self.force_wtext.text ='{:1.2f}'.format(s.value)
        self.force_slider.value = 0.
    
    def _update_force_left(self, s):
        # Update the initial force, then set the slider back to zero.
        self.initial_force_factor_left = s.value
        self.force_wtext_left.text ='{:1.2f}'.format(s.value)
        self.force_slider_left.value = 0.
    
    def _update_force_right(self, s):
        # Update the initial force, then set the slider back to zero.
        self.initial_force_factor_right = s.value
        self.force_wtext_right.text ='{:1.2f}'.format(s.value)
        self.force_slider_right.value = 0.
    
    def _update_spring_constant(self, s):
        # Set the spring constant
        self.spring_constant_wtext.text ='{:1.2f}'.format(s.value)
        for spring in self.spring_list:
            spring.force_constant = s.value
            spring.coils = s.value + 4.