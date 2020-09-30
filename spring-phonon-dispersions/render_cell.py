import numpy as np
import vpython as vp

class RenderCell(object):
    """ Pipes the cell object to our render pipeline,
    which runs through a vpython interface.
    """
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
    
    def __init__(self, cell_obj):
        self.initial_velocity = vp.vector(0.,0.,0.)
        self.cell_obj = cell_obj
        self._initialize()
        
    def _initialize(self):
        every_other = 0
        for _, position, force in self.cell_obj.atom_list:
            atom = vp.sphere(pos=position, radius=0.35,
                    velocity=vp.vector(0.,0.,0.), mass=1.0, initial_force=vp.vector(0.,0.,0),
                    make_trail=False)
            self.sphere_list.append(atom)
            if every_other == 1:
                self.spring_list.append(vp.helix(pos=position, atom_one=atom, atom_two=last_atom, eq_len=(atom.pos-last_atom.pos).mag,
                                                axis=(last_atom.pos-atom.pos), radius=0.25, force_constant=force))
                every_other = 0
            every_other += 1
            last_atom  = atom
            
    
    def render(self):
        self.initial_force_factor = 0.
        
        # Simulation loop
        while True:
            # Determine if we're paused or not
            if self.running:
                vp.rate(self.rate_of_animation)
                
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
                    
                    # Recalculate the force on all atoms attached to every spring... This will be slow... 
                    spring_force = vp.vector(0.,0.,0.)
                    for s in self.spring_list:
                        new_len = (s.atom_one.pos-s.atom_two.pos).mag
                        spring_force += -s.force_constant*(s.eq_len-new_len)*force_dir
                    
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
        atom_position_graph = vp.graph(title='Absolute Displacement of Atom(s)', xtitle=r't [s]', ytitle='r(t) [m]', fast=True)
        self.atom_position_points = vp.gcurve(graph=atom_position_graph, color=vp.color.red, size=0.1)
        atom_velocity_graph = vp.graph(title='Absolute Speed of Atom(s)', xtitle=r't [s]', ytitle=r'|r(t)-vector| [m/s]', fast=True)
        self.atom_velocity_points = vp.gcurve(graph=atom_velocity_graph, color=vp.color.blue, size=0.1)
    
    def _update_plots(self, new_pos, new_vel):
        magnitude = new_pos.mag
        self.atom_position_points.plot(pos=(self.time, magnitude))
        
        magnitude = new_vel.mag
        self.atom_velocity_points.plot(pos=(self.time, magnitude))
    
    def menu(self):
        units = " Newtons \n"
        # Slider for controlling inward force
        vp.scene.append_to_caption("Apply a force towards each atom: \n")
        self.force_slider = vp.slider(pos=(0.,2.), text="Set Force Between Atoms", bind=self._update_force, min=0.0, max=10.0, right=15)
        self.force_wtext  = vp.wtext(text='{:1.2f}'.format(self.initial_force_factor))
        vp.scene.append_to_caption(units)
        
        # Slider for force only in the left direction.
        vp.scene.append_to_caption("Apply a force towards the left: \n")
        self.force_slider_left = vp.slider(pos=(0.,2.), text="Set Force to left", bind=self._update_force_left, min=0.0, max=10.0, right=15)
        self.force_wtext_left  = vp.wtext(text='{:1.2f}'.format(self.initial_force_factor_left))
        vp.scene.append_to_caption(units)
        
        # Slider for force only in the right direction.
        vp.scene.append_to_caption("Apply a force towards the right: \n")
        self.force_slider_right = vp.slider(pos=(0.,2.), text="Set Force to right", bind=self._update_force_right, min=0.0, max=10.0, right=15)
        self.force_wtext_right  = vp.wtext(text='{:1.2f}'.format(self.initial_force_factor_right))
        vp.scene.append_to_caption(units)
        
        # Slider to change the spring constant.
        vp.scene.append_to_caption("Apply a force towards the right: \n")
        self.spring_constant_slider = vp.slider(pos=(0.,2.), text="Set spring constant", bind=self._update_spring_constant, min=1.0, max=10.0, right=15)
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