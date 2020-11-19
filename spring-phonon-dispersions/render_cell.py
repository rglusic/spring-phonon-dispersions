import numpy as np
import vpython as vp

class RenderCell(object):
    """ Pipes the cell object to our render pipeline,
    which runs through a vpython interface.
    """
    # Animation parameters
    animation_time_step = 0.001
    rate_of_animation = 1/animation_time_step
    time_step = 0.005
    time = 0.
    running = True
    plots = False
    
    atom_list = []
    spring_list = []
    
    def __init__(self, cell_obj, threshold):
        self.cell_obj = cell_obj
        self._initialize(threshold)
        
    def _initialize(self, threshold):
        # Construct and place all atoms.
        for _, position, force in self.cell_obj.atom_list:
            atom = vp.sphere(pos=position, radius=0.35,
                    vel=vp.vector(0.,0.,0.), mass=1.0, initial_force=vp.vector(0.,0.,0),
                    make_trail=False, initial_pos=position, force_constant=force,
                    label=position.x+position.y+position.z)
            self.atom_list.append(atom)
        
        # Construct and place all springs between the atoms if they're within the lattice distance.
        for atom in self.atom_list:
            for connect in self.atom_list[:-1]:
                # if the atom connection exists, skip.
                exists = False
                for spring in self.spring_list:
                    if (spring.atom_one == atom and spring.atom_two == connect) or (spring.atom_one == connect and spring.atom_two == atom) or atom == connect:
                        exists = True
                if exists:
                    continue
                
                # Check distance is between threshold.
                separation = (atom.pos - connect.pos).mag
                if separation == 0.0:
                    continue
                
                if (separation <= threshold):
                    # Attach a spring between every atom unless specified otherwise.
                    self.spring_list.append(vp.helix(pos=atom.pos, atom_one=atom, atom_two=connect, eq_len=separation,
                                                axis=(connect.pos-atom.pos), radius=0.25, force_constant=atom.force_constant,
                                                    coils=atom.force_constant+4))
            
    def calculate_numerical(self, spring, dt):
        """ Calculates the Newtonian using a form of Euler's Method. Returns
        the first atoms force, position, and velocity.
        """
        # atoms should be R3 position vectors
        # spring constant and eq_len are constants
        # time is an array of incremented time values. Assume dt is constant.
        pos1 = spring.atom_one.pos
        pos2 = spring.atom_two.pos
        eq_len = spring.eq_len
        spring_constant = spring.force_constant
        mass = spring.atom_one.mass # Assume that mass is the same.
        
        # save previous values
        old_pos1, old_pos2 = pos1, pos2
        old_vel1, old_vel2 = spring.atom_one.vel, spring.atom_two.vel
        
        # find initial spring force
        spr_len = (old_pos2 - old_pos1).mag
        # assume spr_dir_hat must be constant.
        spr_dir_hat = (old_pos2 - old_pos1).hat # vector is towards pos2
        spr_force = spring_constant * (spr_len - eq_len) * (-1)*spr_dir_hat
        
        # Calculate the force contribution from the other atoms.
        for new_spring in self.spring_list:
            cont_spr_len = (new_spring.atom_one.pos - new_spring.atom_two.pos).mag
            cont_spr_dir_hat = (new_spring.atom_one.pos - new_spring.atom_two.pos).hat
            cont_spr_force = spring_constant * (cont_spr_len - new_spring.eq_len) * (+1)*cont_spr_dir_hat
            spr_force += cont_spr_force
        
        # recalculate velocity from old spring force 
        new_vel1 = old_vel1 + (-1)*spr_force / mass * dt
        new_vel2 = old_vel2 + (+1)*spr_force / mass * dt 
        
        # recalculate new position from old velocity 
        new_pos1 = old_pos1 + new_vel1 * dt
        new_pos2 = old_pos2 + new_vel2 * dt
        
        # Update atoms positions, velocities, etc.
        spring.atom_one.vel = new_vel1
        spring.atom_one.pos = new_pos1

        spring.atom_two.vel = new_vel2
        spring.atom_two.pos = new_pos2
        return spr_force, new_vel1, new_pos1, new_vel2, new_pos2
    
    def change_atom_pos(self, new_pos):
        """Change the position of the first atom in the atom list.
        """
        self.atom_list[0].color = vp.vector(1.0, 0. ,0.)
        self.atom_list[0].pos += new_pos
    
    def render(self):
        # Simulation loop
        while True:
            # Determine if we're paused or not
            if self.running:
                vp.rate(self.rate_of_animation)
                
                # Iterate over all of the springs.
                for spring in self.spring_list:
                    self.calculate_numerical(spring, self.animation_time_step)
                    
                    spring.axis = spring.atom_two.pos - spring.atom_one.pos
                    spring.pos  = spring.atom_one.pos
                    
                    # Plot the first atoms position and velocity
                    if self.plots:
                        self._update_plots(self.spring_list[0].atom_one.pos, self.spring_list[0].atom_one.vel)
                        
                    # Update time
                    self.time += self.animation_time_step
    
    def plot_supplimentary_information(self):
        self.plots = True
        atom_position_graph = vp.graph(title='Absolute Displacement of Red Atom', 
                                       xtitle=r't [s]', ytitle=r'|r(t)-vector| [m]', fast=True,
                                       legend=True)
        atom_velocity_graph = vp.graph(title='Absolute Speed of Red Atom', xtitle=r't [s]', ytitle=r'|r(t)-dot-vector| [m/s]', fast=True, 
                                       legend=True)
        atom_phasespace     = vp.graph(title='Phase space of Red Atom', ytitle=r'|r(t)-vector| [m]', fast=True, 
                                       legend=True, xtitle=r'|r(t)-dot-vector| [m/s]')
        label_constant = "Spring Constant: {} N/m"
        self.atom_position_points = vp.gcurve(graph=atom_position_graph, color=vp.color.red, size=0.1, 
        label=label_constant.format(self.atom_list[0].force_constant))
        
        self.atom_velocity_points = vp.gcurve(graph=atom_velocity_graph, color=vp.color.blue, size=0.1, 
            label=label_constant.format(self.atom_list[0].force_constant))

        self.atom_phasespace_points = vp.gcurve(graph=atom_phasespace, color=vp.color.green, size=0.1, 
            label=label_constant.format(self.atom_list[0].force_constant))
    
    def _update_plots(self, new_pos, new_vel):
            pos_mag = new_pos.mag
            self.atom_position_points.plot(pos=(self.time, pos_mag))

            vel_mag = new_vel.mag
            self.atom_velocity_points.plot(pos=(self.time, vel_mag))
            
            self.atom_phasespace_points.plot(pos=(vel_mag, pos_mag))
            
            
            
    def menu(self):
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
    