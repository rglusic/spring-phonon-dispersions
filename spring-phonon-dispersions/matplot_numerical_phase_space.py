import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def calculate_numerical(atoms, spring_constant, eq_len, time):
    """ Calculates the Newtonian using a form of Euler's Method. Returns
    the first atoms force, position, and velocity.

    :param atoms: [description]
    :type atoms: [type]
    :param spring_constant: [description]
    :type spring_constant: [type]
    :param eq_len: [description]
    :type eq_len: [type]
    :param time: [description]
    :type time: [type]
    :return: [description]
    :rtype: [type]
    """
    # atoms should be R3 position vectors
    # spring constant and eq_len are constants
    # time is an array of incremented time values. Assume dt is constant.
    pos1 = atoms[0][0]
    pos2 = atoms[0][1]

    # arrays to be returned
    spring_force = []
    velocity1 = [(pos1*0)] # have zero vectors with the same format
    velocity2 = [(pos2*0)]
    position1 = [pos1]
    position2 = [pos2]
    # key variables
    mass = 1.0 # assume mass = 1
    dt = time[1] - time[0] # time step
    # save previous values
    old_pos1, old_pos2 = pos1, pos2
    old_vel1, old_vel2 = pos1*0, pos2*0 # assume no initial velocity
    # find initial spring force
    spr_len = np.linalg.norm(old_pos2 - old_pos1)
    # assume spr_dir_hat must be constant.
    spr_dir_hat = (old_pos2 - old_pos1) / spr_len # vector is towards pos2
    spr_force = spring_constant * (spr_len - eq_len) * (-1)*spr_dir_hat
    spring_force.append(spr_force)
        
    for _ in time[1::1]:
        # recalculate spring force from previous positions
        spr_len = np.linalg.norm((old_pos2 - old_pos1))
        spr_dir_hat = (old_pos2 - old_pos1) / spr_len # vector is towards pos2
        spr_force = spring_constant * (spr_len - eq_len) * (-1)*spr_dir_hat
        
        # Calculate the force contribution from the other atoms.
        cont_spring_force = []
        for cont_next_pos1, cont_next_pos2 in atoms[1::1]:
            cont_spr_len = np.linalg.norm(np.array(cont_next_pos1) - np.array(cont_next_pos2))
            cont_spr_dir_hat = (np.array(cont_next_pos1) - np.array(cont_next_pos2))/cont_spr_len
            cont_spr_force = spring_constant * (cont_spr_len - eq_len) * (+1)*cont_spr_dir_hat
            cont_spring_force.append(cont_spr_force)
        
        # Calculate the force on the first atom due to the second + all other connected atoms.
        total_force = np.array(spr_force) + np.array(cont_spring_force)
        spring_force = np.append(spring_force, total_force)
        
        # recalculate velocity from old spring force 
        new_vel1 = old_vel1 + (-1)*spr_force / mass * dt
        new_vel2 = old_vel2 + (+1)*spr_force / mass * dt 
        velocity1.append(new_vel1)
        velocity2.append(new_vel2)
        
        # recalculate new position from old velocity 
        new_pos1 = old_pos1 + new_vel1 * dt
        new_pos2 = old_pos2 + new_vel2 * dt
        position1.append(new_pos1)
        position2.append(new_pos2)
        
        # update old vars 
        old_vel1, old_vel2 = new_vel1, new_vel2
        old_pos1, old_pos2 = new_pos1, new_pos2
    
    return spring_force, velocity1, position1, spring_constant

def produce_graphs(numerical, time):
    # Change font, and size.
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams.update({
        'font.size': 14,
    })
    
    fig = plt.figure(figsize=(15,15))
    gp = GridSpec(3,1, fig, hspace=0.4)
    
    # Set preliminary information
    time_label = "Time [s]"
    ax00 = plt.subplot(gp[0])
    ax00.set_title('Numerical Solution of Phase Space for One-Hundred Atom Chain')
    ax00.set_ylabel('Position [m]')
    ax00.set_xlabel("Speed [m/s]")
    ax01 = plt.subplot(gp[1])
    ax01.set_title('Numerical Solution of Velocity for One-Hundred Atom Chain')
    ax01.set_ylabel('Speed [m/s]')
    ax01.set_xlabel(time_label)
    ax02 = plt.subplot(gp[2])
    ax02.set_title('Numerical Solution of Position for One-Hundred Atom Chain')
    ax02.set_ylabel('Displacement [m]')
    ax02.set_xlabel(time_label)
    
    label_str = "Spring Constant: {} [N/m]"
    # Numerical Solution first.
    for _, vel, pos, scon in numerical:
        pos = np.array([p[0] for p in pos])
        
        vel = np.array([v[0] for v in vel])
        
        ax00.plot(vel, pos,
        label=label_str.format(scon))
        ax01.plot(time, vel, 
        label=label_str.format(scon))
        ax02.plot(time, pos, 
        label=label_str.format(scon))
    
    plt.legend(loc='upper left')
    plt.show()

### Main ###

# Define time interval
t = list(np.arange(0, 20, 0.1))
# Define the atoms. Create a 100 body linear chain seperated by 1 angstrom each.
a = 1.0
atoms = [([-2*a, 0, 0], [0, 0, 0])]

for i in range(1, 100):
    atoms.append(([i*a, 0, 0], [(i+1)*a, 0, 0]))

atoms = np.array(atoms)
eq_len = np.linalg.norm(atoms[0][0] - atoms[0][1])

# Displace the first atom.
atoms[0][0] = [-2*a, a+0.1, 0]

numerical = [
         calculate_numerical(atoms, 0.1, eq_len, t)
        ,calculate_numerical(atoms, 0.3, eq_len, t)
        ,calculate_numerical(atoms, 0.5, eq_len, t)
]

produce_graphs(numerical, t)