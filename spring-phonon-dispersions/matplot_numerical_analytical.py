import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def calculate_numerical(atoms, spring_constant, eq_len, time):
    # atoms should be R3 position vectors
    # spring constant and eq_len are constants
    # time is an array of incremented time values. Assume dt is constant.
    
    pos1, pos2 = atoms
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
    spr_len = np.sqrt(np.sum((old_pos2 - old_pos1)**2))
    # assume spr_dir_hat must be constant.
    spr_dir_hat = (old_pos2 - old_pos1) / spr_len # vector is towards pos2
    spr_force = spring_constant * (spr_len - eq_len) * (-1)*spr_dir_hat
    spring_force.append(spr_force)
        
    for t in time[1::1]:
        # recalculate spring force from previous positions
        spr_len = np.linalg.norm((old_pos2 - old_pos1))#np.sqrt(np.sum((old_pos2 - old_pos1)**2))
        spr_dir_hat = (old_pos2 - old_pos1) / spr_len # vector is towards pos2
        spr_force = spring_constant * (spr_len - eq_len) * (-1)*spr_dir_hat
        spring_force.append(spr_force)
        
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

def calculate_analytical(atoms, spring_constant, eq_len, time):
    one_pos, two_pos = atoms
    pos = []
    velocity = []
    spring_force = []
    # Constants
    mass = 1.0
    force_dir = (one_pos - two_pos)/np.linalg.norm(one_pos - two_pos)#/np.sqrt(np.sum(one_pos**2 + two_pos**2))
    k_over_m_sqrt = np.sqrt(spring_constant/(mass))
    two_k_over_m_sqrt = np.sqrt(2*spring_constant/(mass))
    omega = k_over_m_sqrt
    phase = 0.
    
    for t in time:
        pos.append(
            (1/2*(np.linalg.norm(one_pos-two_pos) - eq_len) * np.cos(np.sqrt(2)*omega*t) * one_pos/np.linalg.norm(one_pos) + one_pos - 1/4
        ))
        
        velocity.append(
            -5/7*((np.linalg.norm(one_pos-two_pos) - eq_len) * omega * np.sin(np.sqrt(2)*omega*t) * one_pos/np.linalg.norm(one_pos)
        ))
        
        spring_force.append(
            ((np.linalg.norm(one_pos-two_pos) - eq_len) * omega**2 * np.cos(np.sqrt(2)*omega*t) * one_pos/np.linalg.norm(one_pos)
        ))
        
    return spring_force, velocity, pos, spring_constant

def produce_graphs(numerical, analytical, time):
    # Change font, and size.
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams.update({
        'font.size': 14,
    })
    
    fig = plt.figure(figsize=(15,15))
    gp = GridSpec(3,2, fig, hspace=0.4)
    
    # Set preliminary information
    time_label = "Time [s]"
    ax00 = plt.subplot(gp[0])
    ax00.set_title('Numerical Solution of Force for Two Body Spring')
    ax00.set_ylabel('Force [N]')
    ax00.set_xlabel(time_label)
    ax01 = plt.subplot(gp[2])
    ax01.set_title('Numerical Solution of Velocity for Two Body Spring')
    ax01.set_ylabel('Speed [m/s]')
    ax01.set_xlabel(time_label)
    ax02 = plt.subplot(gp[4])
    ax02.set_title('Numerical Solution of Position for Two Body Spring')
    ax02.set_ylabel('Displacement [m]')
    ax02.set_xlabel(time_label)
    
    ax10 = plt.subplot(gp[1]) 
    ax10.set_title('Analytical Solution of Force for Two Body Spring')
    ax10.set_ylabel('Force [N]')
    ax10.set_xlabel(time_label)
    ax11 = plt.subplot(gp[3])
    ax11.set_title('Analytical Solution of Velocity for Two Body Spring')
    ax11.set_ylabel('Speed [m/s]')
    ax11.set_xlabel(time_label)
    ax12 = plt.subplot(gp[5])
    ax12.set_title('Analytical Solution of Position for Two Body Spring')
    ax12.set_ylabel('Displacement [m]')
    ax12.set_xlabel('Time [s]')
    
    # Numerical Solution first.
    for force, vel, pos, scon in numerical:
        ax00.plot(time, [f[0] for f in force], 
        label="Spring Constant: {} [N/m]".format(scon))
        ax01.plot(time, [v[0] for v in vel], 
        label="Spring Constant: {} [N/m]".format(scon))
        ax02.plot(time, [p[0] for p in pos], 
        label="Spring Constant: {} [N/m]".format(scon))
    
    # Analytical Solution second.
    for force, vel, pos, scon in analytical:
        ax10.plot(time, [f[0] for f in force], 
        label="Spring Constant: {} [N/m]".format(scon))
        ax11.plot(time, [v[0] for v in vel], 
        label="Spring Constant: {} [N/m]".format(scon))
        ax12.plot(time, [p[0] for p in pos], 
        label="Spring Constant: {} [N/m]".format(scon))
    
    plt.legend(bbox_to_anchor=(0.65, 4.45), loc='upper left')
    plt.show()

# Main
# first atoms
a = 1.0
atoms = (np.array([a, 0., 0.]), np.array([-a, 0., 0.]))
t = list(np.arange(0, 10, 0.01))
# Create three conditions
#print(calculate_analytical(atoms, 1.0, 2.0, t))
analytical = [
     calculate_analytical(atoms, 0.1, 1.5, t)
    ,calculate_analytical(atoms, 0.3, 1.5, t)
    ,calculate_analytical(atoms, 0.5, 1.5, t)
]

numerical = [
         calculate_numerical(atoms, 0.1, 1.5, t)
        ,calculate_numerical(atoms, 0.3, 1.5, t)
        ,calculate_numerical(atoms, 0.5, 1.5, t)
]

produce_graphs(numerical, analytical, t)