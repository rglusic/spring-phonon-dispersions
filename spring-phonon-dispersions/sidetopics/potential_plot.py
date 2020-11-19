import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

number_of_points = 100
extreme_position = 50
y_label = r'$U(\theta)$'
x_label = r'$\theta$'
axislabel_fontsize = 16
tickmark_fontsize = 14
whitespace_factor = 0.02

plt.rcParams.update({
    "font.family": "serif"})

thetas = np.linspace(-extreme_position, extreme_position,
                        num=number_of_points)

def eulers_method(f, y, dx, range):
    """ The Eulers method for a first order differential equation.
    
    :param f: First order differential equation to approximate the solution for.
    :type f: lambda
    :param y: The initial condition for the y-value.
    :type y: float, int
    :param dx: Step size. Smaller is better.
    :type dx: float
    :param range: List containing the beginning and end points for our domain.
    :type range: list
    :return: Returns a tuple for the x coordinates corresponding to a set of y coordinates,
    which approximate the solution to f. 
    :rtype: list
    """
    x = min(range)
    y_space = [y]
    x_space = [x]
    while x<=max(range):
        y += f(x, y)*dx
        x += dx
        x_space.append(x)
        y_space.append(y)
    return (x_space, y_space)

def pendulum_potential(theta, mass, L, g=9.8):
    return mass*g*L*(1-np.cos(np.radians(theta)))

def pendulum_theta(total_energy, mass, L, g=9.8):
    return 1-np.arccos(total_energy/(mass*g*L))

figures, axes = plt.subplots(ncols=1)

potential = pendulum_potential(thetas, 1.0, 1.0)

axes.set_xlim(1.05*-extreme_position, 1.05*extreme_position)
axes.set_ylim(1.05*np.min(potential), 1.05*np.max(potential))
axes.xaxis.set_label_coords(1.+whitespace_factor, 0.80)
axes.plot(thetas, potential)

axes.hlines(pendulum_theta(9.8, 1, 1), 1.05*-extreme_position, 1.08*extreme_position, color='gray')
axes.hlines(pendulum_theta(0, 1, 1), 1.05*-extreme_position, 1.08*extreme_position, color='gray')

# dt(theta)
def time_pos(potential, mass=1.0, total_energy=np.max(potential)):
    return np.sqrt(mass/(2*(total_energy-potential)))

# t(theta)
time_points, ypoints = eulers_method(time_pos, 0.001, 0.01, [-30, 30])
axes.plot(time_points, ypoints)

axes.text(-35, 2.5, r"$t = \int\sqrt{\frac{m}{2\cdot(E-U(\theta))}}$")
axes.set_xlabel(x_label, verticalalignment='center', horizontalalignment='left', fontsize=axislabel_fontsize)
axes.set_ylabel(y_label, rotation='horizontal', fontsize=axislabel_fontsize)
axes.yaxis.set_label_coords(0., 1.+whitespace_factor)

axes.plot(0, 1, ls="", marker="^", ms=5, color="k",
          transform=axes.get_xaxis_transform(), clip_on=False)
axes.plot(1, 0, ls="", marker=">", ms=5, color="k",
          transform=axes.get_yaxis_transform(), clip_on=False)

axes.tick_params(axis='y', labelsize=tickmark_fontsize)

axes.spines["right"].set_visible(False)
axes.spines["left"].set_position(("data", 0.))
axes.spines["top"].set_visible(False)
axes.spines["bottom"].set_position(("data", 0.))

plt.show()
