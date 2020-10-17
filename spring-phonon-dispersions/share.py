import math
import numpy as np
    
import time 

def factorial(n):
    total = n
    for i in range(1, n):
        total *= i
    if (n==0):
        return 1
    return total

def choose(n,k):
    """Standard Choose function.
    
    :param n: The total sample size.
    :type n: int
    :param k: The number of elements you're choosing.
    :type k: int
    :return: n choose k
    :rtype: int
    """
    return (math.factorial(n)/(math.factorial(k)*math.factorial(n-k)))

def probability(p, n, k):
    """Binomial probability function.
    
    :param p: Odds of success. (0.5 for a coin flip)
    :type p: float
    :param n: total sample size.
    :type n: int
    :param k: The number of elements you're choosing.
    :type k: int
    :return: The probability of the inputs.
    :rtype: float
    """
    return choose(n, k)*p**(n-k)*(1-p)**k

def nth_derivative(f, x, n):
    """Calculates the nth derivative using Newton's difference quotient method.
    
    :param f: input function.
    :type f: lambda
    :param x: Particular value at which f is evaluated.
    :type x: float
    :param n: The particular derivative requested.
    :type n: int
    :return: The nth derivative of f(x) at the given x.
    :rtype: float
    """
    h = 10e-2
    out_h = 1/(h**n)
    out = 0
    for k in range(0, n+1):
        out += (-1)**(k+n)*choose(n,k)*f(x +k*h)
    return out_h*out

def rectangular_integral(f, xrange, intervals):
    """Standard Riemann sum integral function.
    
    :param f: Input function to calculate the integral of.
    :type f: lambda
    :param xrange: List of the begin and end points for x.
    :type xrange: List
    :param intervals: How many rectangles.
    :type intervals: int
    :return: The value of the integral between the xrange.
    :rtype: float
    """
    int_out = 0
    delta_x = (max(xrange)-min(xrange))/intervals
    new_xrange = np.linspace(min(xrange), max(xrange), intervals)
    for x in new_xrange:
        int_out += f(x)
    return delta_x*int_out

def trapezoid_integral(f, xrange, intervals):
    """Calculates an integral using the trapezoidal rule for integration.
    
    :param f: Function to evaluate the integral on.
    :type f: lambda
    :param xrange: The range on which to evaluate f(x)
    :type xrange: list
    :param intervals: The total number of subdivisions.
    :type intervals: int
    :return: The value for the integral at x.
    :rtype: float
    """
    
    a, b = min(xrange), max(xrange)
    delta_x = (b-a)/intervals
    x = np.arange(1, intervals)
    
    int_out = f(a)
    int_out += f(b)
    int_out += sum(2*f(a+x*delta_x))
    
    return delta_x/2*int_out

def maclaurin_expansion(f, x, N):
    sum = f(0)
    for i in range(1, N+1):
        sum += nth_derivative(f, 0, i)/factorial(i)*x**i
    return sum
    
def create_graph_from_lambda(f, xrange):
    """Takes a function as an input with a specific interval xrange then creates a list with the output
    y-points. Inefficient, but useful if f is a simple math function not involving NumPy.
    
    :param f: The function to evaluate.
    :type f: lambda
    :param xrange: The interval on which f(x) is evaluated.
    :type xrange: list
    :return: The list of f(x) points for all x in xrange.
    :rtype: list of floats
    """
    out = []
    for x in xrange:
        out.append(f(x))
    return out

def create_derivative_graph(f, xrange, n):
    """Takes a function as an input with a specific interval xrange, then creates a list with the ouput
    y-points for the nth derivative of f.
    
    :param f: Input function that we wish to take the derivative of.
    :type f: lambda
    :param xrange: The interval on which to evaluate f^n(x).
    :type xrange: list
    :param n: The derivative (1st, 2nd, 3rd, etc)
    :type n: int
    :return: A list of all f^n(x) points for all x in xrange.
    :rtype: list of floats
    """
    plot_points = []
    for x in xrange:
        plot_points.append(nth_derivative(f, x, n))
    return plot_points

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

def eulers_cromer_method(f, dx, y, yp, range, cromer=True):
    """ The Eulers method, and Eulers-Cromer method for second order
    differential equations. 
    
    :param f: Second order differential equation to approximate the solution for.
    :type f: lambda
    :param dx: Step size. Smaller is better.
    :type dx: float
    :param y: The initial value of y given by initial condition.
    :type y: int, float
    :param yp: The initial value of y' given by initial condition.
    :type yp: int, float
    :param range: List containing the beginning and end points for our domain.
    :type range: list
    :param cromer: Use Cromer method or just regular Eulers, defaults to True
    :type cromer: bool, optional
    :return: Returns a tuple for the x coordinates corresponding to a set of y coordinates,
    which approximate the solution to f. 
    :rtype: list
    """
    x=min(range)
    y_space = [y]
    x_space = [x]
    while x<=max(range):
        if cromer:
            yp += f(x,y,yp)*dx
            y += yp*dx
        else:
            y += yp*dx
            yp += f(x,y,yp)*dx
        
        x += dx
        x_space.append(x)
        y_space.append(y)
    return (x_space, y_space)

def eulers_richardson_method(f, dx, y, yp, range, return_yp = False):
    """ The Eulers Richardson method for solving a differential equation of second
    order. Works by taking the Euler method, but uses the average values instead.
    This produces a better approximation to the solution faster (in theory) than
    just the plain Eulers method.
    
    :param f: The input math function derivative whom to approximate its solution
    :type f: lambda
    :param dx: The step size to use. Smaller is better.
    :type dx: float
    :param y: The initial value of y given by initial condition.
    :type y: float, int
    :param yp: The initial value of y' given by initial condition.
    :type yp: float, int
    :param range: A list which specifies the beginning and the ending of our domain.
    :type range: list
    :return: Returns a tuple for the x coordinates corresponding to a set of y coordinates,
    which approximate the solution to f.
    :rtype: list
    """
    x       = min(range)
    y_space = [y]
    yp_space = [yp]
    x_space = [x]
    
    while x<=max(range):
        yp_mid  = yp + 1/2*f(x,y,yp)*dx
        y_mid   = y  + 1/2*yp*dx
        ypp_mid = f(1/2*x*dx, y_mid, yp_mid)
        yp      += ypp_mid*dx
        y       += yp_mid*dx
        
        x       += dx
        x_space.append(x)
        y_space.append(y)
        yp_space.append(yp)
    if (return_yp):
        return (x_space, y_space, yp_space)
    return (x_space, y_space)

def euler_richardson_method_2system_ode2(f, g, dt, x, y, xp, yp, range):
    """ The Euler-Richardson method working on a two-coupled system. Required
    for chapter 6, problem 5 in order to express two coupled parameterized
    functions as a single output y(x).
    
    :param f: The first second order diffeq expressed as a lambda.
    :type f: lambda
    :param g: The second second order diffeq expressed as a lambda.
    :type g: lambda
    :param dt: Step size. Smaller is better.
    :type dt: float
    :param x: The initial condition for x.
    :type x: float,int
    :param y: The initial condition for y.
    :type y: float,int
    :param xp: The initial condition for xp.
    :type xp: float,int
    :param yp: The initial condition for yp.
    :type yp: float,int
    :param range: A list which specifies the beginning and the ending of our domain.
    :type range: list
    :return: Returns a tuple for the t,x,y,xp,yp values as lists.
    :rtype: 5-tuple(list)
    """
    # f = x'' and g = y''
    # both requires (t, x, y, x', y')
    # get initial conditions and setup arrays
    t = min(range)
    t_space = [t]
    x_space = [x]
    y_space = [y]
    xp_space = [xp]
    yp_space = [yp]
    
    while t <= max(range):
        # find get midpoints
        t_mid = t + (1/2)*dt
        xp_mid = xp + 1/2*f(t, x, y, xp, yp)*dt
        yp_mid = yp + 1/2*g(t, x, y, xp, yp)*dt
        x_mid = x + (1/2)*xp*dt
        y_mid = y + (1/2)*yp*dt
        
        # get slopes
        xp_s = f(t_mid, x_mid, y_mid, xp_mid, yp_mid)
        yp_s = g(t_mid, x_mid, y_mid, xp_mid, yp_mid)
        x_s = xp_mid
        y_s = yp_mid
        
        # update values
        t += dt
        x += x_s*dt
        y += y_s*dt
        xp += xp_s*dt
        yp += yp_s*dt

        # append values
        t_space.append(t)
        x_space.append(x)
        xp_space.append(xp)
        y_space.append(y)
        yp_space.append(yp)
    
    
    return (t_space, x_space, y_space, xp_space, yp_space)
    

def rk2_first_order_method(f, y, dx, range):
    """ Runge-Kutta 2 method for a first order differential 
    equation.
    
    :param f: Input first order derivative to appromixate.
    :type f: lambda
    :param y: The initial value given for y.
    :type y: float, int
    :param dx: Step size. Smaller is better.
    :type dx: float
    :param range: A list which specifies the beginning and the ending of our domain.
    :type range: list
    :return: Returns a tuple for the x coordinates corresponding to a set of y coordinates,
    which approximate the solution to f.
    :rtype: tuple(list, list)
    """
    x = min(range)
    
    x_space = [x]
    y_space = [y]
    
    while x<=max(range):
        yp_mid = f(x+1/2*dx, y + 1/2*dx*f(x,y))
        y += yp_mid*dx
        
        x += dx
        x_space.append(x)
        y_space.append(y)
    return (x_space, y_space)

def rk4_first_order_method(f, y, dx, range):
    """Runge-Kutta 4 method for a first order differential 
    equation.
    
    :param f: Input first order derivative to appromixate.
    :type f: lambda
    :param y: The initial value given for y.
    :type y: float, int
    :param dx: Step size. Smaller is better.
    :type dx: float
    :param range: A list which specifies the beginning and the ending of our domain.
    :type range: list
    :return: Returns a tuple for the x coordinates corresponding to a set of y coordinates,
    which approximate the solution to f.
    :rtype: list
    """
    x = min(range)
    
    x_space = [x]
    y_space = [y]
    
    while x<=max(range):
        k_1 = f(x, y)*dx
        
        k_2 = f(x+1/2*dx, y + 1/2*k_1)*dx
        
        k_3 = f(x+1/2*dx, y + 1/2*k_2)*dx
        
        k_4 = f(x + dx, y + k_3)*dx
        
        y   += 1/6*(k_1+2*(k_2+k_3)+k_4)
        
        x += dx
        x_space.append(x)
        y_space.append(y)
    return (x_space, y_space)

def rk4_second_order_method(f, y, z, dx, range):
    """Runge-Kutta 4 method for a second order differential 
    equation.
    
    :param f: Input first order derivative to appromixate.
    :type f: lambda
    :param y: The initial value given for y.
    :type y: float, int
    :param z: The initial value given for z.
    :type z: float, int
    :param dx: Step size. Smaller is better.
    :type dx: float
    :param range: A list which specifies the beginning and the ending of our domain.
    :type range: list
    :return: Returns a tuple for the x coordinates corresponding to a set of y coordinates,
    which approximate the solution to f.
    :rtype: list
    """
    x = min(range)
    
    x_space = [x]
    y_space = [y]
    z_space = []
    
    while x<=max(range):
        k_1 = z*dx
        l_1 = f(x, y, z)*dx
        
        k_2 = (z+1/2*l_1)*dx
        l_2 = f(x+1/2*dx, y + 1/2*k_1, z + 1/2*l_1)*dx
        
        k_3 = (z + 1/2*l_2)*dx
        l_3 = f(x+1/2*dx, y + 1/2*k_2, z + 1/2*l_2)*dx
        
        k_4 = (z + l_3)*dx
        l_4 = f(x + dx, y + k_3, z + l_3)*dx
        
        y += 1/6*(k_1+2*k_2+2*k_3+k_4)
        z += 1/6*(l_1+2*l_2+2*l_3+l_4)
        
        x += dx
        x_space.append(x)
        y_space.append(y)
        z_space.append(z)
    return (x_space, y_space, z_space)
    

def build_extracted_list(input_list, subinterval):
    """ A utility function to extract a number of elements from a list, leaving only a certain subset.
    Generates a new list with just the subset. Creates the subset by specifying a sub-interval.
    
    :param input_list: The list to be extracted
    :type input_list: list
    :param subinterval: How many other elements to keep (for example, 10 means keep every 10 elements).
    :type subinterval: int
    :return: The extracted list.
    :rtype: list
    """
    out = []
    wait = subinterval
    for i in input_list:
        if wait == subinterval:
            out.append(i)
            wait = 0
        else:
            wait += 1
    return out

def newtons_law_of_cooling(room_temp, object_temp, t, k):
    return room_temp + (object_temp-room_temp)*math.exp(-k*t)

def diffusion_monte_carlo(num_of_gas, time_total, t_min=None):
    """ Generates a list of points to which a gas of num_of_gas
    molecules diffuses through a membrane and into a separate
    container within a total time total_time.

    :param num_of_gas: The amount of gas.
    :type num_of_gas: int
    :param time_total: The total time to run the simulation for.
    :type time_total: int
    :param t_min: Start at t=0 or not, defaults to None
    :type t_min: int, float, optional
    :return: Returns a list of diffusion of molecules per
    time.
    :rtype: tuple(time, molecules)
    """
    if t_min:
        time_array  = np.linspace(t_min, time_total, time_total+1, dtype=int)
    else:
        time_array  = np.linspace(1, time_total, time_total+1, dtype=int)
        
    gas_left    = np.array([num_of_gas])
    random_gas  = np.random.randint(1, num_of_gas+1, size=time_total)
    
    current_gas = num_of_gas
    
    for x in random_gas:
        if x <= current_gas:
            current_gas -= 1
        else:
            current_gas += 1
        gas_left = np.append(gas_left, current_gas)
    return (time_array, gas_left)

def nuclear_decay_monte_carlo(initial_amount, probability, t_max, ret_half=False):
    """ Generates a list of points to which can be graphed, tracing out the
    nuclear decay of an element given an initial amount initial_amount, probability,
    a maximum time interval.

    :param initial_amount: The starting mass.
    :type initial_amount: int, float
    :param probability: The likelyhood for decay to occur.
    :type probability: float 0 to 1.
    :param t_max: The maximum time to go for.
    :type t_max: int
    :param ret_half: Should half-life be returned or not, defaults to False.
    :type ret_half: bool, optional
    :return: Returns the amount of nuclei decayed per time and the half-life
    if specified by the optional variable.
    :rtype: tuple(time, nuclei, [optional] half-life)
    """
    particle_array = np.arange(1, initial_amount+1, dtype=int)
    
    nuclei_left    = np.array([initial_amount+1])
    
    half_life      = 0
    
    # Array math.
    for t in range(0, t_max):
        random_array   = np.random.uniform(0, 1, size=particle_array.size)
        random_bool    = np.where(random_array <= probability, True, False)
        particle_array = particle_array[random_bool != True]
        nuclei_left    = np.append(nuclei_left, particle_array.size)
        if not half_life and particle_array.size <= initial_amount/2:
            # Average.
            half_life  = (2*t-1)/2
       
    time_array = np.linspace(1, nuclei_left.size, nuclei_left.size, dtype=int)
    if ret_half:
        return (time_array, nuclei_left, half_life)
    return (time_array, nuclei_left)

def virus_monte_carlo(initial_infected, population, k):
    """ Generates a list of points to which some is infected
    at a given value k starting with initial_infected infected.
    There is no mechanism to stop the infection from reaching
    the entire population.

    :param initial_infected: The amount of people whom are infected at the
    start.
    :type initial_infected: int
    :param population: The total population sample.
    :type population: int
    :param k: The rate of infection.
    :type k: float
    :return: An array of the amount of people per time infected.
    :rtype: tuple(time, infected)
    """
    people_array    = np.arange(1, population+1, dtype=int)
    current_infected = initial_infected
    people_infected = np.array([current_infected])
    time_array      = np.array([0])
    
    # Array math.
    counter = 0
    for _ in people_array:
        probability      = (k)*current_infected/population
        random_array     = np.random.uniform(0, 1, size=people_array.size)
        random_bool      = np.where(random_array <= probability, True, False)
        people_array     = people_array[random_bool != True]
        if people_array.size != population:
            current_infected = (population-people_array.size)
        people_infected  = np.append(people_infected, current_infected)
        counter+=1
        time_array = np.append(time_array, counter)
        if people_infected.size == population:
            break
        
    return (time_array, people_infected)

def random_walk(n, p):
    """ Based on the number of times n, with a probability
    to move to the right p, this function calculates the mean
    of the end value along a one-dimensional axis to which it
    'walked'.

    :param n: Number of steps to take.
    :type n: int
    :param p: Probability from 0 to 1
    :type p: float
    :return: The expected value on the number line to which we
    end up.
    :rtype: float
    """
    random_array = np.random.uniform(0, 1, n)
    left = random_array[random_array > p].size
    right = n - left
    
    return (right-left)

def monte_carlo_integration(f, n, a, b, ret_arrays=False):
    """ Calculate the integral of a function f
    from a to b with a number of random points n. Tried to
    optimize by removing for loops in favor of NumPy arrays with
    conditional indexing due to their cache efficiency vs Python
    list pointer dereferences.
    Fastest speed I've gotten is about 130ms.

    :param f: The input math function.
    :type f: lambda
    :param n: The total number of points.
    :type n: int
    :param a: Starting position.
    :type a: int, float
    :param b: Ending position.
    :type b: int, float
    :param ret_arrays: Return arrays in order to graph, defaults to False.
    :type ret_arrays: Boolean, defaults to False.
    :return: The estimated value of the integral of f.
    :rtype: float
    """
    x = np.random.uniform(0, 1, n)*(b-a)+a
    f_array = f(x)

    positive_x = x[f_array >= 0]
    negative_x = x[f_array < 0]
    if positive_x.size > 0:
        h = np.max(f_array)
    else:
        h = np.max(-f_array)
    
    y_positive = np.random.uniform(0, 1, positive_x.size)*h
    y_negative = np.random.uniform(0, 1, negative_x.size)*h
    
    xy_indices_below = y_positive <= f(positive_x)
    xy_indices_above = y_negative <= -f(negative_x)
    n_inside_below = y_positive[xy_indices_below]
    n_inside_above = -y_negative[xy_indices_above]
    
    if ret_arrays:
        n_inside_x = np.append(positive_x[xy_indices_below],negative_x[xy_indices_above])
        n_inside_y = np.append(n_inside_below, n_inside_above)
        return n_inside_x, n_inside_y
    
    return h*(b-a)*(n_inside_below.size-n_inside_above.size)/(n)