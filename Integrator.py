'''
3-body ANN solver.
3-body Integrator.
-----------------------
Author: Andrew Francey
-----------------------
Date: 05/01/24
'''

import numpy as np
import pylab as plt


def solver(x_initial, v_intial, period, dt):
    '''
    Using a velocity Verlet algorithm to solve for the position and velocities
    at a certien time step dt over the length of time period. given the intial
    position and velocities x_initial and v_inital respectively.
    
    The reference frame is centered at the center of mass and positions are 
    scaled with respect to the farthest body which will be body 1. All masses
    are 1 and the gravitalal constant G is 1.

    Parameters
    ----------
    x_initial : Numpy Array.
        The intial positions of the second body. The first mass being on the
        x-axis at (1,0) of the unit circle and the third being found
        from symmetry.
        
        Requires : (x,y) where 0 <= (x**2)+(y**2) <= 1, x <= 0, and y >= 0 .  
    v_intial : Numpy Array.
        The intial velocities of the 3 bodies.
    period : Float
        Length of time to run the model over.
    dt : Float
        Time step to update position and velocity of the bodies.

    Returns
    -------
    A numpy array with the positions and velocity of the 3 bodies.
    '''
    
    def grav(x):
        '''
        Produces the acceleration of all three bodies based on the gravitational
        force from the three bodies.

        Returns
        -------
        Numpy array.
            The acceleration of all three bodies in 2D.
        '''
        
        ## Initialize the acceration matrix
        a = np.zeros((3,2),float)
        
        ## Loop through and summing the acceleration from each body on each body.
        for i in range(3):
            for j in range(3):
                ## The below ensures no division of zero.
                if j != i or (x[j][0]-x[i][0]) != 0 or (x[j][1]-x[i][1]) != 0: 
                    a[i] += 10*(x[j]-x[i])*(np.linalg.norm(x[j]-x[i])**(-3))

        return a
    
    def dxdt(v):
        '''
        The eqaution for velocity of the system.

        Parameters
        ----------
        v : Numpy array
            The velocity matrix of the system.

        Returns
        -------
        v : Numpy array
            The velocity matrix of the system.
        '''

        return v
    
    def center_mass(x1,x2):
        '''
        Produces the position vector for the third body to keep the center of 
        mass at the origin (0,0).

        Parameters
        ----------
        x1 : Numpy array
            The position of the first body.
        x2 : Numpy array
            The position of the second body.

        Returns
        -------
        Numpy array
            The position of the third body.
        '''
        
        x3 = -(x1+x2)
        
        return x3
    
    def RK4Coef(f, x, h):
        '''
        Produces the coefficents for use in the Runge-Kutta method 
        to solve the function f given a value x with a time step h.

        Parameters
        ----------
        f : Function
            The ODE to solve. Must only take in one varible.
        x : Numpy array
            The data array to be inputted into the function.
        h : Float
            The timestep to be applied.

        Returns
        -------
        k : Numpy array
            The 4 coefficents for the RK4 method.
        '''
        
        k1 = f(x)
        k2 = f(x + 0.5*h*k1)
        k3 = f(x + 0.5*h*k2)
        k4 = f(x + h*k3)
        
        k = np.array([k1,k2,k3,k4])
        
        return k
    
    ## Check that the values inputed meet the criteria.
    if x_initial[0] > 0:
        raise Exception("The second body must be in the negative x domain.")
    elif x_initial[1] < 0:
        raise Exception("The second body must be in the positive y domain.")
    elif 0 >= x_initial[0]**2 + x_initial[1]**2 >= 1:
        raise Exception("The second body must be in the unit sphere.")
    elif period <= 0:
        raise Exception("The period must be greater then zero")
    elif dt <= 0:
        raise Exception("The time step dt must be greater then zero")
    
    ## Create the time array for all the time steps
    t = np.arange(0, period + dt, dt)
    
    ## Determine how many steps are in the solution.
    n = t.size
    
    ## Create arrays to store the position and velocity values
    x = np.zeros((n+1,3,2))
    v = np.zeros((n+1,3,2))
    
    ## Find the position of the 3rd mass to center at the center of mass
    x3 = center_mass(np.array([1., 0.]), x_initial)
    
    ## Input the intial conditions
    x[0] = np.array([[1.,0.],[x_initial[0],x_initial[1]],[x3[0],x3[1]]])
    v[0] = v_intial
    
    for i in range(n):
        
        ## Compute the coefficents
        k = RK4Coef(grav, x[i], dt)
        l = RK4Coef(dxdt, v[i], dt)
        
        ## Compute the new velocities and positions.
        v[i+1] = v[i] + (dt/6)*(k[0]+2*k[1]+2*k[2]+k[3])
        x[i+1] = x[i] + (dt/6)*(l[0]+2*l[1]+2*l[2]+l[3])
        
    return x, v, t
    
    

def plot_x(x):
    '''
    Plots the position of the 3 bodies in 2 dimensions.

    Parameters
    ----------
    x : Numpy Array
        The trajectory of the 3 bodies.

    Effects
    -------
    Creates image of the 3 bodies trajectories.
    
    Returns
    -------
    None.
    '''    
    
    ## Reshape the output from the solver so all the x axis values for each body
    ##  are in the same row and each body is a matrix.
    
    y = np.transpose(x,(1,2,0))
    
    plt.plot(y[0][0][0],y[0][1][0],"ro")
    plt.plot(y[0][0],y[0][1],"r")
    plt.plot(y[1][0][0],y[1][1][0],"bo")
    plt.plot(y[1][0],y[1][1],"b")
    plt.plot(y[2][0][0],y[2][1][0],"go")
    plt.plot(y[2][0],y[2][1],"g")
    plt.axis('square')
    plt.show()
    
    
    
    
        
    