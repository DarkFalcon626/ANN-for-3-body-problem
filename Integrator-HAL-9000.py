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


def solver(x_initial, v_intial, period, h):
    '''
    Using a Runge-Kutta-Fehlberg algorithm to solve for the position and velocities
    at a adaptive time step over the length of time period. given the intial
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
    h : Float
        Time step to update position and velocity of the bodies.

    Returns
    -------
    A numpy array with the positions and velocity of the 3 bodies.
    '''
    
    def center_of_mass(x1,x2):
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
    
    
    def dvdt(x,v):
        '''
        Produces the acceleration of all three bodies based on the gravitational
        force from the three bodies.

        Returns
        -------
        a : Numpy array.
            The acceleration of all three bodies in 2D.
        '''
        
        ## Initialize the acceration matrix
        a = np.zeros((3,2),float)
        
        ## Loop through and summing the acceleration from each body on each body.
        for i in range(3):
            for j in range(3):
                ## The below ensures no division of zero.
                if j != i or (x[j][0]-x[i][0]) != 0 or (x[j][1]-x[i][1]) != 0: 
                    a[i] += (abs((x[j]-x[i]))**(-3))*(x[j]-x[i])

        return a
    
    
    def dxdt(x,v):
        '''
        Produces the velocity of all three bodies.
        
        Returns
        -------
        v : Numpy array
            The velocity of all three bodies in 2D.
        '''
        return v
    
    def RK4(f,g,t,x,v,h):
        '''
        Produces the coefficents for the Runge-Kutta-Fehlberg method.

        Parameters
        ----------
        f : Function
            First differential equaiton.
        g : Funciton
            Second differential equaiton.
        t : float
            value of the time.
        x : numpy array
            The position array of the 3 bodies.
        v : numpy array
            The velocity array of the 3 bodies.
        h : float
            The time-step value.

        Returns
        -------
        k : Lstof floats
            list of the coefficents for the first differnetial equation.
        l : Lstof floats
            list of the coefficents for the second differential equation.
        '''
        
        ## Coefficents for the RK4(5) Formula 1.
        A = np.array([0, 2/9, 1/3, 3/4, 1, 5/6])
        
        B = np.array([[0, 0, 0, 0, 0],
                      [2/9, 0, 0, 0, 0], 
                      [1/12, 1/4, 0, 0, 0], 
                      [69/128, -243/128, 135/64, 0, 0],
                      [-17/12, 27/4, -27/5, 16/15, 0],
                      [65/432, -5/16, 13/16, 4/27, 5/144]])
        
        k = np.zeros((6,3,2))
        l = np.zeros((6,3,2))
        
        for i in range(6):
            for j in range(5):
                x += B[i][j]*k[j]
                v += B[i][j]*l[j]
            
            k[i] = h*f(x, v)
            l[i] = h*g(x, v)
        
        return k, l
    
    
    ## Check that the values inputed meet the criteria.
    if x_initial[0] > 0:
        raise Exception("The second body must be in the negative x domain.")
    elif x_initial[1] < 0:
        raise Exception("The second body must be in the positive y domain.")
    elif 0 >= x_initial[0]**2 + x_initial[1]**2 >= 1:
        raise Exception("The second body must be in the unit sphere.")
    elif period <= 0:
        raise Exception("The period must be greater then zero")
    elif h <= 0:
        raise Exception("The time step dt must be greater then zero")
    
    ## Determine how many data points will be calculated based on the time step
    ##  and the period.
    
    ## Coefficents for the RK4(5) mehtod.
    CH = np.array([47/450, 0, 12/25, 32/225, 1/30, 6/25])
    
    CT = np.array([1/150, 0, -3/100, 16/75, 1/20, -6/25])
    
    epsilon = 0.0001 
    
    x3 = center_of_mass(np.array([1.,0.]), x_initial)
    
    ## Initialize lists to store the position and velocity values
    x_lst = [np.array([[1.,0.],
                       [x_initial[0],x_initial[1]],
                       [x3[0],x3[1]]])]
    
    v_lst = [v_intial]
    
    t = 0 
    i = 0
    while t < period:
        
        k, l = RK4(dvdt, dxdt, t, x_lst[i], v_lst[i], h)
        
        v_new = v_lst[-1]+CH[0]*k[0]+CH[1]*k[1]+CH[2]*k[2]+CH[3]*k[3]+CH[4]*k[4]+CH[5]*k[5]
        x_new = x_lst[-1]+CH[0]*l[0]+CH[1]*l[1]+CH[2]*l[2]+CH[3]*l[3]+CH[4]*l[4]+CH[5]*l[5]
        
        TE = abs(CT[0]*l[0]+CT[1]*l[1]+CT[2]*l[2]+CT[3]*l[3]+CT[4]*l[4]+CT[5]*l[5])
        print(TE)
        
        h_new = 0.9*h*(epsilon/TE)**(1/5)
        
        if TE.any() <= epsilon:
            t += h
            x_lst.append(x_new)
            v_lst.append(v_new)
        
        h = h_new
            
    return x_lst
    

def plot_x(x):
    '''
    Plots the position of the 3 bodies in 2 demensions.

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
    
    
    
    
        
    