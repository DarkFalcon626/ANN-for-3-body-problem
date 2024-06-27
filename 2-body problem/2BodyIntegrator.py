# -*- coding: utf-8 -*-
"""
3-body ANN solver.
2-body Integrator
-------------------------
Author: Andrew Francey
-------------------------
Date: 05/12/24
-------------------------
Functions for solving the 2-body
gravitanal problem.
"""

import numpy as np 
import pylab as plt

def solver2B(x0, v0, period, dt):
    '''
    Produces arrays for the position, velocity, and the time steps for both 
    bodies in a two body system requiring the inital position of the first
    body along the x-axis x0 and the velocity vectors of both bodies v0. Also 
    requires the length of time period and the time step dt.

    Parameters
    ----------
    x0 : Float
        The inital position of the first body along the x-axis.
    v0 : Numpy array
        The x,y velocities of both bodies.
    period : Float
        The length of time to compute the results.
    dt : Float
        The time step to use in the computation.

    Raises
    ------
    Exception
        The initial position x0 must be between 0. and 1.
        The period must be greater then zero.
        The time step dt must be greater then zero.

    Returns
    -------
    Triplet of Numpy array
        The positions of both bodies. The velocities of both bodies and the 
        time marks.      
    '''
    
    def grav(x):
        '''
        Produces the acceleration matrix for both bodies from the gravatational
        force.

        Parameters
        ----------
        x : Numpy array
            The postion array of the two bodies.

        Returns
        -------
        a : Numpy array
            The acceleration matrix of the two bodies.
        '''
        
        ## Initialize the acceration matrix
        a = np.zeros((2,2),float)
        
        ## apply Newtons equation for gravity to both bodies
        for i in range(2):
            a[i] = 10*(x[i-1]-x[i])*((np.linalg.norm(x[i-1]-x[i]))**(-3))
        
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
        k2 = f(x+ 0.5*h*k1)
        k3 = f(x+ 0.5*h*k2)
        k4 = f(x+ h*k3)
        
        k = np.array([k1,k2,k3,k4])
        
        return k 
    
    ## Check that the values inputed meet the criteria.
    if period <= 0:
        raise Exception("The period must be greater then zero")
    elif dt <= 0:
        raise Exception("The time step dt must be greater then zero")
    elif x0 <= 0 or x0 > 1:
        raise Exception("The inital position x0 must be between 0 and 1")
    
    ## Initials the values for the loop
    x = np.array([[x0,0],
                  [-x0,0]])
    v = v0
    
    ## Initialize the lists to store the positions and velocities.
    x_lst = [np.array([[x0,0.],[-x0,0.]])] 
    v_lst = [v0]
    t_lst = [0.]
    
    ## Initalize the starting time and indexing counter.
    t = 0.
    
    ## compute all the time steps in the period
    while t < period:
        
        ## Compute the coefficents for the RK process.
        k = RK4Coef(grav, x, dt)
        l = RK4Coef(dxdt, v, dt)
        
        ## Compute the new velocites and positions of the bodies.
        v_new = v + (dt/6)*(k[0]+2*k[1]+2*k[2]+k[3])
        x_new = x + (dt/6)*(l[0]+2*l[1]+2*l[2]+l[3])
        
        ## Compute the next time step.
        t_new = t + dt
    
        ## Update and append the new values to their respective lists.
        t_lst.append(t_new)
        t = t_new
        
        v_lst.append(v_new)
        v = v_new
        
        x_lst.append(x_new)
        x = x_new
    
    ## Convert the lists to arrays.
    x_lst = np.array(x_lst)
    v_lst = np.array(v_lst)
    t_lst = np.array(t_lst)
    
    return x_lst, v_lst, t_lst
    

def plot2B(x):
    
    y = np.transpose(x,(1,2,0))
    
    plt.plot(y[0][0][0], y[0][1][0], 'ro')
    plt.plot(y[0][0], y[0][1], 'r')
    plt.plot(y[1][0][0],y[1][1][0], 'bo')
    plt.plot(y[1][0],y[1][1],'b')
    plt.show()
    
    
    
    
    