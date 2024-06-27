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

def solver2D(x0, v0, period, dt, epsilon):
    
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
            a[i] = (x[i-1]-x[i])*((np.linalg.norm(x[i-1]-x[i]))**(-3))
        
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
    
    def RKF4Coef(f, x, h):
        '''
        Produces the coefficents for use in the Runge-Kutta-Fehlberg method 
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
            The 6 coefficents for the RKF4(5) method.
        '''
        
        ## The coefficents for the RKF5(5) method.
        B = np.array([[0., 0., 0., 0., 0.],
                      [2/9, 0., 0., 0., 0.],
                      [1/12, 1/4, 0., 0., 0.],
                      [69/128, -243/128, 135/64, 0., 0.], 
                      [-17/12, 27/4, -27/5, 16/15, 0.], 
                      [65/432, -5/16, 13/16, 4/27, 5/144]])
        
        ## Initialize the array to store the computed coeffiecents.
        k = np.zeros((6,x.shape[0],x.shape[1]),float)
        
        y = x
        ## loop over all the coefficents
        for i in range(6):
            j = 0 # Counter to keep just the lower triangle of the B matrix.
            while j < i:
                y += B[i][j]*k[j] ## Add the updated value to the x value.
                j += 1 ## Update the counter.
            
            ## Compute the coefficent with the updated x value.
            k[i] = h*f(y)
        
        return k 
    
    ## Check that the values inputed meet the criteria.
    if period <= 0:
        raise Exception("The period must be greater then zero")
    elif dt <= 0:
        raise Exception("The time step dt must be greater then zero")
    
    x = np.array([[x0,0],
                  [-x0,0]])
    v = v0
    
    ## Initialize the lists to store the positions and velocities
    x_lst = [np.array([[x0,0.],[-x0,0.]])] 
    v_lst = [v0]
    t_lst = [0.]
    
    ## Coefficents for the RK4(5) method.
    CH = np.array([47/450, 0., 12/25, 32/225, 1/30, 6/25])
    CT = np.array([1/150, 0., -3/100, 16/75, 1/20, -6/25])
    
    ## Initalize the starting time and indexing counter.
    t = 0.
    i = 0 
    
    while t < period:
        
        k = RKF4Coef(grav, v, dt)
        l = RKF4Coef(dxdt, x, dt)
        
        x_new = x
        v_new = v
        
        for j in range(6):
            v_new += CH[j]*k[j]
            x_new += CH[j]*l[j]
        
        t_new = t + dt
        kTE = abs(sum(CT[i]*k[i] for i in range(6)))
        lTE = abs(sum(CT[i]*l[i] for i in range(6)))
        
        TE = max(np.amax(kTE), np.amax(lTE))
        
        dt = 0.9*dt*(epsilon/TE)**(1/5)
        
        if TE <= epsilon:
            x_lst.append(x_new)
            v_lst.append(v_new)
            t_lst.append(t_new)
        
            x = x_new
            v = v_new
            t = t_new
            
            i += 1
        
    return x_lst, v_lst, t_lst
    
    
    
    
    