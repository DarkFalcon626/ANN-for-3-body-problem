# -*- coding: utf-8 -*-
"""
3-body ANN solver.
3-body INtegrator.
-----------------------
Author: Andrew Francey
-----------------------
Data: 07/01/24
"""

import numpy as np
import pylab as pl
import random as rn
import scipy.constants as sc


def Integator(f, g, x0, v0, period, h):
    '''
    Produces the values for the position, velocity and time of an equation of
    motion using the 4th order Runge-Kutta method to solve a second order ODE,
    decomposed into two first order ODEs f and g, this is done using the 
    initial position x0 and velocity v0. The time stamps are computed from a
    time step h over the period.

    Parameters
    ----------
    f : Function
        Equation definiting the motion of the partical.
    g : Function
        Eqaution describing the change in the position of the partical.
    x0 : Numpy array
        Initial position of the partical.
    v0 : Numpy array
        Initial velocity of the partical.
    period : Float
        Time period to compute the solution over.
    h : Float
        The time step.

    Returns
    -------
    Tripletof Numpy array
        The position, velocity and time marks.

    '''
    
    def RK4Coef(f, x, h):
        '''
        Determines the Runge-Kutta coefficents for a differental equation f at
        the value of x with a time step of h.

        Parameters
        ----------
        f : Function
            Equation for the ODE.
        x : Numpy array
            Value to be inputted into the ODE.
        h : Float
            The time step.

        Returns
        -------
        k : Numpy array
            The 4 coefficents for the RK4 method.

        '''
        
        k1 = f(x)
        k2 = f(x + 0.5*h*k1)
        k3 = f(x + 0.5*h*k2)
        k4 = f(x + h*k3)
        
        ## Combine into an array.
        k = np.array([k1,k2,k3,k4])
        
        return k
    
    ## Create an array with the time steps.
    t = np.arange(0, period + h, h)
    
    ## Determine how many steps are in the solution.
    n = t.size
    
    ## Create arrays to store the position and velocity values.
    x = np.zeros((n+1,2), float)
    v = np.zeros((n+1,2), float)
    
    ## Input the intial conditions.
    x[0] = x0
    v[0] = v0
    
    ## Loop over all values in the time domain.
    for i in range(n):
        
        ## Compute the coefficents
        k = RK4Coef(f, x[i], v[i], h)
        l = RK4Coef(g, v[i], h)
        
        ## Update the next values
        v[i+1] = v[i] + (h/6)*(k[0]+2*k[1]+2*k[2]+k[3])
        x[i+1] = x[i] + (h/6)*(l[0]+2*l[1]+2*l[2]+l[3])
        
    return x, v, t


def dudt(x):
    '''
    Produces the acceleration on a test particle entering the earth and moon 
    gravitational system.

    Parameters
    ----------
    x : Numpy array
        Position of the test particle.

    Returns
    -------
    a : Numpy array
        The acceleration of the test particle.

    '''
    
    ME = 5.972e24        ## Mass of the earth
    MM = 7.34767309e22   ## Mass of the moon
    d = 3.844e8          ## Distance between the earth and moon
    
    ## Position the earth at the origin and the moon along the y axis.
    x2 = np.array([0.,0.])  
    x3 = np.array([0., d])
    
    ## Newtons equation of gravity.
    a = sc.G*(ME*(x2-x1)/(np.linalg.norm(x2-x1)**3)+M[1]*(x3-x1)/(np.linalg.norm(x3-x1)**3))
    
    return a

def dxdt(v):
    '''
    Produces the velocity of the test particle.

    Parameters
    ----------
    v : Numpy array
        The velocity of the test particle.

    Returns
    -------
    v : Numpy array
        The velocity of the test particle.

    '''

    return v 




