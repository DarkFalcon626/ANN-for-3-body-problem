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
import pylab as plt
import random as rn
import pickle as pck
import os


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
        k = RK4Coef(f, x[i], h)
        l = RK4Coef(g, v[i], h)
        
        ## Update the next values
        v[i+1] = v[i] + (h/6)*(k[0]+2*k[1]+2*k[2]+k[3])
        x[i+1] = x[i] + (h/6)*(l[0]+2*l[1]+2*l[2]+l[3])
                
    return x


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
    
    NE = 7.34767309e22/5.9722e24       ## ratio between the moon and earth
    RE = 6.3781e6/3.84e8               ## Nondimensionalized radius of the earth
    RM = 1.74e6/3.84e8                 ## Nondimensionalized radius of the moon
    
    ## Position the earth at the origin and the moon along the y axis.
    x2 = np.array([0.,0.])  
    x3 = np.array([0., 1])
    
    ## Newtons equation of gravity.
    if np.linalg.norm(x-x2) <= RE or np.linalg.norm(x-x3) <= RM:
        ## If the test mass enters the radius of either the earth or moon then
        ## set the acceleration to zero.
        a = np.array([0., 0.])  
    else:
        a = (x2-x)/(np.linalg.norm(x2-x)**3)+NE*(x3-x)/(np.linalg.norm(x3-x)**3)
    
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


def plotSystem(x):
    '''
    Produces a plot of a meteorite traveling throught the earth moon system.

    Parameters
    ----------
    x : Numpy array
        The trajectory of the meteorite.

    Effects
    -------
    Produces plot.
    
    Returns
    -------
    None.
    '''
    
    x = np.transpose(x,(1,0))
    
    plt.axes()
    
    earth = plt.Circle((0,0), 6.3781e6/3.84e8, fc='red')
    moon = plt.Circle((0,1), 1.74e6/3.84e8, fc='blue')
    
    plt.plot(x[0][0],x[1][0], "go")
    plt.plot(x[0],x[1],"g",label='meteor path')
    
    plt.gca().add_patch(earth)
    plt.gca().add_patch(moon)
    
    plt.axis('square')
    plt.show()
    
    
def generateData(plot=False):
    '''
    Generates random values for the starting position and velocity for a meteor
    then uses the RK4 method to solve for the trajectory of the particle. The
    solutions and the inital conditions are then serielized as a pickle file in
    a new folder for each data set generated.
    
    Effects
    -------
    Create a new folder to store the initial conditions and trajectors.
    Create file for the initial conditions.
    Create file for the trajectors.

    Returns
    -------
    values : Numpy array
        The initial conditions.
    Data : Numpy array
        The trajectories from the initial conditions.
    '''
    
    n = 3             # The number of different trajectories to generate.
    T = 2              # The nondimensionalized period.
    dt = 0.000001      # The time step value.
    
    ## An array for each time step values.
    t = np.arange(0, T+dt, dt)
    
    ## Initialize arrays to store the inital conditions.
    x0 = np.zeros((n, 2), float)
    v0 = np.zeros((n, 2), float)
    
    ## Generate random inital conditions.
    for i in range(n):
        x = rn.uniform(0.05, 1)
        y = rn.uniform(-1.2, 1.2)
        
        x0[i] = np.array([x,y])      # Store the position values.
        
        dx = rn.uniform(-0.5, 0.5)
        dy = rn.uniform(-0.5, 0.5)
        
        v0[i] = np.array([dx,dy])    # Store the velocity values.
    
    ## Initialize an array to store the trajectory values.
    Data = np.zeros((n, t.size+1, 2))
    print('Initial conditions generated')
    print(x0)
    print(v0)
    
    ## Solve for the trajectory of each initial conidtions.
    for i in range(n):
        Data[i] = Integator(dudt, dxdt, x0[i], v0[i], T, dt)
        print('Solution number {} has been solved'.format(i+1))
        if plot: # If plotting is turned on show plots of new data
            plotSystem(Data[i])
    
    ## Determine if a folder for the solution already exists.
    i = 0
    DatasetExists = True
    while DatasetExists:
        DatasetPath = os.getcwd()+'\\Datasets\\Dataset' + str(i)
        
        DatasetExists = os.path.exists(DatasetPath)
        
        i += 1
    
    ## When we find one that doesn't exist create a new one.
    os.makedirs(DatasetPath)
    
    ## Open a new text file for the trajectorys called Targets.
    with open(DatasetPath+'\\Targets.pkl', 'wb') as data:
        pck.dump(Data, data)
    
    data.close()
    
    ## Store both the positions and velocities in same array
    values = np.array([x0,v0])     
    
    ## Open a new text file for the initial conditions.
    with open(DatasetPath+'\\Values.pkl','wb') as val:
        pck.dump(values,val)
    
    val.close()
    
    return values, Data
        
    
    
    
    
    
