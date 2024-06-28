# -*- coding: utf-8 -*-
"""
3-body ANN solver.
tester functions.
-----------------------------------
Author: Andrew Francey
-----------------------------------
Date: 05/01/24
"""

import numpy as np
import Integrator

## Run tests for the solver function in the Integrator module.
## -----------------------------------------------------------

## Test 1 conditions
x_initial_t1 = np.array([-0.3,0.2])
v_initial_t1 = np.zeros((3,2),float)
dt_t1 = 0.00001 
period_t1 = 3

x_t1, v_t1, t_t1 = Integrator.solver(x_initial_t1,v_initial_t1,period_t1,dt_t1)


## Test 2 conditions
x_initial_t2 = np.array([-0.7071,0.7071])
v_initial_t2 = np.zeros((3,2),float)
dt_t2 = 0.00001 
period_t2 = 3

x_t2, v_t2, t_t2 = Integrator.solver(x_initial_t2,v_initial_t2,period_t2,dt_t2)


Integrator.plot_x(x_t1)
Integrator.plot_x(x_t2)