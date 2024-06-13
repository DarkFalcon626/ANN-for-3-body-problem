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
h_t1 = 0.2 
period_t1 = 2

solver_t1 = Integrator.solver(x_initial_t1,v_initial_t1,period_t1,h_t1)

print(solver_t1)

## Test 2 conditions
x_initial_t2 = np.array([-0.7071,0.7071])
v_initial_t2 = np.zeros((3,2),float)
h_t2 = 0.2
period_t2 = 2

solver_t2 = Integrator.solver(x_initial_t2,v_initial_t2,period_t2,h_t2)

print(solver_t2)


Integrator.plot_x(solver_t1)
Integrator.plot_x(solver_t2)