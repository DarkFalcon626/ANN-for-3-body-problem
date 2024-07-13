# Solving the 3 body gravitational problem using ANN 

## Overview

  This projects objection is to build and train an artificial neural network to solve the famous 3 body gravitational problem. Due to the chaotic nature of the 3 body problem no close form solution exists to the problem. With the power of computers came aproximate solutions using methods such as the Runge-Kutta methods for solving systems of ODEs, now with the raise of neural networks we again make advancements in solving the 3 body problem.  


  This projects focus is on a restricted version of the 3 body problem in which two of the bodies are static and the motion of the third much smaller body is what we aim to solve for.  We set the problem up to model an asteroid traveling through or entering the earth and moon system.

## Set Up of the Problem
We set up our problem with the earth at the origin of the system (i.e (0,0)) with a mass of $`M_{E}=5.9722*10^{24}`$ and the moon resting on the y-axis at a distance of $`d_{m} = 3.84*10^{8}`$ with a mass of $`M_{M}=7.35*10^{22}`$. 

The problem is described by the second order differential equation for the acceleration of the asteroid,
```math
\frac{d^{2}\vec{x}}{dt^{2}} = G\biggl(\frac{M_{E}}{||\vec{x}||^{3}}\vec{x}+\frac{M_{M}}{||\vec{x}-\vec{x_{M}}||^{3}}(\vec{x}-\vec{x_{M}})\biggr)
```
This equation can be turned into two first order differential equations by making the substatution $`\vec{u}=\frac{d\vec{x}}{dt}`$, this gives us the following,
```math
\frac{d\vec{u}}{dt} = G\biggl(\frac{M_{E}}{||\vec{x}||^{3}}\vec{x}+\frac{M_{M}}{||\vec{x}-\vec{x_{M}}||^{3}}(\vec{x}-\vec{x_{M}})\biggr)
```
```math
\frac{d\vec{x}}{dt} = \vec{u}
```
