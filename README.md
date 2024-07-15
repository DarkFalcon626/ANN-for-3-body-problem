# Solving the 3 body gravitational problem using ANN 


## Overview

  This projects objection is to build and train an artificial neural network to solve the famous 3 body gravitational problem. Due to the chaotic nature of the 3 body problem no close form solution exists to the problem. With the power of computers came aproximate solutions using methods such as the Runge-Kutta methods for solving systems of ODEs, now with the raise of neural networks we again make advancements in solving the 3 body problem.  


  This projects focus is on a restricted version of the 3 body problem in which two of the bodies are static and the motion of the third much smaller body is what we aim to solve for.  We set the problem up to model an asteroid traveling through or entering the earth and moon system.

## Set Up of the Problem
We set up our problem with the earth at the origin of the system (i.e (0,0)) with a mass of $`M_{E}=5.9722*10^{24}`$ and the moon resting on the y-axis at a distance of $`d_{m} = 3.84*10^{8}`$ with a mass of $`M_{M}=7.35*10^{22}`$. 

The problem is described by the second order differential equation for the acceleration of the asteroid,
```math
\frac{d^{2}\vec{x}}{dt^{2}} = G\biggl(\frac{m_{E}}{||\vec{x}||^{3}}\vec{x}+\frac{m_{M}}{||\vec{x}-\vec{x_{M}}||^{3}}(\vec{x}-\vec{x_{M}})\biggr)
```
This equation can be turned into two first order differential equations by making the substatution $`\vec{u}=\frac{d\vec{x}}{dt}`$, this gives us the following,
```math
\frac{d\vec{u}}{dt} = G\biggl(\frac{m_{E}}{||\vec{x}||^{3}}\vec{x}+\frac{m_{M}}{||\vec{x}-\vec{x_{M}}||^{3}}(\vec{x}-\vec{x_{M}})\biggr)
```
```math
\frac{d\vec{x}}{dt} = \vec{u}.
```
Next we nondimensionalize the above equations using the mass of earth as the characteristic mass and the distance between earth and the moon and the characteristic lenght. Using the characteristic mass, lenght and the gravitational constant we can find an expression for the characteristic time.
```math
T=\sqrt{\frac{L^{3}}{GM}}
```
Thus we can write out the following $`\vec{x}=L\vec{r}`, $`m_{E}=Mn_{E}`$, $`m_{M}=Mn_{M}`$, $`t=T\tau`$, and $`\vec{u}=(L/T)\vec{v}`$. Using this we get,
```math
\frac{GML}{L^{3}}\frac{d\vec{v}}{d\tau}=\frac{GML}{L^{3}}\biggl(\frac{\vec{r}}{||\vec{r}||^{3}}+\frac{n_{M}}{||\vec{r}-\vec{r}_{M}||^{3}}(\vec{r}-\vec{r}_{M})\biggr)
```
```math
\frac{L}{T}\frac{d\vec{r}}{d\tau}=\frac{L}{T}\vec{v}
```
This simplifies to,
```math
\frac{d\vec{v}}{d\tau}=\biggl(\frac{\vec{r}}{||\vec{r}||^{3}}+\frac{n_{M}}{||\vec{r}-\vec{r}_{M}||^{3}}(\vec{r}-\vec{r}_{M})\biggr)
```
```math
\frac{d\vec{r}}{d\tau}=\vec{v}
```
A solution to the above equation does not exist in a closed form solution, A Numeretical solution can be found using computer algorithms such as the 4th order Runge-Kutta method. These methods can be quite computationally expensive to run and only build solutions step by step not allowing us to put in a set of inputs (i.e $`\vec{x}`$) and get the solution (i.e $`\vec{y}`$) as we would be able to if we knew the function $`f(\vec{x})=\vec{y}`$, this is where our project comes into play.

Using the universal approximation theorem which is stated below, we can train a artifical neural network $`N(t,\vec{x}_{0}, \vec{v}_{0})`$ to approximate the solution to the above equations.

![UATScreenshot](/Graphics/UATScreenshot.png "The Universal Approximation Theorem")


## Creating The Network




