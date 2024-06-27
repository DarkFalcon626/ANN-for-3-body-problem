<h1 align=center>Solving the 3 body gravitational problem using ANN</h1>

<hr>
<h3>Overview</h3>
<p>
  This projects objection is to build and train an artificial neural network to solve the famous 3 body gravitational problem. Due to the chaotic nature of the 3 body problem no close form solution exists to the problem. With the power of computers came aproximate solutions using methods such as the Runge-Kutta methods for solving systems of ODEs, now with the raise of neural networks we again make advancements in solving the 3 body problem.  
</p>

<hr>

<h3>The Plan</h3>
<p>First we try to build a ANN to solve the 2-body problem. By trying the method on a less choatic system with a known solution we can test to see if the method will work for a simplier case before trying it on a more complex system.
  <ol>
    <li>
      We start the process by building a dataset of solutions by solving the differential equations using the velocity verlot method for solving equations of motion.
    </li>
    <li>
      Next we build an artifical neural network.
    </li>
  </ol>
</p>
