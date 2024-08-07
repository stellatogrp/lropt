
Learning for Optimization under Uncertainty
=====================

LROPT is a package for decision making under uncertainty. It is based on Python, and built on top of [CVXPY](https://www.cvxpy.org). It allows you to add uncertainty to your optimization problem **(add sentence for learning here)**. 

## Example

The following code solves the robust knapsack problem where there are $n$ items, $\mathbf{x}$ are the binary decision variables, their values are denoted by $\mathbf{c}$, and their weights $\mathbf{w}$ belong to a box uncertainty set.

LROPT is not a solver. It relies upon the open source solvers 
[Clarabel](https://github.com/oxfordcontrol/Clarabel.rs), 
[ECOS](https://github.com/embotech/ecos), [SCS](https://github.com/bodono/scs-python),
and [OSQP](https://github.com/oxfordcontrol/osqp).

**(need to add additional solvers)**

## Contributing

If you'd like to add a new example to our library, or implement a new feature,
please get in touch with us first to make sure that your priorities align with
ours. 

## Team
LROPT is  built from the contributions of many
researchers and engineers. A list of people who contributed to the development of LROPT include Irina Wang, Amit Solomon, Bartolomeo Stellato, Cole Becker, Bart Van Parys and Manav Jairam. 

## Citing

