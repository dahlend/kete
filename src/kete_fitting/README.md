# kete_fitting

Orbit determination and fitting tools for the Kete solar system survey simulator.

This crate provides batch least-squares differential correction using chained
state transition matrix (STM) propagation, initial orbit determination (IOD)
methods (Gauss, Laplace), and observation types (optical RA/Dec, radar range,
radar range-rate).
