Kete Solar System Survey Tools
==============================

Kete is a collection of tools which enable the simulation of ground or spaced based
surveys of minor planets. Included here are:

   - Orbit propagation code capable of accurately calculating the orbits for many
     thousands of minor planets over decades on a laptop.
   - Orbit determination and fitting, including initial orbit determination (IOD),
     batch least-squares differential correction, and MCMC posterior sampling.
   - Thermal modeling, including NEATM (Near Earth Asteroid Thermal Model) and the FRM
     (Fast Rotator Model) for asteroids. Including support for non-spherical asteroids.
   - Optical modeling.
   - Interfaces to IPAC's IRSA, JPL's Horizons, and the Minor Planet Centers web tools.
   - Multi-core support for reading SPICE SPK files (SPICE itself is not used).
   - Geometric calculations to check if an object is visible from an observer's
     position.

Many of these tools are independent from one another, and can be combined in many ways
to answer questions. There are a collection of worked examples below which demonstrate
some of the capabilities.

It is recommended that the background reading section be covered before using kete,
as it is very important to understand the coordinate frames and reference times used.
Conversions such as UTC to TDB time can result in over a minute of difference in the
position of minor planets, which for fast moving objects is long enough that they may
move out of frame.

.. toctree::
   :maxdepth: 1

   code_structure
   installation
   auto_examples/getting_started
   tutorials/index
   auto_examples/index
   api/api

