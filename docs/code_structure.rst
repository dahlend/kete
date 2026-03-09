Code Organization
=================

Goals of Kete
-------------
Kete is a collection of tools for calculating the orbits and expected fluxes for minor
planets for the purpose of estimating which objects are visible in current, past, or
future sky surveys.  Specifically the goal is that these calculations may be performed
on the full set of all known asteroids in a reasonable amount of time on a laptop. 

Propagation
~~~~~~~~~~~
There are a number of existing tools which enable the propagation of orbits for
many millions or billions of years, but these typically are designed for a comparatively
small number of objects (perhaps 10-100 thousand), whereas kete's design intent is
10-100 million objects over the course of decades. The total compute approximately
scales like::

    computation time ~ (number of objects) x (length of time being simulated)

However solving the large number of objects for a relatively short length of time can be
optimized differently than a small number of objects for Giga-years.

Flux Estimation
~~~~~~~~~~~~~~~
After propagation has been performed, it is important to estimate the total expected
flux of the objects from the point of view of an observer. There are a number of models
for this available, including the Near Earth Asteroid Thermal Model (NEATM), the Fast
Rotator Model (FRM), and the HG-Magnitude system which is common for asteroids in the
visible band.

SPICE Kernel Interoperability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SPICE is a commonly used software package which has been in development for several
decades at this point, and is often used to keep track of the ephemeris of planets,
satellites (both natural and artificial), asteroids, and comets. Essentially the motion
of anything in the Solar System can be encoded in some flavor of SPICE kernel. The
primary downside of using cSPICE is that there is no native support for multi-core cpu
queries (an artifact of the age of the code). Kete has native multi-core support for
the majority of all commonly used SPICE kernels.


High level Structure
--------------------

Why Rust?
~~~~~~~~~
Given the goals listed above, it is clear that performance plays a key roll in design
considerations of the tool. As a result of this, the Rust language was chosen as the
primary language for a majority of the business logic. A Python "wrapper" was written on
top of the Rust core, allowing users to call the compiled Rust code from Python. This
wrapper lowers the barrier to entry for users doing data-analysis or simulations without
having to learn the underlying Rust.

Rust has a number of advantages over existing languages, it's performance is typically
comparable to C++/C, however due to the structure of the language it is less prone to
the most common type of errors to do with memory allocation and management. In addition
to this, Rust has excellent native multi-core support, especially for embarrassingly
parallel problems such as the orbit propagation required for kete.

Kete Core
~~~~~~~~~
The Rust core of the library, which does the underlying orbit and flux calculations is
written entirely without any reference to Python. This core part is available as
`kete_core`, and programming can be done entirely within Rust for tools which do not
require the Python wrapper. This design was chosen so that systems tools which would
benefit from orbit computation can be written without having to have Python installed.
It is important to note that if performance is a concern, then removing the Python is an
important step to get the maximum possible performance.

Kete Fitting
~~~~~~~~~~~~
The `kete_fitting` crate contains orbit determination and fitting algorithms, all
written in Rust without reference to Python. This includes:

- **Initial Orbit Determination (IOD)** -- Statistical ranging over observation
  pairs followed by two-body scoring to produce candidate orbits from short arcs.
- **Differential Correction** -- Batch least-squares with Levenberg--Marquardt
  damping, progressive arc expansion, and optional chi-squared outlier rejection.
- **NUTS MCMC Sampling** -- No-U-Turn Sampler for posterior orbit characterization
  on short arcs where the Gaussian approximation breaks down.
- **Lambert Solver** -- Universal-variable Stumpff-function solver for single-
  revolution Keplerian transfers.
- **Observation Types** -- Optical (RA/Dec), radar range, and radar range-rate.

Like `kete_core`, this crate can be used from pure Rust without Python.

Core Python Wrapper
~~~~~~~~~~~~~~~~~~~
The Rust libraries described above then have Python wrappers written over them,
allowing users to call these compiled tools inside of Python. In order to do this,
some boiler-plate code is required to glue these independent parts together. This is
where the `rust` folder inside of kete comes from. Inside of this folder there are
rust files which are mostly a one-to-one mapping to their respective counterparts
inside of `kete_core` and `kete_fitting`. Ideally there should be no 'business' logic
contained within these wrappers, and they should largely exist to provide convenient
mappings from the Python concepts to the Rust internal organization.

Kete Python
~~~~~~~~~~~
The remaining part of the code which is strictly Python is mostly quality of life
functions, plotting, and web query code. There is little to no mathematics or physics
contained within the Python.