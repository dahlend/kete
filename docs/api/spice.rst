SPICE
=====

This module is a thread-safe reader and writer of SPICE kernels. It loads SPICE
kernel files directly into memory.

.. note::

   This does not use cSPICE, or any original SPICE library code.

   cSPICE is difficult to use in a thread safe manner which is limiting when
   performing orbit calculations on millions of objects.

Data files which are automatically downloaded:

DE440 - A SPICE file containing the planets within a few hundred years.

BSP files are also automatically downloaded for the 5 largest asteroids, which are
used for numerical integrations when the correct flags are set.

PCK Files which enable coordinate transformations between Earths surface and the
common inertial frames.

.. automodule:: kete.spice
   :members:
   :inherited-members:
