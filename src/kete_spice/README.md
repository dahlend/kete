# kete_spice

SPICE kernel I/O for Kete.

This crate provides:
- Readers for SPK, PCK, CK and SCLK kernels, frames kernels (FK), text PCK kernels,
  and instrument kernels (IK)
- DAF file writing, and repacking of SPK files into type 2 or type 13 segments
- `SpiceEphemeris`: the loaded kernels as a `kete_core` ephemeris (body states, and
  body-frame orientations resolved through the frames kernels), which the N-body
  propagation and FOV visibility checks in `kete_core` take
