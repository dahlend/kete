# kete_spice

SPICE kernel I/O for Kete.

This crate provides:
- Readers for SPK, PCK, CK and SCLK kernels
- DAF file writing, and repacking of SPK files into type 2 or type 13 segments
- `SpiceEphemeris`: the loaded SPK, PCK and CK files as a `kete_core` ephemeris (body
  states and body-frame orientations), which the N-body propagation and FOV visibility
  checks in `kete_core` take
