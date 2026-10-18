# kete_linking

Observation linking for the Kete solar system survey simulator.

This crate groups detections of unknown moving objects into candidate objects
and confirms them with orbit fits, for any observer, observing geometry, and
cadence. The main entry points are:

- `search`, which links and confirms observations in passes. The default
  settings, for main belt surveys from the ground, run two: slow objects first
  on a coarse grid of hypotheses, then the rest on a finer one.
- `search_windows`, which runs the same passes over a whole survey one tile of
  the sky at a time, optionally in windows of time.

Both need SPICE kernels loaded through `kete_spice`: the Sun at the time of
every observation, and the planets and the Moon over the span of the
observations, for the orbit fits which confirm the candidates.

The building blocks they use are public too. The crate documentation describes
each of them and the design.
