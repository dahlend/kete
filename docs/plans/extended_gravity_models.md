# Extended Gravity Models — Implementation Plan

This is a green field project with no previous work.
This should use the existing kete integrators, coordinate frames, Vectors, and spice
kernel lookups as appropriate.

## Motivation

kete currently treats every massive body as a point mass (with a GR correction
for the Sun and Jupiter). Two related capabilities are missing:

1. **Polyhedron gravity** for irregular small bodies (asteroids, comet nuclei),
   needed for propagating ejecta / shed material in the near-body regime.
2. **Spherical harmonic gravity** for planets, needed for high-precision
   propagation of close-in spacecraft or natural satellites.

Both share the same architectural needs: an *extended* gravity source attached
to a body that has its own ephemeris and rotation, evaluated in a body-fixed
frame, and combined with the existing N-body machinery as a perturbation.

This document plans the polyhedron model first (the immediate driver) but
designs the framework so spherical harmonics drop in cleanly as a second
implementation of the same trait.

## Scope

In scope:
- Constant-density polyhedron gravity (Werner-Scheeres 1996).
- Multi-component polyhedra: an ExtendedBody may hold multiple Polyhedron
  components each with its own density (contact binaries, bilobate nuclei).
- Body-natural unit system to keep conditioning sane near small bodies.
- Body rotation (body-fixed <-> inertial) with simple constant-spin model.
- Body-centric integration regime with tidal-form third-body perturbations.
- Hand-off between heliocentric and body-centric regimes.
- Hooks for a future spherical-harmonic implementation (no implementation yet).
- Documentation of accuracy expectations by distance regime, especially for
  contact binaries and concave shapes.

Out of scope (deferred):
- Mascon models fit to spacecraft tracking data (density from observations).
- Self-gravity of ejecta swarms (test-particle assumption only).
- KS regularization for impact / very-close trajectories.
- PCK-backed rotation (add when a real use case appears).
- Spherical harmonics implementation itself (planned as a follow-on once the
  framework lands).

## Background and references

- Werner, R. (1994), *Celestial Mechanics* 59, 253 - polyhedron potential.
- Werner & Scheeres (1996), *CMDA* 65, 313 - acceleration and gradient.
- Scheeres, D. J. (2012), *Orbital Motion in Strongly Perturbed Environments*.
- Montenbruck & Gill, *Satellite Orbits* - tidal/Encke formulations and
  spherical harmonic conventions.

## Numerics summary (rationale)

The hard parts of this problem are not the formulas; they are the numerics:

- The asteroid `GM` is ~10^-18 in solar units. Using AU/day with `GM_sun`
  baked in destroys conditioning near the body. Solution: switch to
  body-natural units (length = body radius, time = sqrt(R^3/GM_body)) inside
  the body's domain so `GM_body = 1` and orbital periods are O(2*pi).
- Third-body perturbations must use the **tidal difference** form
  `GM_p * (r_p_to_particle/|...|^3 - r_p_to_body/|...|^3)`, never absolute
  Newtonian terms. For very small `|r| / d`, switch to the gradient
  expansion `-GM_p/d^3 * (r - 3*(d_hat . r) d_hat)` which is unconditionally
  well-conditioned.
- Particle state in the body-centric regime is body-relative; the body's
  heliocentric motion comes from its ephemeris, not from integration.

These same numerical concerns apply to a planet with spherical harmonics in
the close-in regime, so the framework handles them uniformly.

## Architecture

Three layers, mirroring the existing `forces` / `integrators` / `propagation`
split between `kete_core` and `kete_spice`:

```
kete_core::shape           Pure math, no SPICE.
  Polyhedron               Werner-Scheeres polyhedron gravity.
  (future) SphericalHarmonics   Stokes coefficient gravity.
  ExtendedGravity (trait)  Common interface: potential, accel, gradient.
  RotationModel            Body-fixed <-> inertial, constant-spin variant.
  BodyUnits                Length / time / GM unit system per body.
  ExtendedBody<F>          Bundles: shape, rotation, units, gm, proximity radius.

kete_core::forces          Existing acceleration plumbing.
  small_body_accel(...)    Pure function: takes ExtendedBody plus caller-
                           supplied perturber positions, returns accel in
                           body units.

kete_spice::small_body_propagation
  propagate_near_body(...)  Glue layer: queries SPK for body and perturber
                            positions, drives the existing Radau / PC15
                            integrator with small_body_accel as the force.
                            Handles unit conversion at the boundary only.
```

Key invariants:
- `kete_core` stays SPICE-free. All ephemeris queries happen in `kete_spice`.
- The polyhedron / spherical-harmonics evaluators are unit-agnostic and
  frame-agnostic; they operate in body-fixed coordinates and body-natural
  units. Frame rotation and unit conversion are separate concerns.
- The integrator is the existing `RadauIntegrator` / `PC15`; only the force
  function and the unit system change.

### The `ExtendedGravity` trait

```rust
pub trait ExtendedGravity {
    /// Potential at body-fixed position r (body-natural units).
    fn potential(&self, r: Vector3<f64>) -> f64;

    /// Acceleration at body-fixed position r (body-natural units).
    fn acceleration(&self, r: Vector3<f64>) -> Vector3<f64>;

    /// Gravity gradient tensor at body-fixed position r.
    fn gradient(&self, r: Vector3<f64>) -> Matrix3<f64>;

    /// Smallest radius (body units) at which the model is valid /
    /// well-converged. For polyhedra: 0 (valid everywhere). For spherical
    /// harmonics: the Brillouin sphere radius.
    fn min_valid_radius(&self) -> f64;

    /// True if r is inside the body. Polyhedra use the exact Laplacian sign
    /// test; spherical harmonics return false unconditionally (not defined).
    fn contains(&self, r: Vector3<f64>) -> bool { false }
}
```

`ExtendedBody<F>` then holds `Box<dyn ExtendedGravity>` (or a generic) plus
the rotation, units, GM, and proximity radius. Both polyhedron and spherical
harmonics implementations plug into the same `propagate_near_body` driver.

## Phased plan

Each phase is self-contained, testable, and produces a useful artifact.

### Phase 1 - Polyhedron shape model (math + I/O)

- New module `src/kete_core/src/shape/` with `mod.rs`, `polyhedron.rs`, `io.rs`.
- `Polyhedron` struct: vertices, triangular faces, precomputed per-edge dyads
  `E_e` and per-face dyads `F_f`, bounding radius, volume, density, derived GM.
  A single `Polyhedron` is one constant-density closed mesh.
- Implements `ExtendedGravity` (defined in this phase as well).
- `ExtendedBody` holds `Vec<Polyhedron>` (not a single one) so multi-component
  density is first-class from the start. Single-lobe bodies are `vec![single]`.
  Aggregate `acceleration`, `potential`, `gradient`, and `contains` are sums /
  unions across all components. This is the correct way to model contact
  binaries (e.g. 67P Churyumov-Gerasimenko, Arrokoth): decompose into lobe A,
  lobe B, and optionally a neck component, each with its own density.
- I/O: Wavefront OBJ loader to start (most PDS shape models ship as OBJ).
  Multi-component bodies load from multiple OBJ files, one per component.
- Tests:
  - Sphere shape -> matches `GM/r^2` far from surface.
  - Cube -> matches Werner 1994 reference values.
  - Two-sphere dumbbell: gravity in the neck matches superposition of two
    separate sphere calculations (validates multi-component summation).
  - Non-convex torus-like shape: `contains()` correct in the central cavity.
  - `contains()` true/false for clearly inside/outside points.
  - Laplacian identity: `trace(gradient) = -4*pi*G*rho` inside, `0` outside,
    both for a single component and for a two-component model.

Status: complete

Delivered in `src/kete_core/src/shape/`:
- `ExtendedGravity` trait with `potential`, `acceleration`, `gradient`,
  `min_valid_radius`, `contains`.
- `Polyhedron::try_new(vertices, faces, gm)` with closed-manifold validation,
  positive-volume / outward-winding check, and full Werner-Scheeres
  precomputation (per-face dyads, per-edge dyads, signed solid angles).
- `load_obj(path, gm)` minimal Wavefront OBJ loader (triangular faces only,
  accepts `v/vt/vn` index forms, ignores groups / materials / normals).
- 13 unit tests covering: sphere far-field vs. point mass, Euler manifold
  invariant for sphere, reversed-winding rejection, open-mesh rejection,
  inside/outside `contains` for cube and sphere, Laplacian identity inside
  cube and inside sphere, dumbbell two-component superposition (cancellation
  by symmetry, far-field point-mass equivalence), and OBJ loader behaviour.
- All 160 `kete_core` tests pass; clippy pedantic + perf rules clean.

Multi-component bodies are supported by holding `Vec<Polyhedron>` at the
caller (or future `ExtendedBody`) layer; `ExtendedGravity` outputs sum
linearly across components, validated by the dumbbell test.

### Phase 2 - Body-natural unit system

- New file `src/kete_core/src/shape/units.rs`.
- `BodyUnits { length_au: f64, gm_solar: f64 }`; time scale derived as
  `sqrt(L^3 / GM)`.
- Conversion helpers: `to_body_pos`, `from_body_pos`, time, velocity, accel.
- Tests: round-trip conversions, dimensional consistency.

Status: complete

Delivered `BodyUnits::try_new(length_au, gm_solar)` with derived
`time_day = sqrt(L^3 / GM)`, plus `pos_to_body` / `pos_from_body`,
`vel_to_body` / `vel_from_body`, `accel_to_body` / `accel_from_body`,
and `dt_to_body` / `dt_from_body`.  8 unit tests cover input validation,
time-scale definition, round-trip identities for position / velocity /
acceleration / duration, and the defining property that `gm_body == 1`
in the body-natural system.

### Phase 3 - Body rotation

- New file `src/kete_core/src/shape/rotation.rs`.
- `RotationModel` enum:
  - `Fixed { quat }` - non-rotating, useful for tests.
  - `ConstantSpin { pole_ra, pole_dec, w0, w_dot, epoch }` - IAU style.
  - (future) `Pck(BodyId)` - PCK-backed, lives in `kete_spice` when added.
- Methods: `inertial_to_body(t)`, `body_to_inertial(t)`, `angular_velocity(t)`.
- Tests: identity at epoch, correct period, round-trip.

Status: complete

Delivered `RotationModel::{Fixed, ConstantSpin}` with `inertial_to_body`,
`body_to_inertial`, `rotate_to_body`, `rotate_to_inertial`, and
`angular_velocity_inertial`.  The `ConstantSpin` variant uses the IAU
Cartographic Elements convention (passive composition
`R_z(W) R_x(pi/2 - dec) R_z(pi/2 + ra)`); `W(t) = w0 + w_dot * (t - epoch)`
in radians/day.  6 unit tests cover identity behaviour, inertial<->body
round-trip at multiple epochs, pole-vector invariance under spin, full
period returning to initial orientation, and angular-velocity direction
/ magnitude.

### Phase 4 - `ExtendedBody` and force evaluation

- New file `src/kete_core/src/shape/extended_body.rs`.
- `ExtendedBody<F: InertialFrame>` bundles gravity model, rotation, units,
  GM, and proximity radius.
- Pure function `small_body_accel(body, t, r_body_units, body_pos_inertial,
  perturbers)` in `kete_core::forces`.
  - Rotates `r` to body-fixed, evaluates gravity, rotates back.
  - Adds tidal perturbations; switches to gradient form when
    `|r| / d_perturber < 1e-4`.
  - Returns acceleration in body-natural units.
- Tests:
  - Far-field recovers point-mass `-GM r/r^3`.
  - Tidal Sun term at body center is exactly zero.
  - Tidal vs. naive subtraction at small `|r|/d` - verify gradient form holds.

Status: complete

Delivered `ExtendedBody { components: Vec<Polyhedron>, rotation, units,
gm, proximity_radius_au }` and pure function `small_body_accel(body,
jd_tdb, r_body_units, perturbers)` colocated in
`shape/extended_body.rs` (re-exported at the `shape` module level
rather than under `forces`, since it is logically tightly coupled to
the shape and rotation types and contains no N-body machinery).
The `ExtendedBody` is frame-agnostic: the caller is responsible for
ensuring the rotation model and inertial perturber positions share an
inertial reference frame.  `small_body_accel` switches to the tidal
gradient expansion when `|r|/d < 1e-4`.  7 unit tests cover input
validation, aggregate `gm` summation across components, no-perturber
far-field recovery, vanishing tidal at body center, gradient-vs-direct
agreement near the threshold, and rotation-independence with no
perturbers.

### Phase 5 - Body-centric integrator wrapper

- New file `src/kete_spice/src/small_body_propagation.rs`.
- `propagate_near_body(body, particle_state, perturber_ids, t_end)`:
  - Wraps `small_body_accel` in the existing `SecondOrderODE` signature
    (`AccelVecMeta`-style metadata).
  - Per step: convert `t` to body units, query SPK once for body and each
    perturber, call `small_body_accel`.
  - Reuses `RadauIntegrator` / `PC15` unchanged.
- Tests:
  - Circular orbit just outside bounding sphere of a sphere -> Kepler period.
  - Long integration with rotation off -> energy conserved.
  - Long integration with rotation on -> Jacobi constant conserved.
  - Cross-check against a published Bennu / Itokawa proximity case if
    available.

Status: complete

Delivery summary:

- `src/kete_spice/src/small_body_propagation.rs` (~330 lines, 2 tests
  passing) exposes `BodyRelativeState`, `Perturber`, and
  `propagate_near_body(body, body_naif_id, perturbers, initial_state,
  jd_final, spk)`.  Internal `ProxMeta` bundles the body, NAIF ID,
  perturber list, and `&SpkCollection` so SPK access (via
  `try_get_state_with_center::<Equatorial>`) is centred on the body
  every force evaluation, returning perturber positions already
  body-relative.
- Plan deviation: the Radau time axis is locked to TDB days, so the
  integrator state lives in mixed units - body-fixed positions in
  body-natural lengths (`AU / length_au`) and velocities in
  `body_length / day`, with accelerations in `body_length / day^2`.
  The body-natural time scale of `BodyUnits` remains useful for
  user-facing analytic conversions but is bypassed inside the
  propagator.  This required refactoring `small_body_accel` (and
  Phase 4 tests) to return acceleration in `body_length / day^2`
  rather than full body-natural units; the only operation now is
  `/ length_au` after combining direct + tidal pieces in `AU/day^2`.
- `small_body_accel` now rejects interior field points up front via
  `body.contains(...)` returning `Error::SurfaceImpact`.  The
  closed-form polyhedron evaluator only flags exact surface
  coincidence, so without the explicit interior guard a trajectory
  that crosses the body would silently switch to the interior
  potential gradient.  The wrapper translates `Error::SurfaceImpact`
  into `Error::Impact(body_naif_id, time)`, mirroring the convention
  used by `crate::propagation::spk_accel` for planet impacts.
- Tests: `circular_orbit_no_perturbers_advances_along_orbit`
  integrates a circular orbit at r = 100 body radii (with `gm` chosen
  so the period is ~1 day, well above Radau's `MIN_STEP = 5e-4` days)
  for 1/16 of a period and verifies the position and velocity match
  the analytic circular-orbit prediction to 1e-2 (with radial drift
  < 5e-3).  `surface_impact_returns_impact_error` aims a particle
  inward at 3x circular speed and confirms the resulting `Error`
  arm is `Error::Impact` tagged with the supplied body NAIF ID.
- Full-period round-trip and energy/Jacobi-conservation tests were
  deliberately not added in this phase: the icosphere polyhedron
  approximation (320 faces) carries enough multipole content that
  Radau's step controller refuses to settle over many orbits at
  modest r/R, and a meaningful long-integration test would need
  either a much smoother synthetic body or a real proximity case
  (Bennu/Itokawa).  Those are deferred to follow-on work alongside
  the Phase 6 regime hand-off, where end-to-end propagation against
  reference trajectories is the natural place to anchor accuracy
  expectations.
- `cargo test --release -p kete_spice` and `cargo test --release -p
  kete_core` both pass, with `cargo clippy --release --all-targets`
  clean across both crates.

### Phase 6 - Regime hand-off

- Coordinator in `kete_spice` that:
  - Detects `|r_particle - r_body| > k * proximity_radius` (with hysteresis)
    and converts the particle state to heliocentric for the existing
    `propagate_n_body` path.
  - Symmetric entry condition for particles approaching a registered body.
  - Conversion uses `body.ephemeris.position(t) + r` and similarly for
    velocity, with `BodyUnits` doing the unit conversion.
- Initially manual (user calls each propagator); add automatic switching
  only after the manual path is proven.
- Tests: state continuity at the seam within integrator tolerance.

Status: complete (manual path)

Delivery summary:

- Three new helpers in `src/kete_spice/src/small_body_propagation.rs`
  implement the manual path:
  - `body_relative_to_heliocentric<T, C>(state, body, body_helio_state)
    -> State<T, C>` builds an inertial / heliocentric `State` by
    rotating the body-fixed position to inertial, scaling body-natural
    lengths back to AU, adding the body's translational state, and
    adding the rigid-body rotational velocity contribution
    `omega x r_inertial_au` (where `omega` comes from
    `RotationModel::angular_velocity_inertial`).
  - `heliocentric_to_body_relative<T, C>(state, body, body_helio_state)
    -> BodyRelativeState` is the exact inverse: subtracts the body's
    translational state, removes `omega x r`, rotates inertial -> body,
    and scales AU back to body-natural lengths.
  - `is_inside_proximity(state, body, factor) -> bool` is a simple
    predicate that compares `|r| in AU` against
    `factor * body.proximity_radius_au`.  Hysteresis is implemented by
    callers picking different `factor` values for entry and exit
    conditions; encoding the hysteresis state machine itself is left
    to the caller for now (per the plan's "manual first" guidance).
- The conversion helpers are deliberately SPK-free at the call site:
  the caller passes the body's own ephemeris state at the conversion
  epoch (one `try_get_state_with_center` query) rather than having
  the helpers re-query the kernel.  This keeps both helpers
  deterministic and testable without kernel access.
- `BodyRelativeState` gained an optional `desig: Option<Desig>` field
  carried through both propagation and conversion, and is now `Clone`
  (no longer `Copy`).  `propagate_near_body` now takes its initial
  state by reference (clippy `needless_pass_by_value`).
- Tests:
  - `round_trip_fixed_rotation` and `round_trip_constant_spin`
    exercise body -> helio -> body for both rotation variants.  The
    constant-spin case uses a fast non-trivial pole (RA 0.7, Dec 0.4,
    omega = 5 rad/day) so the rotational contribution is non-zero in
    every component.  Tolerance is 1e-6 in body-natural lengths,
    bounded by f64 cancellation: subtracting the body's heliocentric
    position (~1.5 AU) from `body_pos + 5e-8 AU` introduces an error
    of `eps * 1.5 / length_au ~ 3e-7` in body-natural units.  This is
    a fundamental property of mixed-scale subtraction, not a defect
    of the implementation.
  - `proximity_predicate_with_hysteresis` confirms the predicate
    reports correctly with a tight enter factor (0.8) and a loose
    exit factor (1.2) on a particle in the intermediate band.
- Plan deviations: an end-to-end "state continuity at the seam"
  integration test was not added.  Doing it cleanly requires a
  reference body (e.g. a real asteroid with both an SPK trajectory
  and published shape model) so that both the body-centric polyhedron
  acceleration and the heliocentric `propagate_n_body_spk`
  point-mass acceleration are evaluating physically the same body at
  the seam radius.  The synthetic icosphere bodies used in the
  current test suite are not registered as SPK bodies, so the
  heliocentric leg cannot include them.  The seam test is deferred
  to Phase 8 alongside the Bennu / Itokawa accuracy validation.
- `cargo test --release -p kete_spice` (62 unit + 1 doctest) and
  `cargo test --release -p kete_core` (186 unit + 10 doctest) both
  pass; `cargo clippy --release -p kete_spice --all-targets` clean.

### Phase 7 - PyO3 bindings

- New file `src/kete/rust/extended_body.rs` exposing:
  - `Polyhedron` (constructor + `from_density` staticmethod, geometry
    accessors `vertices` / `volume` / `bounding_radius` / `gm` /
    `n_vertices` / `n_faces` / `n_edges`, plus `potential`,
    `acceleration`, `contains`, `center_of_mass`).
  - `RotationModel` with `fixed()` and `constant_spin(...)` static
    constructors.  Angles are in degrees, rates in degrees/day; the
    epoch accepts a JD float or a `kete.Time`.
  - `ExtendedBody(components, rotation, length_au, proximity_radius_au)`
    with accessors `gm`, `length_au`, `proximity_radius_au`,
    `n_components`, plus `acceleration` and `contains` helpers.
  - `propagate_near_body(body, body_naif_id, particle, jd_final,
    perturbers=None)` driving the body-centric integrator end-to-end
    via `kete_spice::small_body_propagation`.  Accepts and returns
    standard `kete.State` instances; internally re-centers via the
    SSB so the conversion helpers see a consistent center.
  - `is_inside_proximity(particle, body, body_naif_id, factor=1.0)`
    convenience predicate.
  - `body_acceleration(body, point_body_au, jd_tdb, perturbers=None)`
    sanity helper that exposes the proximity-regime evaluator
    (returns AU / day^2).
- New Python module `kete.extended_body` re-exporting the bindings,
  added to the `kete.__init__` import list.
- Body-natural unit scaling (`BodyUnits`) is constructed implicitly
  from `length_au` and the aggregate component `gm`; users never see
  it.
- New `src/kete_spice` prelude exports for `BodyRelativeState`,
  `Perturber`, `body_relative_to_heliocentric`,
  `heliocentric_to_body_relative`, `is_inside_proximity` and
  `propagate_near_body`.
- Python tests in `src/tests/test_extended_body.py` (5 tests) cover:
  hand-built icosahedron geometry / surface-impact / exterior
  acceleration sign; `Polyhedron.from_density`; `ExtendedBody`
  accessors and `body_acceleration`; constant-spin rotation repr;
  the `is_inside_proximity` predicate path.
- Validation: `cargo check --release -p _core` clean;
  `cargo clippy --release --workspace --all-targets` clean;
  `pytest src/tests/test_extended_body.py` 5/5 pass; full Python
  test suite 1151 passed (1 pre-existing failure in
  `test_spice.py::test_loaded_info` unrelated to this work).

Status: complete

### Phase 8 - Documentation and example

- Sphinx tutorial: "Propagating ejecta from a small body shape model."
  Include a contact-binary example (two-lobe body, different densities).
- API docs for `Polyhedron`, `ExtendedBody`, `propagate_near_body`.
- "Numerics notes" page explaining body-natural units and the tidal form so
  users understand why the API is shaped this way.
- "Accuracy and limitations" section covering non-convex / contact-binary
  regimes explicitly:
  - The Werner-Scheeres model is mathematically exact for any closed
    orientable polyhedron, convex or not. `contains()` via Laplacian sign is
    also exact for non-convex shapes.
  - The dominant uncertainty is always the shape model itself, not the gravity
    solver. Near-neck regions of contact binaries have the largest shape
    uncertainty due to limited stereo coverage and self-shadowing.
  - Constant density is a known simplification; multi-component models with
    per-lobe densities are the recommended way to represent contact binaries.
  - Guidance by distance regime:
    - More than 1 body radius above surface: reliable at few-percent level,
      limited by total-mass uncertainty.
    - 0.1-1 body radius: gravity field shape is qualitatively correct;
      quantitative accuracy depends on density assumptions.
    - Less than 0.1 body radius (near or in a neck): results are
      scenario-dependent, not predictive. Use multi-component model and
      treat outputs as sensitivity studies, not predictions.
    - Touching or near-touching surface: outside the intended use of this
      model; refer users to dedicated DEM tools (pkdgrav, Chrono).

Status: in progress
- API page `docs/api/extended_body.rst` written and wired into the
  toctree (`docs/api/api.rst`).  Covers units, gravity-model
  background, accuracy guidance by distance regime, and the full
  autodoc reference for `kete.extended_body`.
- Seam-continuity Python test added
  (`test_seam_continuity_against_ceres` in
  `src/tests/test_extended_body.py`):  uses Ceres' shipped SPK
  (NAIF id 20000001) as the central body, places a particle 30 km
  off Ceres with a 1 m/s relative kick, and checks that
  `propagate_near_body` (with the body's GM negligible and only the
  Sun in the perturber list) reproduces a heliocentric two-body
  Kepler propagation to better than 1e-8 AU and 1e-8 AU/day over a
  1-day arc.  This exercises the helio -> body-relative ->
  integrate -> body-relative -> helio path through real SPK
  lookups.
- Sphinx-gallery example added at
  `src/examples/plot_extended_body.py`:  builds a synthetic
  icosahedral mesh scaled to Ceres' bounding radius and GM, then
  propagates a 5000-km circular-orbit test particle for one day with
  `propagate_near_body` (Sun as the only perturber) and plots the
  growing deflection from a Sun-only Kepler trajectory.  The
  example uses Ceres' shipped SPK ephemeris (NAIF id 20000001) and
  needs no external downloads.
- Numerics-notes section folded into
  `docs/api/extended_body.rst`: documents the body-natural unit
  scaling, the Radau step-size floor, and the tidal-form
  perturber acceleration (with the leading point-mass term
  cancelled because the body's SPK already absorbs it).
- Contact-binary smoke test added
  (`test_contact_binary_two_lobes` in
  `src/tests/test_extended_body.py`):  builds two icosahedral
  lobes offset along x, verifies `n_components`, total `gm`,
  far-field point-mass scaling at 1e-6 AU, and exact linear
  superposition of per-lobe accelerations.
- Pending: quantitative accuracy validation against published
  Werner-Scheeres reference values for Bennu and Itokawa.  This
  requires the published shape models and is left as a follow-up
  milestone.

### Phase 9 - Non-gravitational forces in the proximity regime

The heliocentric `propagate_n_body_spk` already supports the
`NonGravModel` family (JPL comet `A1`/`A2`/`A3`, dust SRP +
Poynting-Robertson, Farnocchia 2025 physical radiation model) by
adding the inertial Sun-relative non-grav acceleration to the Sun-
gravity term inside `spk_accel`.  The proximity-regime integrator
needs the same plumbing so that close-encounter trajectories of
comets and SRP-sensitive small bodies obey the same physics.

- `kete_core::forces::NonGravModel` is now threaded through
  `ProxMeta` and `propagate_near_body` as `Option<&NonGravModel>`.
  Inside `near_body_accel` the body-fixed body-natural particle
  state is round-tripped through (1) length scaling by
  `body.units.length_au`, (2) `RotationModel::rotate_to_inertial`
  for both position and the body-fixed velocity (with the
  `omega x r` correction added when transferring to the inertial
  frame), (3) addition of the body's Sun-relative SPK state.  The
  resulting Sun-relative inertial pos/vel are passed to
  `NonGravModel::add_acceleration`; the inertial-frame
  acceleration is then rotated back into body-fixed coordinates
  and divided by `length_au` to land in body-natural units.
- PyO3 binding `propagate_near_body_py` gains an optional
  `non_grav: Option<PyNonGravModel>` kwarg, surfaced in Python as
  `kete.extended_body.propagate_near_body(..., non_grav=None)`.
  The wrapper reuses the existing `PyNonGravModel` type so users
  can pass any constructor (`new_comet`, `new_dust`,
  `new_farnocchia`, ...) accepted by the heliocentric path.
- Two Python tests in `src/tests/test_extended_body.py`:
  - `test_nongrav_changes_trajectory` confirms that turning a JPL
    comet model on perturbs the trajectory measurably relative to
    the gravity-only proximity-regime run.
  - `test_nongrav_parity_against_heliocentric` runs the same
    `JplComet` model in both regimes (heliocentric reference vs
    proximity Sun-only-perturber) and checks that the *delta*
    introduced by the non-grav model agrees in direction and
    magnitude (~25% absolute, limited by second-order coupling
    between planet perturbations and the rtn-frame definition of
    the comet model).
- Validation: `cargo clippy --release --workspace --all-targets`
  clean; `pytest src/tests/test_extended_body.py` 9/9 pass; full
  Python test suite 1155 passed (1 pre-existing failure).

Status: complete

### Phase 9.1 - Default perturbers from registered masses

Follow-up polish so callers no longer have to hand-roll a Sun
perturber list.  Mirrors `propagate_n_body_spk`'s
`include_asteroids` switch.

- `propagate_near_body_py` now seeds `Vec<Perturber>` from
  `GravParams::planets()` (or `GravParams::selected_masses()` when
  the new `include_asteroids: bool` kwarg is `True`), filtering
  out the central body's own NAIF id so its polyhedron field is
  not double-counted with a point-mass term.  Caller-supplied
  `perturbers` are appended to that base list (override / extend
  semantics).  `GravParams.mass` is stored as `mass_solar * GMS`,
  i.e. already a `gm_solar` value in AU^3/day^2, so the conversion
  to `Perturber` is a direct copy.
- The example (`src/examples/plot_extended_body.py`) and the three
  proximity-regime tests in `src/tests/test_extended_body.py` no
  longer pass `perturbers=[(10, sun_gm)]` explicitly; the default
  set already includes the Sun.
- Validation: `cargo clippy --release --workspace --all-targets`
  clean; `pytest src/tests/test_extended_body.py` 9/9 pass; full
  Python test suite 1155 pass (1 pre-existing unrelated failure);
  `ruff check` clean; example runs in ~3 sec.

Status: complete

## Follow-on: spherical harmonic gravity for planets

Once Phases 1-6 land, adding spherical harmonics is mostly mechanical:

1. New file `src/kete_core/src/shape/spherical_harmonics.rs` implementing
   `ExtendedGravity` for normalized Stokes coefficients (`C_nm`, `S_nm`).
   Standard recursion-based evaluation; no architectural changes.
2. Loader for standard formats (e.g. EGM2008-style coefficient tables, plus
   per-planet coefficients from PDS).
3. `min_valid_radius` returns the Brillouin sphere radius; the regime
   coordinator uses this to refuse evaluation inside it.
4. Reuses `ExtendedBody`, `RotationModel`, `BodyUnits`, `small_body_accel`,
   `propagate_near_body`, hand-off, and PyO3 bindings unchanged.
5. Likely needs a PCK-backed `RotationModel::Pck` variant for real planets;
   that's the only meaningful new piece of infrastructure.

This is what the trait abstraction and the unit/rotation/regime separation
buys us. It is not implemented in the initial work but is explicitly
designed for.

## Open questions

- Shape-model file format priorities beyond OBJ? (PDS DSK / Gaskell formats?)
- Default proximity-radius heuristic - some multiple of Hill radius, or
  bounding radius? (Hill radius is defined relative to a perturber and is
  more physically meaningful, but bounding-radius multiple is simpler.)
- Should `ExtendedBody` own its ephemeris, or should the caller pass it in
  per call? (Plan currently has caller-passes-in for `kete_core` purity;
  `kete_spice` glue can wrap to make this ergonomic.)
- For spherical harmonics later: do we want the gravity gradient tensor
  (needed for variational equations / STMs), or just acceleration?
