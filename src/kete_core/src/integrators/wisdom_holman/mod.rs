// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! # Wisdom-Holman Symplectic Integrator
//! A fixed-step symplectic N-body map for long term orbital evolution.
//!
//! Unlike the general purpose ODE integrators in this module, this is a specialized
//! N-body map. It advances the whole system at once, and the gravitational dynamics
//! are part of the method itself rather than a user supplied acceleration function.
use crate::analysis::hill_radius;
use crate::constants::{C_AU_PER_DAY, C_AU_PER_DAY_INV_SQUARED, GMS};
use crate::desigs::Desig;
use crate::errors::{Error, KeteResult};
use crate::forces::{
    FarnocchiaNonGrav, GravParams, JplCometNonGrav, NonGravKind, NonGravMask, Shape,
    apply_gr_correction, j2_correction, radiation_accel,
};
use crate::frames::{Ecliptic, Equatorial, SSB, StaticFrame, Vector};
use crate::integrators::RadauIntegrator;
use crate::kepler::analytic_2_body_delta;
use crate::kepler::{compute_peri_dist, compute_semi_major};
use crate::state::State;
use crate::time::{TDB, Time};
use nalgebra::Vector3;
use rayon::prelude::*;
use std::cell::Cell;
use std::f64::consts::TAU;
use std::marker::PhantomData;

#[cfg(test)]
mod tests;

/// Radius of the Sun in AU, from the built-in mass table (`masses.tsv`).
/// The literal fallback (the same value) only applies if the table were
/// somehow missing the Sun.
static SUN_RADIUS_AU: std::sync::LazyLock<f64> = std::sync::LazyLock::new(|| {
    GravParams::planets()
        .iter()
        .find(|p| p.naif_id == 10)
        .map_or(0.0046547587, |p| f64::from(p.radius))
});

/// J2 coefficient and spin pole of the Sun from its [`GravParams`] entry, the
/// pole rotated onto the axes of frame `T`.
///
/// This is the description the Radau N-body path evaluates, so the two
/// integrators apply the same solar oblateness.
///
/// # Errors
/// Returns an error if the mass table has no Sun, or the Sun's entry is not
/// [`Shape::Oblate`].
fn sun_oblateness<T: StaticFrame>() -> KeteResult<(f64, Vector3<f64>)> {
    let planets = GravParams::planets();
    let sun = planets.iter().find(|p| p.naif_id == 10).ok_or_else(|| {
        Error::ValueError("The Sun (NAIF id 10) is not in the table of massive objects.".into())
    })?;
    match &sun.shape {
        Shape::Oblate { j2, pole } => Ok((
            *j2,
            Vector::<Equatorial>::new((*pole).into())
                .into_frame::<T>()
                .into(),
        )),
        Shape::Point => Err(Error::ValueError(
            "The solar J2 term was requested, but the Sun is described as a point mass.".into(),
        )),
        Shape::Polyhedron { .. } => Err(Error::ValueError(
            "The solar J2 term was requested, but the Sun is described as a polyhedron.".into(),
        )),
        Shape::SphericalHarmonics { .. } => Err(Error::ValueError(
            "The solar J2 term was requested, but the Sun is described by spherical harmonics."
                .into(),
        )),
    }
}

/// Effective oblateness coefficient `J2 R^2` in AU^2 of the Earth-Moon pair,
/// treated as an azimuthally symmetric quadrupole about the ecliptic pole:
/// the orbit-averaged lunar quadrupole of Quinn, Tremaine & Duncan (1991).
///
/// For two point masses the quadrupole moment is `mu a_m^2` with `mu` the
/// reduced mass. Averaging the separation direction over the lunar orbit
/// (taken circular in the ecliptic, with small factors for the lunar
/// eccentricity and inclination) leaves an interaction with the Sun of
/// exactly the J2 form, with
/// `J2 R^2 = (mu / M_emb) a_m^2 (1 + 3/2 e_m^2) (1 - 3/2 sin^2 i_m) / 2`.
/// The Earth and Moon masses come from the built-in table; the lunar
/// elements are fixed (a = 384400 km, e = 0.0549, i = 5.145 deg).
static EMB_QUAD_J2R2: std::sync::LazyLock<f64> = std::sync::LazyLock::new(|| {
    let planets = GravParams::planets();
    let mass = |id: i32| {
        planets
            .iter()
            .find(|p| p.naif_id == id)
            .map_or(0.0, |p| p.mass)
    };
    let (earth, moon) = (mass(399), mass(301));
    if earth == 0.0 || moon == 0.0 {
        return 0.0;
    }
    let a_m = 384_400.0 / 149_597_870.7;
    let ecc_fac = 1.0 + 1.5 * 0.0549_f64.powi(2);
    let inc_fac = 1.0 - 1.5 * 5.145_f64.to_radians().sin().powi(2);
    let mu_over_m = earth * moon / (earth + moon).powi(2);
    0.5 * mu_over_m * a_m * a_m * ecc_fac * inc_fac
});

/// Maximum allowed difference between input epochs, in days.
const EPOCH_TOL: f64 = 1e-8;

/// Massive body count above which the pairwise kick and the Kepler drifts
/// are computed in parallel.
const PARALLEL_THRESHOLD: usize = 32;

/// Test particle count below which the per-sub-flow fan-outs run serially.
///
/// The per-particle work in a sub-flow is small (a Kepler solve, a few
/// pairwise accelerations, a compensated add), so the rayon fork-join has to
/// be amortized over enough particles to pay for itself. Below this count the
/// coordination costs more than it saves, including at the scale of a typical
/// collisional family study, so a sub-flow runs on a single chunk; at or above it the
/// per-sub-flow chunk floors take over and the work splits across all cores.
/// The threshold is deliberately above `2 x` the largest chunk floor so the
/// serial-to-parallel transition skips the region where only two or three
/// chunks would form (parallel overhead, little parallel benefit).
const TP_SERIAL_MAX: usize = 512;

/// Chunk floor for a test particle fan-out of `n` particles. Below
/// [`TP_SERIAL_MAX`] the whole array is one chunk (serial); at or above it the
/// sub-flow's own `fine` floor sets the granularity. `fine` is smaller for the
/// heavier sub-flows (the kick and Kepler drift) than for the cheap jump, so
/// each splits only as finely as its per-particle cost rewards.
fn tp_chunk(n: usize, fine: usize) -> usize {
    if n < TP_SERIAL_MAX { n.max(1) } else { fine }
}

/// Encounter candidates are tracked as `(hill_ratio, first, second)`, using an
/// infinite ratio to mean no encounter. This is the identity of the reduction
/// which keeps the closest approach.
const NO_ENCOUNTER: (f64, usize, usize) = (f64::INFINITY, 0, 0);

/// The Wisdom (2006) corrector `a` coefficients form the lattice
/// `a_i = i * sqrt(7/40)` in units of the step size; the base value is
/// computed rather than transcribed.
const CORRECTOR_A_BASE: f64 = 0.175; // 7/40, sqrt taken at use site

/// Order-17 corrector `b` coefficients (Wisdom 2006), in units of the step
/// size, indexed `b_171 .. b_178`, transcribed at f64 precision from the
/// `WHFast` reference implementation (Rein & Tamayo 2015). The stage using
/// `a_i` uses `b_{9-i}`.
#[allow(
    clippy::excessive_precision,
    reason = "published corrector table values kept verbatim at 17 significant digits"
)]
const CORRECTOR_B17: [f64; 8] = [
    -0.000_004_334_741_547_337_358,
    0.000_076_436_355_227_935_738,
    -0.000_635_999_830_758_176_59,
    0.003_313_257_706_938_065_6,
    -0.012_071_760_822_342_291,
    0.032_422_198_864_713_58,
    -0.065_192_863_576_377_894,
    0.093_056_103_771_425_959,
];

/// Internal per-particle radiation and thermal recoil (Yarkovsky) binding,
/// destructured from a frozen [`FarnocchiaNonGrav`] at construction. The spin
/// pole is a unit vector already rotated into the map's frame, so the kick
/// evaluates the force with no per-step rotation.
#[derive(Debug, Clone)]
struct Yarkovsky {
    /// Geometric (Lambert) albedo. Enters the radiation pressure term only.
    albedo: f64,
    /// `alpha = 1 - A_B`; multiplies the thermal terms.
    absorptivity: f64,
    /// Axis ratio `e = R_P / R_E`; 1.0 for a sphere.
    flattening: f64,
    /// Unit spin pole on the axes of the map's frame.
    spin_pole: Vector3<f64>,
    /// Area-to-mass ratio in m^2 / kg, the radiation pressure coupling.
    a_over_m: f64,
    /// Dimensionless thermal lag at 1 AU.
    lambda_0: f64,
}

/// Internal per-particle non-gravitational binding, destructured from the
/// [`NonGravMask`] inputs of [`WisdomHolman::new`] by [`bind_non_grav`].
#[derive(Debug, Clone)]
enum NonGrav {
    /// Radiation pressure and thermal recoil, applied in the kick.
    Yarkovsky(Yarkovsky),
    /// Dust grain: the drift runs on the radiation-reduced `(1 - beta) GMS`
    /// and the kick applies the exact Poynting-Robertson drag flow.
    Dust {
        /// Ratio of solar radiation pressure to solar gravity, in `[0, 1)`.
        beta: f64,
    },
    /// JPL-style `A1/A2/A3` accelerations on the radial / transverse / normal
    /// axes of the instantaneous heliocentric orbit, scaled by the model's
    /// `g(r)`. Applied in the kick with the pre-kick heliocentric velocity;
    /// only the un-lagged (`dt = 0`) form is supported.
    JplComet {
        /// The `g(r)` shape, validated at construction with `dt = 0`.
        force: JplCometNonGrav,
        /// Radial acceleration coefficient in AU/Day^2 at `g = 1`.
        a1: f64,
        /// Transverse acceleration coefficient in AU/Day^2 at `g = 1`.
        a2: f64,
        /// Normal acceleration coefficient in AU/Day^2 at `g = 1`.
        a3: f64,
    },
}

/// Why a test particle was removed from the simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LostReason {
    /// The particle came within one solar radius of the Sun's center.
    SunImpact,
    /// The universal Kepler solver failed to converge, which indicates an
    /// extreme state such as a deep close encounter driven by too large a step.
    KeplerFailure,
    /// The particle passed perihelion inside the perihelion bound set with
    /// [`WisdomHolman::set_removal_bounds`].
    PerihelionBound,
    /// The particle moved beyond the distance bound set with
    /// [`WisdomHolman::set_removal_bounds`].
    DistanceBound,
    /// The particle came within the radius of a planet during a resolved
    /// close encounter (see [`WisdomHolman::with_close_encounters`]).
    PlanetImpact,
    /// The Radau integration of a resolved close encounter failed.
    EncounterFailure,
}

/// Record of a test particle removed during integration.
#[derive(Debug, Clone)]
pub struct LostParticle {
    /// Designation of the removed particle.
    pub desig: Desig,
    /// When the particle was removed. In a resolved close encounter, the time
    /// of the impact or perihelion passage; otherwise the start of the drift,
    /// or of the resolved step, in which its removal was found.
    pub epoch: Time<TDB>,
    /// Why the particle was removed.
    pub reason: LostReason,
}

/// The closest approach between two bodies seen during a run.
///
/// Only unresolved approaches within 3 Hill radii are recorded, since the
/// map's accuracy degrades inside that region; test particle steps resolved
/// as close encounters (see [`WisdomHolman::with_close_encounters`]) are not.
/// If this is reported by a simulation, the trajectories of the involved
/// bodies should be treated with suspicion.
#[derive(Debug, Clone)]
pub struct Encounter {
    /// Separation in units of the larger of the two bodies' Hill radii.
    pub hill_ratio: f64,
    /// Designation of the massive body involved; for an approach between two
    /// massive bodies, the one earlier in the input list.
    pub first: Desig,
    /// Designation of the other body or test particle.
    pub second: Desig,
    /// Epoch of the step during which the approach was seen.
    pub epoch: Time<TDB>,
}

/// Wisdom-Holman symplectic N-body integrator.
///
/// This is a fixed-step symplectic map using the democratic heliocentric
/// splitting of Duncan, Levison & Lee (1998). It is aimed at long term
/// (millions to billions of years) integrations of the Sun, the planets, a
/// moderate number of massive asteroids, and up to thousands of massless test
/// particles. For short term high accuracy propagation the adaptive Radau
/// integrator is the better tool; this map instead keeps the energy error
/// bounded, with no secular drift, over very long spans.
///
/// The Hamiltonian is split in democratic heliocentric coordinates: for each
/// body `i`, the position `Q_i` is heliocentric while the momentum `P_i` stays
/// barycentric. This gives three parts, each with an exact flow:
///
/// - Kepler drift: each body moves on an independent two-body orbit about the
///   solar `GMS`, using the universal-variable solver in `kepler.rs`.
/// - Jump: every `Q_i` shifts by the total massive momentum divided by the
///   solar mass; momenta unchanged. This is the price of the non-inertial
///   heliocentric origin.
/// - Kick: pairwise gravity between all non-solar bodies; positions unchanged.
///
/// One step is the symmetric composition
/// `kepler(dt/2) jump(dt/2) kick(dt) jump(dt/2) kepler(dt/2)`, which is second
/// order and time reversible. Across the steps of a single
/// [`Self::integrate_n_steps`] call the adjacent half Kepler drifts are fused
/// into whole drifts, which is exact since the Kepler flow composes as a group.
///
/// Constructed from barycentric states via [`Self::new`], stepped with
/// [`Self::step`] or [`Self::integrate_n_steps`], and queried with
/// [`Self::massive_states`] / [`Self::test_particle_states`]. Output states are
/// reconstructed in the frame and origin of the input states. All internal
/// updates use Neumaier compensated summation, which keeps the accumulated
/// roundoff a random walk rather than a linear drift over billions of steps.
///
/// Behavior worth noting:
///
/// - The step size is fixed at construction. Larger steps do not fail loudly,
///   they degrade the error and eventually destabilize; at most 1/20th of the
///   shortest orbital period is the usual guidance (see
///   [`Self::shortest_period`]).
/// - Earth and Moon are best provided as their combined barycenter, otherwise
///   the Moon's short period forces a very small step. A massive body which
///   resolves to the Earth-Moon barycenter (NAIF id 3) automatically receives
///   the orbit-averaged lunar quadrupole of Quinn, Tremaine & Duncan (1991):
///   the merged pair acts as an effective oblateness about the ecliptic pole,
///   restoring most of the solar interaction the point-mass merge loses.
/// - The splitting assumes the interaction term is small compared to the
///   Kepler term, which fails in a close encounter. Test particle encounters
///   can be resolved with [`Self::with_close_encounters`], which integrates a
///   particle's step with Radau whenever it passes near a massive body;
///   encounters between massive bodies are never resolved. The map tracks
///   the closest approach seen, in Hill-radius units, via
///   [`Self::closest_encounter`] so runs can be audited.
/// - Test particles which fall into the Sun, or whose Kepler drift fails to
///   converge, are removed and recorded in [`Self::lost_particles`]. The same
///   situations raise an error for a massive body. Resolved close encounters
///   also record planet impacts and failed integrations there.
/// - Test particles may carry a frozen non-gravitational force, using the
///   same per-object vocabulary as the Radau N-body propagation: the
///   Farnocchia radiation and thermal recoil (Yarkovsky) model, the dust
///   radiation-pressure plus Poynting-Robertson drag model, or the JPL
///   `A1/A2/A3` model (the form JPL orbit solutions use both for comet
///   outgassing and for asteroid Yarkovsky detections via `A2`). A dust
///   particle drifts on its radiation-reduced two-body orbit `(1 - beta) GMS`
///   and is damped by the exact drag flow in the kick, so its semi-major axis
///   decays secularly; these forces are dissipative, so the map is neither
///   symplectic nor reversible for particles carrying them, by design. The
///   fitted constants are extrapolated unchanged over the whole integration
///   (no spin evolution or activity changes). The time-lagged (`dt != 0`)
///   outgassing variant is not supported.
/// - The optional general relativity correction applies the first-order
///   Schwarzschild acceleration of the Sun, the same term used by the Radau
///   N-body path, which reproduces both the secular apsidal precession and
///   the relativistic mean motion (measured against JPL DE441 this is what
///   keeps Mercury's long-term phase drift small). The term is velocity
///   dependent, evaluated at the pre-kick velocities, so with it enabled the
///   map is only approximately symplectic; the energy error remains a
///   bounded band in practice (see the test suite). It is the single-body
///   term only; planet-planet 1PN cross terms are absent.
/// - The optional solar J2 term applies the oblateness acceleration shared
///   with the Radau N-body path, with the J2 coefficient and spin pole of the
///   Sun's [`GravParams`] entry. It is a position-only potential,
///   so the map remains symplectic; it drives the secular nodal regression
///   `-(3/2) n J2 (R/p)^2 cos(i)` and the corresponding apsidal precession.
/// - The optional order-17 symplectic corrector (Wisdom 2006) removes the
///   dominant oscillating error of the map at a fixed cost per integration
///   call.
/// - Removal bounds ([`Self::set_removal_bounds`]) drop test particles that
///   pass perihelion inside a distance or move beyond one, recording them in
///   [`Self::lost_particles`].
///
/// References:
///
/// J. Wisdom and M. Holman (1991), 'Symplectic maps for the n-body problem',
/// The Astronomical Journal, vol. 102, no. 4, pp. 1528-1538
///
/// M. J. Duncan, H. F. Levison and M. H. Lee (1998), 'A multiple time step
/// symplectic algorithm for integrating close encounters',
/// The Astronomical Journal, vol. 116, no. 4, pp. 2067-2077
///
/// T. R. Quinn, S. Tremaine and M. Duncan (1991), 'A three million year
/// integration of the earth's orbit', The Astronomical Journal, vol. 101,
/// pp. 2287-2305
///
/// J. Wisdom (2006), 'Symplectic correctors for canonical heliocentric n-body
/// maps', The Astronomical Journal, vol. 131, no. 4, pp. 2294-2298
///
/// H. Rein and D. Tamayo (2015), '`WHFast`: a fast and unbiased implementation
/// of a symplectic Wisdom-Holman integrator for long-term gravitational
/// simulations', Monthly Notices of the Royal Astronomical Society, vol. 452,
/// no. 1, pp. 376-388
///
/// A. M. Nobili and I. W. Roxburgh (1986), 'Simulation of general relativistic
/// corrections in long term numerical integrations of planetary orbits',
/// Relativity in Celestial Mechanics and Astrometry (IAU Symposium 114),
/// pp. 105-110
#[derive(Debug, Clone)]
pub struct WisdomHolman<T: StaticFrame> {
    /// Fixed step size in days, sign gives the direction of integration.
    dt: f64,
    /// Epoch of the initial conditions.
    epoch0: Time<TDB>,
    /// Number of steps taken so far.
    steps: i64,
    /// Whether the GR potential correction is applied.
    include_gr: bool,
    /// J2 coefficient and spin pole of the Sun, the pole on the axes of the
    /// map's frame, when the solar J2 oblateness term is applied.
    solar_j2: Option<(f64, Vector3<f64>)>,
    /// Whether the order-17 symplectic corrector wraps each integration call.
    use_correctors: bool,
    /// Test particles passing perihelion inside this distance (AU) are removed.
    peri_bound: Option<f64>,
    /// Test particles beyond this heliocentric distance (AU) are removed.
    dist_bound: Option<f64>,
    /// Encounter radius in Hill radii inside which a test particle's step is
    /// integrated with Radau, when close encounters are resolved.
    encounter_radius: Option<f64>,
    /// Physical radius (AU) of each massive body (Sun excluded) from the mass
    /// table, 0 where the body is not in it.
    radii: Vec<f64>,
    /// Number of test particle steps resolved as close encounters.
    close_encounter_steps: u64,
    /// Test particles the symplectic corrector leaves alone, each with the
    /// massive body it is near, aligned with the test particle arrays while a
    /// corrector is applied; empty otherwise.
    corrector_skip: Vec<Option<usize>>,
    /// The ecliptic pole on the axes of the map's frame: the symmetry axis of
    /// the orbit-averaged lunar quadrupole.
    ecliptic_pole: Vector3<f64>,
    /// Index into the massive arrays of the body resolving to the Earth-Moon
    /// barycenter (NAIF id 3), which receives the orbit-averaged lunar
    /// quadrupole correction. [`None`] when no such body is present.
    emb_idx: Option<usize>,

    /// Designation of the central body.
    sun_desig: Desig,
    /// Designations of the massive bodies (Sun excluded).
    desigs: Vec<Desig>,
    /// GM of each massive body in AU^3/Day^2 (Sun excluded).
    gms: Vec<f64>,
    /// Heliocentric positions in AU.
    q: Vec<CompVec3>,
    /// Barycentric velocities in AU/Day (center of mass frame).
    v: Vec<CompVec3>,

    /// Test particle designations.
    tp_desigs: Vec<Desig>,
    /// Test particle heliocentric positions in AU.
    tp_q: Vec<CompVec3>,
    /// Test particle barycentric velocities in AU/Day.
    tp_v: Vec<CompVec3>,
    /// Non-gravitational binding per test particle. Left empty when no
    /// particle has one, so the common gravity-only case streams nothing
    /// extra through the drift and kick.
    tp_forces: Vec<Option<NonGrav>>,
    /// `q . v` of each test particle at the end of its last drift of a map
    /// step, NaN where unknown. The perihelion bound compares it with the
    /// next drift, since the jump and kick between drifts change it too.
    tp_rv: Vec<f64>,

    /// Sum of all GMs including the Sun.
    total_gm: f64,
    /// Center of mass position of the inputs at the initial epoch.
    com_pos0: Vector3<f64>,
    /// Center of mass velocity of the inputs, constant for all time.
    com_vel: Vector3<f64>,

    /// Precomputed `cbrt(gm / (3 GMS))` per massive body.
    hill_fac: Vec<f64>,
    /// Scratch: instantaneous Hill radius per massive body.
    hill_r: Vec<f64>,
    /// Scratch: squared 3-Hill-radius encounter threshold per massive body.
    hill_thresh: Vec<f64>,
    /// Scratch: kick accelerations for the massive bodies.
    accel: Vec<Vector3<f64>>,

    /// Closest sub-3-Hill-radius approach seen, if any.
    closest: Option<Encounter>,
    /// Test particles removed during integration.
    lost: Vec<LostParticle>,
    /// The states are stored as raw vectors; this pins the frame they are
    /// expressed in.
    _frame: PhantomData<T>,
}

impl<T: StaticFrame> WisdomHolman<T> {
    /// Construct the integrator from barycentric states.
    ///
    /// # Arguments
    ///
    /// * `massive` - Barycentric states of all massive bodies. The first
    ///   entry must be the Sun. All states must share the same epoch. For
    ///   long integrations the Earth-Moon barycenter should be used in place
    ///   of the separate Earth and Moon.
    /// * `gms` - GM of each massive body in AU^3 / Day^2, parallel to
    ///   `massive`, the same convention as [`crate::forces::GravParams`].
    ///   The Sun's must equal [`GMS`], since the Kepler drift is built
    ///   around the solar GM.
    /// * `test_particles` - Massless particles, in the same frame and at the
    ///   same epoch. They feel all massive bodies but affect nothing.
    /// * `non_gravs` - Optional non-gravitational force per test particle, with
    ///   every parameter fixed: either empty (every particle feels gravity alone) or one
    ///   entry per particle, with [`None`] meaning gravity only. This is the
    ///   same per-object vocabulary used by the Radau N-body propagation.
    ///   [`NonGravKind::Farnocchia`] (frozen values `[a_over_m, lambda_0]`)
    ///   gives a particle radiation pressure and thermal recoil; its
    ///   equatorial spin pole is rotated into the map's frame here.
    ///   [`NonGravKind::Dust`] (frozen value `[beta]`, in `[0, 1)`) makes
    ///   the particle a dust grain on a radiation-reduced orbit with
    ///   Poynting-Robertson drag. [`NonGravKind::JplComet`] (frozen values
    ///   `[a1, a2, a3]`) applies the JPL RTN accelerations with their `g(r)`
    ///   scaling; only the un-lagged (`dt = 0`) form is accepted.
    /// * `dt` - Fixed step size in days. Negative integrates backwards. Should
    ///   be at most 1/20th of the shortest orbital period in the system.
    /// * `include_gr` - Apply the solar GR correction (first-order
    ///   Schwarzschild acceleration, shared with the Radau N-body path). The
    ///   term is velocity dependent, so with it enabled the map is only
    ///   approximately symplectic; energy stays in a bounded band.
    /// * `include_j2` - Apply the solar J2 oblateness term, with the J2
    ///   coefficient and spin pole of the Sun's [`GravParams`] entry, as used by
    ///   the Radau N-body path. This is a position-only potential, so the map
    ///   remains symplectic; it drives the small secular nodal regression and
    ///   apsidal precession of low-`a` orbits. The term is symmetric about
    ///   the solar spin pole, so of [`Self::angular_momentum`] only the
    ///   component along that pole is conserved (the reaction torque on the
    ///   solar spin is not modeled), and only when no Earth-Moon barycenter
    ///   carries the lunar quadrupole, whose axis is the ecliptic pole.
    /// * `use_correctors` - Wrap each integration call in the order-17
    ///   symplectic corrector (Wisdom 2006), which removes the dominant
    ///   oscillating error of the map (roughly a factor of the planet/Sun
    ///   mass ratio). The overhead is a fixed ~30 steps worth of work per
    ///   [`Self::integrate_n_steps`] call, so batch many steps per call.
    ///
    /// # Errors
    ///
    /// Returns an error if `massive` and `gms` differ in length, the first
    /// massive body is not the Sun, epochs disagree, any GM is non-positive,
    /// any state is non-finite, `non_gravs` is non-empty with a length other
    /// than `test_particles`, any non-grav entry fails the checks of
    /// binding the non-gravitational force (a parameter left free, non-finite or
    /// out-of-range values, invalid surface description, or an unsupported
    /// kind), or `dt` is zero or non-finite.
    #[allow(
        clippy::too_many_arguments,
        reason = "parallel per-object lists plus independent physics toggles"
    )]
    pub fn new(
        massive: &[State<T, SSB>],
        gms: &[f64],
        test_particles: &[State<T, SSB>],
        non_gravs: &[Option<NonGravMask>],
        dt: f64,
        include_gr: bool,
        include_j2: bool,
        use_correctors: bool,
    ) -> KeteResult<Self> {
        if !dt.is_normal() {
            Err(Error::ValueError("dt must be finite and non-zero.".into()))?;
        }
        if massive.len() != gms.len() {
            Err(Error::ValueError(format!(
                "massive and gms must have the same length, found {} states and {} gms.",
                massive.len(),
                gms.len()
            )))?;
        }
        let Some(sun) = massive.first() else {
            return Err(Error::ValueError(
                "The massive body list must contain at least the Sun.".into(),
            ));
        };
        if ((gms[0] - GMS) / GMS).abs() > 1e-12 {
            Err(Error::ValueError(format!(
                "The first massive body must be the Sun with gm = GMS = {GMS:e}, found gm = {:e}. \
                 The Kepler drift of this integrator is built around the solar GM.",
                gms[0]
            )))?;
        }
        let epoch0 = sun.epoch;

        for (state, gm) in massive.iter().zip(gms) {
            if *gm <= 0.0 || !gm.is_finite() {
                Err(Error::ValueError(format!(
                    "Massive body {:?} has non-positive or non-finite gm.",
                    state.desig
                )))?;
            }
        }
        if !(non_gravs.is_empty() || non_gravs.len() == test_particles.len()) {
            Err(Error::ValueError(format!(
                "non_gravs must be empty or have one entry per test particle, \
                 found {} entries for {} particles.",
                non_gravs.len(),
                test_particles.len()
            )))?;
        }
        for state in massive.iter().chain(test_particles) {
            if (state.epoch - epoch0).elapsed.abs() > EPOCH_TOL {
                Err(Error::ValueError(format!(
                    "State {:?} has epoch {} which does not match the Sun's epoch {}.",
                    state.desig,
                    state.epoch.jd(),
                    epoch0.jd()
                )))?;
            }
            if !state.is_finite() {
                Err(Error::ValueError(format!(
                    "State {:?} contains non-finite position or velocity.",
                    state.desig
                )))?;
            }
        }

        // Center of mass of the massive bodies; velocities are stored relative
        // to it so the internal momenta are exactly the canonical barycentric
        // momenta regardless of the inertial origin the caller used.
        let total_gm: f64 = gms.iter().sum();
        let mut com_pos0 = Vector3::zeros();
        let mut com_vel = Vector3::zeros();
        for (state, gm) in massive.iter().zip(gms) {
            com_pos0 += Vector3::from(state.pos) * *gm;
            com_vel += Vector3::from(state.vel) * *gm;
        }
        com_pos0 /= total_gm;
        com_vel /= total_gm;

        let sun_pos: Vector3<f64> = sun.pos.into();

        let n = massive.len() - 1;
        let mut desigs = Vec::with_capacity(n);
        let mut q = Vec::with_capacity(n);
        let mut v = Vec::with_capacity(n);
        for state in &massive[1..] {
            desigs.push(state.desig.clone());
            q.push(CompVec3::new(Vector3::from(state.pos) - sun_pos));
            v.push(CompVec3::new(Vector3::from(state.vel) - com_vel));
        }
        let gms = gms[1..].to_vec();
        let selected = GravParams::selected_masses();
        let radii = desigs
            .iter()
            .map(|desig| {
                desig.clone().naif_id().map_or(0.0, |id| {
                    impact_radius(id, &selected, GravParams::known_masses())
                })
            })
            .collect();
        drop(selected);
        // Hill radius per AU of heliocentric distance; scaled by the current
        // distance during each kick.
        let hill_fac = gms
            .iter()
            .map(|gm| hill_radius(1.0, 0.0, *gm, GMS))
            .collect();

        let tp_desigs = test_particles.iter().map(|p| p.desig.clone()).collect();
        let tp_q = test_particles
            .iter()
            .map(|p| CompVec3::new(Vector3::from(p.pos) - sun_pos))
            .collect();
        let tp_v = test_particles
            .iter()
            .map(|p| CompVec3::new(Vector3::from(p.vel) - com_vel))
            .collect();
        // Left empty unless at least one particle has a non-grav force, so the
        // gravity-only case reads nothing extra in the drift and kick.
        let tp_forces = if non_gravs.iter().any(Option::is_some) {
            non_gravs
                .iter()
                .zip(test_particles)
                .map(|(frozen, state)| {
                    frozen
                        .as_ref()
                        .map(|frozen| bind_non_grav::<T>(frozen, &state.desig))
                        .transpose()
                })
                .collect::<KeteResult<Vec<_>>>()?
        } else {
            Vec::new()
        };

        let tp_rv = vec![f64::NAN; test_particles.len()];

        let solar_j2 = if include_j2 {
            Some(sun_oblateness::<T>()?)
        } else {
            None
        };

        // The merged Earth-Moon barycenter is recognized by its NAIF id so it
        // can receive the orbit-averaged lunar quadrupole correction.
        let emb_idx = desigs.iter().position(|d| d.clone().naif_id() == Some(3));

        Ok(Self {
            dt,
            epoch0,
            steps: 0,
            include_gr,
            solar_j2,
            use_correctors,
            peri_bound: None,
            dist_bound: None,
            encounter_radius: None,
            radii,
            close_encounter_steps: 0,
            corrector_skip: Vec::new(),
            ecliptic_pole: Vector::<Ecliptic>::new([0.0, 0.0, 1.0])
                .into_frame::<T>()
                .into(),
            emb_idx,
            sun_desig: sun.desig.clone(),
            desigs,
            gms,
            q,
            v,
            tp_desigs,
            tp_q,
            tp_v,
            tp_forces,
            tp_rv,
            total_gm,
            com_pos0,
            com_vel,
            hill_fac,
            hill_r: vec![0.0; n],
            hill_thresh: vec![0.0; n],
            accel: vec![Vector3::zeros(); n],
            closest: None,
            lost: Vec::new(),
            _frame: PhantomData,
        })
    }

    /// Resolve test particle close encounters from here on.
    ///
    /// At each step's kick, a test particle that passes within `hill_radii`
    /// Hill radii of a massive body during the step is taken out of the map
    /// for that step. The test takes the relative motion as a straight line
    /// over half a step either side of the kick, with the radius widened by
    /// how far the Sun's differential pull can bend it. The particle's state
    /// at the start of the step is recovered by undoing the half drift and
    /// jump already applied, and the whole step is integrated with the
    /// adaptive Radau integrator: the Sun, every massive body, the indirect
    /// term of the heliocentric frame, and the same GR, J2 and
    /// non-gravitational terms as the map. A body the particle passes near
    /// follows its two-body heliocentric orbit from the start of the step,
    /// corrected by cubic Hermite interpolation onto the map's own state at
    /// the end of the step; the others follow the cubic Hermite of their two
    /// end states. The result is put back so that the map's next half drift
    /// lands on it.
    ///
    /// During the integration a particle at an accepted state inside the
    /// Sun is removed as [`LostReason::SunImpact`], inside a massive body as
    /// [`LostReason::PlanetImpact`], and one crossing the removal bounds
    /// ([`Self::set_removal_bounds`]) as for the map, except that a perihelion
    /// passage is a turn of its integrated heliocentric path inside the bound,
    /// so that a flyby's deflection is not one; one whose integration
    /// fails is removed as [`LostReason::EncounterFailure`]. A body's impact
    /// radius is found by NAIF id: its radius among the registered masses,
    /// else in the mass table, the Earth's for the Earth-Moon barycenter. A
    /// body with neither cannot be hit.
    ///
    /// The massive bodies are not affected: their own encounters stay
    /// unresolved. Steps outside an encounter cost only the test. The
    /// resolved steps are not symplectic. With the symplectic corrector
    /// enabled, a particle the encounter test flags at the start or end of an
    /// integration call is left out of the corrector there, whose own drifts
    /// span several steps, and is moved instead with the body it passes
    /// nearest, keeping its state relative to that body.
    /// [`Self::close_encounter_steps`] counts the particle steps resolved.
    ///
    /// # Errors
    ///
    /// Returns an error if `hill_radii` is not finite and positive.
    pub fn with_close_encounters(mut self, hill_radii: f64) -> KeteResult<Self> {
        if !(hill_radii.is_finite() && hill_radii > 0.0) {
            Err(Error::ValueError(format!(
                "The close encounter radius must be finite and positive, found {hill_radii}."
            )))?;
        }
        self.encounter_radius = Some(hill_radii);
        Ok(self)
    }

    /// Close encounter radius in Hill radii, when test particle encounters
    /// are resolved (see [`Self::with_close_encounters`]).
    #[must_use]
    pub fn close_encounter_radius(&self) -> Option<f64> {
        self.encounter_radius
    }

    /// Number of test particle steps resolved as close encounters so far,
    /// including those ending in a removal.
    #[must_use]
    pub fn close_encounter_steps(&self) -> u64 {
        self.close_encounter_steps
    }

    /// Remove test particles that pass perihelion inside `min_perihelion` AU
    /// (an absorbing boundary, such as the planetary region for comets), or
    /// that move beyond `max_distance` AU from the Sun. [`None`] disables a
    /// bound. Removed particles are recorded in [`Self::lost_particles`].
    ///
    /// The perihelion test uses each particle's Kepler orbit: a particle is
    /// removed when its radial velocity turns from inward to outward with
    /// perihelion inside the bound, during a drift or between two, or when a
    /// drift with perihelion inside the bound is long enough to have passed
    /// it whatever its ends show: over half an orbit with both ends moving the
    /// same way, or a whole orbit. Only the
    /// drifts of the map's steps are tested, not those inside the symplectic
    /// corrector.
    ///
    /// # Errors
    ///
    /// Returns an error if a bound is non-positive or non-finite.
    pub fn set_removal_bounds(
        &mut self,
        min_perihelion: Option<f64>,
        max_distance: Option<f64>,
    ) -> KeteResult<()> {
        for (name, value) in [
            ("min_perihelion", min_perihelion),
            ("max_distance", max_distance),
        ] {
            if let Some(v) = value
                && !(v.is_finite() && v > 0.0)
            {
                Err(Error::ValueError(format!(
                    "{name} must be positive and finite, found {v}."
                )))?;
            }
        }
        self.peri_bound = min_perihelion;
        self.dist_bound = max_distance;
        Ok(())
    }

    /// Advance the simulation by a single step of `dt`.
    ///
    /// Equivalent to `integrate_n_steps(1)`. With correctors enabled this
    /// pays the full corrector overhead for a single step; prefer batching
    /// steps through [`Self::integrate_n_steps`] or [`Self::integrate_to`].
    ///
    /// # Errors
    ///
    /// See [`Self::integrate_n_steps`].
    pub fn step(&mut self) -> KeteResult<()> {
        self.integrate_n_steps(1)
    }

    /// Advance the simulation by `n` steps.
    ///
    /// The adjacent half Kepler drifts of consecutive steps are fused into
    /// whole drifts, and when correctors are enabled the whole call is wrapped
    /// as `C (kernel)^n C^-1`. The state between calls is always the physical
    /// one.
    ///
    /// # Errors
    ///
    /// Returns an error if the Kepler drift fails for a massive body or a
    /// massive body hits the Sun; the simulation state is not usable after an
    /// error. Test particles in either situation do not raise an error, they
    /// are moved to [`Self::lost_particles`].
    pub fn integrate_n_steps(&mut self, n: u64) -> KeteResult<()> {
        if n == 0 {
            return Ok(());
        }
        let dt = self.dt;
        if self.use_correctors {
            self.guarded_corrector(1.0)?;
        }
        self.kepler(0.5 * dt, Some(self.epoch()))?;
        for i in 0..n {
            self.jump(0.5 * dt);
            self.kick(dt, true)?;
            self.jump(0.5 * dt);
            // Each later drift starts half a step into the current step.
            let start = Some(self.epoch() + 0.5 * dt);
            if i + 1 < n {
                self.kepler(dt, start)?;
            } else {
                self.kepler(0.5 * dt, start)?;
            }
            self.steps += 1;
        }
        if self.use_correctors {
            self.guarded_corrector(-1.0)?;
        }
        Ok(())
    }

    /// Refresh the instantaneous Hill radius of each massive body, and its
    /// squared 3-Hill-radius encounter threshold, from the current positions.
    fn update_hill_radii(&mut self) {
        for ((r, thresh), (pos, fac)) in self
            .hill_r
            .iter_mut()
            .zip(&mut self.hill_thresh)
            .zip(self.q.iter().zip(&self.hill_fac))
        {
            *r = pos.val.norm() * fac;
            *thresh = 9.0 * *r * *r;
        }
    }

    /// Apply the corrector or its inverse, leaving out the test particles the
    /// close encounter test flags when encounters are resolved: those that
    /// would pass within the encounter radius of a body within half a step.
    /// The corrector's transformation assumes the map's splitting holds, which
    /// it does not there, and its own drifts span several steps. Each such
    /// particle is instead moved with the body it passes nearest (in units of
    /// that body's encounter radius), by the change the corrector makes to that
    /// body's state, so its state relative to the body is kept.
    fn guarded_corrector(&mut self, inv: f64) -> KeteResult<()> {
        if let Some(k) = self.encounter_radius {
            self.update_hill_radii();
            let check = EncounterCheck::new(&self.q, &self.v, &self.hill_r, k, self.dt);
            let skip: Vec<Option<usize>> = self
                .tp_q
                .par_iter()
                .zip(&self.tp_v)
                .map(|(pos, vel)| {
                    let terms = check.particle(&pos.val, &vel.val);
                    self.q
                        .iter()
                        .zip(&self.v)
                        .enumerate()
                        .filter_map(|(j, (bp, bv))| {
                            let sep = bp.val - pos.val;
                            let r2 = sep.norm_squared();
                            check
                                .flags(j, &sep, r2, &(bv.val - vel.val), &terms)
                                .then(|| (j, r2 / check.radius[j].powi(2)))
                        })
                        .min_by(|a, b| a.1.total_cmp(&b.1))
                        .map(|(j, _)| j)
                })
                .collect();
            if skip.iter().any(Option::is_some) {
                self.corrector_skip = skip;
            }
        }
        if self.corrector_skip.is_empty() {
            return self.apply_corrector(inv);
        }
        let before: Vec<(Vector3<f64>, Vector3<f64>)> = self
            .q
            .iter()
            .zip(&self.v)
            .map(|(q, v)| (q.val, v.val))
            .collect();
        let result = self.apply_corrector(inv);
        let skip = std::mem::take(&mut self.corrector_skip);
        for (idx, body) in skip.iter().enumerate() {
            if let Some(j) = *body {
                self.tp_q[idx].add(&(self.q[j].val - before[j].0));
                self.tp_v[idx].add(&(self.v[j].val - before[j].1));
            }
        }
        result
    }

    /// Number of whole steps from the current epoch to the step closest to `time`.
    ///
    /// # Errors
    ///
    /// Returns an error if the target time is behind the current epoch with
    /// respect to the sign of `dt`.
    pub fn steps_to(&self, time: Time<TDB>) -> KeteResult<u64> {
        let n_steps = (time - self.epoch()).elapsed / self.dt;
        if n_steps < -0.5 {
            Err(Error::ValueError(format!(
                "Target time {} is behind the current epoch {} for dt = {}; construct the \
                 integrator with the opposite sign of dt to integrate in that direction.",
                time.jd(),
                self.epoch().jd(),
                self.dt
            )))?;
        }
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "clamped non-negative and bounded by f64 integer range"
        )]
        Ok(n_steps.round().max(0.0) as u64)
    }

    /// Advance the simulation to approximately the target time.
    ///
    /// The map has a fixed step size, so the integration lands on the whole
    /// step closest to `time`; no partial step is taken. The final epoch is
    /// reported by [`Self::epoch`].
    ///
    /// # Errors
    ///
    /// Returns an error if the target time is behind the current epoch with
    /// respect to the sign of `dt`, or if stepping fails (see [`Self::step`]).
    pub fn integrate_to(&mut self, time: Time<TDB>) -> KeteResult<()> {
        let n_steps = self.steps_to(time)?;
        self.integrate_n_steps(n_steps)
    }

    /// Current epoch, computed as `epoch0 + steps * dt`.
    pub fn epoch(&self) -> Time<TDB> {
        self.epoch0 + self.elapsed()
    }

    /// Fixed step size in days.
    #[must_use]
    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// Number of steps taken so far.
    #[must_use]
    pub fn steps_taken(&self) -> i64 {
        self.steps
    }

    /// Whether the GR potential correction is enabled.
    #[must_use]
    pub fn include_gr(&self) -> bool {
        self.include_gr
    }

    /// Whether the solar J2 oblateness term is enabled.
    #[must_use]
    pub fn include_j2(&self) -> bool {
        self.solar_j2.is_some()
    }

    /// Whether the order-17 symplectic corrector is enabled.
    #[must_use]
    pub fn use_correctors(&self) -> bool {
        self.use_correctors
    }

    /// Number of massive bodies, including the Sun.
    #[must_use]
    pub fn n_massive(&self) -> usize {
        self.gms.len() + 1
    }

    /// Number of currently active test particles.
    #[must_use]
    pub fn n_test_particles(&self) -> usize {
        self.tp_q.len()
    }

    /// Test particles removed during integration, in order of removal.
    #[must_use]
    pub fn lost_particles(&self) -> &[LostParticle] {
        &self.lost
    }

    /// The closest unresolved approach within 3 Hill radii seen so far, if any
    /// occurred.
    ///
    /// The map does not resolve these; if this returns [`Some`], the
    /// subsequent trajectories of the involved bodies (and the quantitative
    /// accuracy of the run) should be treated with suspicion. Test particle
    /// steps resolved as close encounters ([`Self::with_close_encounters`])
    /// are not recorded.
    #[must_use]
    pub fn closest_encounter(&self) -> Option<&Encounter> {
        self.closest.as_ref()
    }

    /// The shortest osculating heliocentric orbital period among the massive
    /// bodies, in days. `dt` should be at most 1/20th of this. Returns
    /// infinity if no massive body is on a bound orbit.
    #[must_use]
    pub fn shortest_period(&self) -> f64 {
        let sun_vel = self.sun_internal_vel();
        let mut min_period = f64::INFINITY;
        for (pos, vel) in self.q.iter().zip(&self.v) {
            let v_helio = vel.val - sun_vel;
            let semi_major = compute_semi_major(&pos.val, &v_helio, GMS);
            if semi_major.is_finite() && semi_major > 0.0 {
                min_period = min_period.min(TAU * (semi_major.powi(3) / GMS).sqrt());
            }
        }
        min_period
    }

    /// Total energy of the massive system multiplied by the gravitational
    /// constant, in AU^5 / Day^4.
    ///
    /// This is the conserved energy of the map's dynamics (Kepler,
    /// solar-drift, and interaction terms, plus the 1PN energy of the GR term
    /// and the J2 potential when enabled), excluding the constant
    /// center-of-mass kinetic term. Test particles are massless and
    /// contribute nothing. This oscillates within a bounded band rather than
    /// trending; with GR enabled the band reflects the splitting error of the
    /// velocity-dependent kick rather than pure roundoff.
    #[must_use]
    pub fn energy(&self) -> f64 {
        let mut energy = 0.0;
        let mut momentum = Vector3::zeros();
        let n = self.gms.len();
        let sun_r2 = *SUN_RADIUS_AU * *SUN_RADIUS_AU;
        let sun_vel = self.sun_internal_vel();
        for i in 0..n {
            let gm = self.gms[i];
            let r = self.q[i].val.norm();
            energy += 0.5 * gm * self.v[i].val.norm_squared() - GMS * gm / r;
            momentum += self.v[i].val * gm;
            if self.include_gr {
                // 1PN energy of the Schwarzschild term applied in the kick:
                // (1/c^2) [3/8 v^4 + 3/2 (mu/r) v^2 + mu^2 / (2 r^2)].
                let v2 = (self.v[i].val - sun_vel).norm_squared();
                let mu_r = GMS / r;
                energy += gm
                    * C_AU_PER_DAY_INV_SQUARED
                    * (0.375 * v2 * v2 + 1.5 * mu_r * v2 + 0.5 * mu_r * mu_r);
            }
            if let Some((j2, pole)) = &self.solar_j2 {
                let u = self.q[i].val.dot(pole) / r;
                energy += 0.5 * GMS * j2 * sun_r2 * gm * (3.0 * u * u - 1.0) / (r * r * r);
            }
            if self.emb_idx == Some(i) {
                let u = self.q[i].val.dot(&self.ecliptic_pole) / r;
                energy += 0.5 * GMS * *EMB_QUAD_J2R2 * gm * (3.0 * u * u - 1.0) / (r * r * r);
            }
            for j in (i + 1)..n {
                let sep = (self.q[j].val - self.q[i].val).norm();
                energy -= gm * self.gms[j] / sep;
            }
        }
        energy + 0.5 * momentum.norm_squared() / GMS
    }

    /// Total angular momentum of the massive system multiplied by the
    /// gravitational constant, in AU^5 / Day^3, expressed on the axes of the
    /// frame `T`.
    ///
    /// Every sub-flow of the map is rotationally invariant, so this is
    /// conserved to roundoff (the center-of-mass contribution is excluded and
    /// separately constant). The solar J2 term is symmetric about the solar
    /// spin pole and the lunar quadrupole on an Earth-Moon barycenter about
    /// the ecliptic pole; each conserves only the component along its own
    /// axis, since the reaction torques on the solar spin and on the lunar
    /// orbit are not modeled. With both enabled no component is exactly
    /// conserved. With the GR term enabled the magnitude oscillates at the
    /// tiny 1PN scale over each orbit with no secular trend.
    #[must_use]
    pub fn angular_momentum(&self) -> Vector3<f64> {
        let mut total = Vector3::zeros();
        for ((gm, pos), vel) in self.gms.iter().zip(&self.q).zip(&self.v) {
            total += pos.val.cross(&vel.val) * *gm;
        }
        total
    }

    /// Current barycentric states of the Sun and all massive bodies, in the
    /// frame and origin of the input states. The Sun is the first entry.
    #[must_use]
    pub fn massive_states(&self) -> Vec<State<T, SSB>> {
        let epoch = self.epoch();
        let (sun_pos, sun_vel) = self.sun_pos_vel();
        let mut out = Vec::with_capacity(self.gms.len() + 1);
        out.push(State::new(
            self.sun_desig.clone(),
            epoch,
            sun_pos,
            sun_vel,
            SSB,
        ));
        for ((desig, pos), vel) in self.desigs.iter().zip(&self.q).zip(&self.v) {
            out.push(State::new(
                desig.clone(),
                epoch,
                pos.val + sun_pos,
                vel.val + self.com_vel,
                SSB,
            ));
        }
        out
    }

    /// Current barycentric states of the active test particles, in the frame
    /// and origin of the input states.
    #[must_use]
    pub fn test_particle_states(&self) -> Vec<State<T, SSB>> {
        let epoch = self.epoch();
        let (sun_pos, _) = self.sun_pos_vel();
        self.tp_desigs
            .iter()
            .zip(&self.tp_q)
            .zip(&self.tp_v)
            .map(|((desig, pos), vel)| {
                State::new(
                    desig.clone(),
                    epoch,
                    pos.val + sun_pos,
                    vel.val + self.com_vel,
                    SSB,
                )
            })
            .collect()
    }

    /// Heliocentric positions (AU) and velocities (AU/day) of the active test
    /// particles, on the axes of frame `T`, in the order of
    /// [`Self::test_particle_states`].
    #[must_use]
    pub fn test_particle_heliocentric(&self) -> Vec<(Vector3<f64>, Vector3<f64>)> {
        let sun_vel = self.sun_internal_vel();
        self.tp_q
            .iter()
            .zip(&self.tp_v)
            .map(|(pos, vel)| (pos.val, vel.val - sun_vel))
            .collect()
    }

    /// Designations of the active test particles, in order.
    #[must_use]
    pub fn test_particle_desigs(&self) -> &[Desig] {
        &self.tp_desigs
    }

    /// Remove the test particles at the given indices (into the current
    /// order) and return their barycentric states, in the order the indices
    /// were given. The remaining particles keep their relative order.
    ///
    /// # Errors
    ///
    /// Returns an error if an index is out of range or repeated.
    pub fn remove_test_particles(&mut self, indices: &[usize]) -> KeteResult<Vec<State<T, SSB>>> {
        let n = self.tp_q.len();
        let mut remove = vec![false; n];
        for &idx in indices {
            if idx >= n {
                Err(Error::ValueError(format!(
                    "Test particle index {idx} is out of range for {n} particles."
                )))?;
            }
            if remove[idx] {
                Err(Error::ValueError(format!(
                    "Test particle index {idx} is repeated."
                )))?;
            }
            remove[idx] = true;
        }
        let epoch = self.epoch();
        let (sun_pos, _) = self.sun_pos_vel();
        let removed = indices
            .iter()
            .map(|&idx| {
                State::new(
                    self.tp_desigs[idx].clone(),
                    epoch,
                    self.tp_q[idx].val + sun_pos,
                    self.tp_v[idx].val + self.com_vel,
                    SSB,
                )
            })
            .collect();
        let flags: Vec<Option<()>> = remove.iter().map(|&r| r.then_some(())).collect();
        retain_by_flags(&mut self.tp_desigs, &flags);
        retain_by_flags(&mut self.tp_q, &flags);
        retain_by_flags(&mut self.tp_v, &flags);
        retain_by_flags(&mut self.tp_forces, &flags);
        retain_by_flags(&mut self.tp_rv, &flags);
        Ok(removed)
    }

    /// Append test particles, given as barycentric states at the current
    /// epoch, with optional non-gravitational forces (empty, or one entry per
    /// state, with the same meaning as in [`Self::new`]).
    ///
    /// # Errors
    ///
    /// Returns an error if a state's epoch differs from the current epoch, a
    /// state is non-finite, `non_gravs` has the wrong length, or a non-grav
    /// entry fails the checks of [`Self::new`].
    pub fn add_test_particles(
        &mut self,
        states: &[State<T, SSB>],
        non_gravs: &[Option<NonGravMask>],
    ) -> KeteResult<()> {
        if !(non_gravs.is_empty() || non_gravs.len() == states.len()) {
            Err(Error::ValueError(format!(
                "non_gravs must be empty or have one entry per state, found {} entries for {} \
                 states.",
                non_gravs.len(),
                states.len()
            )))?;
        }
        let epoch = self.epoch();
        for state in states {
            if (state.epoch - epoch).elapsed.abs() > EPOCH_TOL {
                Err(Error::ValueError(format!(
                    "State {:?} has epoch {} which does not match the simulation epoch {}.",
                    state.desig,
                    state.epoch.jd(),
                    epoch.jd()
                )))?;
            }
            if !state.is_finite() {
                Err(Error::ValueError(format!(
                    "State {:?} contains non-finite position or velocity.",
                    state.desig
                )))?;
            }
        }
        let forces = non_gravs
            .iter()
            .zip(states)
            .map(|(frozen, state)| {
                frozen
                    .as_ref()
                    .map(|frozen| bind_non_grav::<T>(frozen, &state.desig))
                    .transpose()
            })
            .collect::<KeteResult<Vec<_>>>()?;
        // `tp_forces` stays empty while no particle carries a force.
        if forces.iter().any(Option::is_some) && self.tp_forces.is_empty() {
            self.tp_forces = vec![None; self.tp_q.len()];
        }
        let (sun_pos, _) = self.sun_pos_vel();
        for (k, state) in states.iter().enumerate() {
            self.tp_desigs.push(state.desig.clone());
            self.tp_q
                .push(CompVec3::new(Vector3::from(state.pos) - sun_pos));
            self.tp_v
                .push(CompVec3::new(Vector3::from(state.vel) - self.com_vel));
            if !self.tp_forces.is_empty() {
                self.tp_forces.push(forces.get(k).cloned().flatten());
            }
            self.tp_rv.push(f64::NAN);
        }
        Ok(())
    }

    /// Add a velocity change (AU/day, on the axes of frame `T`) to every active
    /// test particle, in order: an instantaneous impulse such as a passing
    /// star's.
    ///
    /// # Errors
    ///
    /// Returns an error if `dv` does not have one entry per particle or any
    /// entry is non-finite.
    pub fn kick_test_particles(&mut self, dv: &[Vector3<f64>]) -> KeteResult<()> {
        if dv.len() != self.tp_v.len() {
            Err(Error::ValueError(format!(
                "dv has {} entries for {} test particles.",
                dv.len(),
                self.tp_v.len()
            )))?;
        }
        if dv.iter().any(|d| !d.iter().all(|x| x.is_finite())) {
            Err(Error::ValueError("dv contains non-finite values.".into()))?;
        }
        for (vel, d) in self.tp_v.iter_mut().zip(dv) {
            vel.add(d);
        }
        Ok(())
    }

    /// Apply the order-17 symplectic corrector (`inv = 1.0`) or its inverse
    /// (`inv = -1.0`). See Wisdom (2006); the stage structure and coefficients
    /// follow the `WHFast` reference implementation.
    fn apply_corrector(&mut self, inv: f64) -> KeteResult<()> {
        let alpha = CORRECTOR_A_BASE.sqrt() * self.dt;
        #[allow(clippy::cast_precision_loss, reason = "index below 8")]
        for (j, b) in CORRECTOR_B17.iter().enumerate() {
            let a = ((8 - j) as f64) * alpha;
            self.corrector_z(-a, -inv * b * self.dt)?;
        }
        #[allow(clippy::cast_precision_loss, reason = "index below 8")]
        for (j, b) in CORRECTOR_B17.iter().rev().enumerate() {
            let a = ((j + 1) as f64) * alpha;
            self.corrector_z(a, inv * b * self.dt)?;
        }
        Ok(())
    }

    /// The elementary corrector operator `Z(a, b) = X(a) Y(-b) X(-2a) Y(b) X(a)`
    /// where `X` is the Kepler flow and `Y` the perturbation flow.
    fn corrector_z(&mut self, a: f64, b: f64) -> KeteResult<()> {
        self.kepler(a, None)?;
        self.perturbation(-b)?;
        self.kepler(-2.0 * a, None)?;
        self.perturbation(b)?;
        self.kepler(a, None)
    }

    /// The full perturbation flow (everything except the Kepler part) for
    /// time `b`, as the symmetric split `jump(b/2) kick(b) jump(b/2)`. The
    /// jump/kick composition error is one order higher in the planet masses
    /// than anything the corrector corrects. This is not a step of the map, so
    /// its kick neither resolves nor records encounters.
    fn perturbation(&mut self, b: f64) -> KeteResult<()> {
        self.jump(0.5 * b);
        self.kick(b, false)?;
        self.jump(0.5 * b);
        Ok(())
    }

    /// Interaction kick: velocity updates from all non-solar pairwise gravity
    /// (plus the optional GR and J2 terms), positions unchanged.
    ///
    /// In a step of the map (`map_step`) it also tracks sub-3-Hill-radius
    /// approaches, and with close encounters enabled, test particles that pass
    /// within the encounter radius of a massive body during the step skip the
    /// kick and have the whole step integrated by [`Self::resolve_encounters`].
    /// The corrector's kicks act at positions the run never visits, so they do
    /// neither.
    ///
    /// # Errors
    ///
    /// Returns an error if the massive bodies' states at the ends of the step
    /// cannot be reconstructed for a resolved encounter.
    fn kick(&mut self, dt: f64, map_step: bool) -> KeteResult<()> {
        let epoch = self.epoch();
        let encounter_radius = if map_step {
            self.encounter_radius
        } else {
            None
        };
        let n = self.gms.len();
        // Velocities and total momentum before the kick, which a resolved
        // encounter needs to rebuild the state at the start of the step.
        let pre_kick = encounter_radius.map(|_| {
            let vel: Vec<Vector3<f64>> = self.v.iter().map(|v| v.val).collect();
            let momentum = vel.iter().zip(&self.gms).map(|(v, gm)| v * *gm).sum();
            (vel, momentum)
        });

        // Sun velocity frozen at the start of the kick, used to form the
        // heliocentric velocity the dust drag depends on. Captured before the
        // massive velocities are updated below.
        let sun_vel = self.sun_internal_vel();

        self.update_hill_radii();
        for accel in &mut self.accel {
            *accel = Vector3::zeros();
        }

        // Massive-massive interactions. Below the threshold, iterate the
        // O(n^2 / 2) pairs serially; above it, compute the full O(n^2) sum
        // row-parallel (twice the flops, but each row is independent and the
        // result stays deterministic since every row sums serially in order).
        let worst = if n > PARALLEL_THRESHOLD {
            // Borrowed field by field so the accelerations can be written in
            // parallel while the positions and GMs are read.
            let (q, gms, hill_r, hill_thresh) =
                (&self.q, &self.gms, &self.hill_r, &self.hill_thresh);
            self.accel
                .par_iter_mut()
                .enumerate()
                .with_min_len(4)
                .map(|(i, accel)| {
                    let pos_i = q[i].val;
                    let mut best = NO_ENCOUNTER;
                    for (j, (pos_j, gm_j)) in q.iter().zip(gms).enumerate() {
                        if j == i {
                            continue;
                        }
                        let sep = pos_j.val - pos_i;
                        let r2 = sep.norm_squared();
                        let inv_r3 = (r2 * r2.sqrt()).recip();
                        *accel += sep * (*gm_j * inv_r3);
                        if j > i && r2 < hill_thresh[i].max(hill_thresh[j]) {
                            let ratio = r2.sqrt() / hill_r[i].max(hill_r[j]);
                            if ratio < best.0 {
                                best = (ratio, i, j);
                            }
                        }
                    }
                    best
                })
                .reduce(|| NO_ENCOUNTER, |a, b| if a.0 <= b.0 { a } else { b })
        } else {
            let mut worst = NO_ENCOUNTER;
            for i in 0..n {
                let pos_i = self.q[i].val;
                for j in (i + 1)..n {
                    let sep = self.q[j].val - pos_i;
                    let r2 = sep.norm_squared();
                    let inv_r3 = (r2 * r2.sqrt()).recip();
                    self.accel[i] += sep * (self.gms[j] * inv_r3);
                    self.accel[j] -= sep * (self.gms[i] * inv_r3);
                    if r2 < self.hill_thresh[i].max(self.hill_thresh[j]) {
                        let ratio = r2.sqrt() / self.hill_r[i].max(self.hill_r[j]);
                        if ratio < worst.0 {
                            worst = (ratio, i, j);
                        }
                    }
                }
            }
            worst
        };
        if self.include_gr {
            for i in 0..n {
                let v_helio = self.v[i].val - sun_vel;
                apply_gr_correction(&mut self.accel[i], &self.q[i].val, &v_helio, GMS);
            }
        }
        if let Some((j2, pole)) = &self.solar_j2 {
            let sun_radius = *SUN_RADIUS_AU;
            for i in 0..n {
                self.accel[i] += j2_correction(&self.q[i].val, pole, sun_radius, *j2, GMS);
            }
        }
        // Orbit-averaged lunar quadrupole on the Earth-Moon barycenter (Quinn,
        // Tremaine & Duncan 1991): the J2 form about the ecliptic pole, with
        // the effective `J2 R^2` carried entirely by the coefficient (radius 1).
        if let Some(i) = self.emb_idx {
            self.accel[i] += j2_correction(
                &self.q[i].val,
                &self.ecliptic_pole,
                1.0,
                *EMB_QUAD_J2R2,
                GMS,
            );
        }
        for i in 0..n {
            self.v[i].add(&(self.accel[i] * dt));
        }

        // Test particles, in parallel.
        let (q, gms, hill_r, hill_thresh) = (&self.q, &self.gms, &self.hill_r, &self.hill_thresh);
        let body_vel = &self.v;
        let include_gr = self.include_gr;
        let solar_j2 = self.solar_j2;
        let tp_forces = &self.tp_forces;
        let skip = &self.corrector_skip;
        let check = encounter_radius.map(|k| EncounterCheck::new(q, body_vel, hill_r, k, dt));
        let tp_min_chunk = tp_chunk(self.tp_q.len(), 32);
        // Closest unresolved approach, and the (particle, body) pairs whose
        // step is resolved instead of kicked.
        let (tp_worst, flagged) = self
            .tp_q
            .par_iter()
            .zip(self.tp_v.par_iter_mut())
            .enumerate()
            .with_min_len(tp_min_chunk)
            .fold(
                || (NO_ENCOUNTER, Vec::new()),
                |(mut worst, mut flagged), (idx, (pos, vel))| {
                    // `get` covers the usual case, where `skip` is empty.
                    if let Some(Some(_)) = skip.get(idx) {
                        return (worst, flagged);
                    }
                    let mut accel = Vector3::zeros();
                    let mut nearest = NO_ENCOUNTER;
                    // The encounter test, when enabled, shares the gravity
                    // loop's separations; only a pair inside its distance
                    // bound gets the full test.
                    let check = check
                        .as_ref()
                        .map(|check| (check, check.particle(&pos.val, &vel.val)));
                    let before = flagged.len();
                    for (j, (body_pos, gm)) in q.iter().zip(gms).enumerate() {
                        let sep = body_pos.val - pos.val;
                        let r2 = sep.norm_squared();
                        let inv_r3 = (r2 * r2.sqrt()).recip();
                        accel += sep * (*gm * inv_r3);
                        if r2 < hill_thresh[j] {
                            let ratio = r2.sqrt() / hill_r[j];
                            if ratio < nearest.0 {
                                nearest = (ratio, j, idx);
                            }
                        }
                        if let Some((check, terms)) = &check
                            && check.flags(j, &sep, r2, &(body_vel[j].val - vel.val), terms)
                        {
                            flagged.push((idx, j));
                        }
                    }
                    if flagged.len() > before {
                        return (worst, flagged);
                    }
                    if nearest.0 < worst.0 {
                        worst = nearest;
                    }
                    let u = vel.val - sun_vel;
                    accel += field_accel(&pos.val, &u, include_gr, solar_j2.as_ref());
                    // `get` covers the gravity-only case, where `tp_forces` is empty.
                    match tp_forces.get(idx) {
                        // Dust: radiation pressure is folded into the reduced-mu
                        // drift; the kick applies the exact Poynting-Robertson
                        // drag flow, linear in the heliocentric velocity.
                        Some(Some(NonGrav::Dust { beta })) => {
                            let k = beta * GMS / (pos.val.norm_squared() * C_AU_PER_DAY);
                            vel.add(&drag_flow(dt, k, &pos.val, &u, &accel));
                        }
                        Some(Some(force)) => {
                            accel += non_grav_accel(force, &pos.val, &u);
                            vel.add(&(accel * dt));
                        }
                        _ => vel.add(&(accel * dt)),
                    }
                    (worst, flagged)
                },
            )
            .reduce(
                || (NO_ENCOUNTER, Vec::new()),
                |(worst_a, mut flagged_a), (worst_b, flagged_b)| {
                    flagged_a.extend(flagged_b);
                    (
                        if worst_a.0 <= worst_b.0 {
                            worst_a
                        } else {
                            worst_b
                        },
                        flagged_a,
                    )
                },
            );

        if map_step && worst.0.is_finite() {
            let (ratio, i, j) = worst;
            self.record_encounter(ratio, i, self.desigs[j].clone(), epoch);
        }
        if map_step && tp_worst.0.is_finite() {
            let (ratio, j, idx) = tp_worst;
            self.record_encounter(ratio, j, self.tp_desigs[idx].clone(), epoch);
        }
        if let Some((pre_vel, pre_momentum)) = pre_kick
            && !flagged.is_empty()
        {
            self.resolve_encounters(&flagged, dt, &pre_vel, &pre_momentum)?;
        }
        Ok(())
    }

    /// Integrate the step of each flagged test particle with Radau; see
    /// [`Self::with_close_encounters`]. Called from the kick, after the
    /// massive bodies' kick, with their velocities and total momentum from
    /// before it; the flagged particles have not been kicked. `flagged` holds
    /// `(particle, body)` pairs, grouped by particle in increasing order, for
    /// every body the particle passes near.
    ///
    /// In the map the particle's state at the kick is
    /// `jump(dt/2) kepler(dt/2)` of its state at the start of the step, the
    /// jump using the momentum before the kick, and after the kick the map
    /// applies `kepler(dt/2) jump(dt/2)` with the momentum after it. Both are
    /// inverted exactly here, so the particle's end state is what the map
    /// reaches at the end of the step.
    ///
    /// # Errors
    ///
    /// Returns an error if a massive body's Kepler drift to either end of the
    /// step fails.
    fn resolve_encounters(
        &mut self,
        flagged: &[(usize, usize)],
        dt: f64,
        pre_vel: &[Vector3<f64>],
        pre_momentum: &Vector3<f64>,
    ) -> KeteResult<()> {
        let start = self.epoch();
        let post_momentum: Vector3<f64> = self
            .v
            .iter()
            .zip(&self.gms)
            .map(|(v, gm)| v.val * *gm)
            .sum();
        let shift_pre = pre_momentum * (0.5 * dt / GMS);
        let shift_post = post_momentum * (0.5 * dt / GMS);

        let mut ends = Vec::with_capacity(self.gms.len());
        for (j, (pos, vel)) in self.q.iter().zip(&self.v).enumerate() {
            let start_state = drift_vec(-0.5 * dt, &(pos.val - shift_pre), &pre_vel[j], GMS)?;
            let end_state = drift_vec(0.5 * dt, &(pos.val + shift_post), &vel.val, GMS)?;
            ends.push((start_state, end_state));
        }
        // Heliocentric velocity = barycentric velocity - Sun's velocity, the
        // Sun's being -momentum / GMS with the momentum at that instant (the
        // Kepler drifts change it, so it is not the momentum at the kick).
        let sun_start = -ends
            .iter()
            .zip(&self.gms)
            .map(|(((_, v0), _), gm)| v0 * *gm)
            .sum::<Vector3<f64>>()
            / GMS;
        let sun_end = -ends
            .iter()
            .zip(&self.gms)
            .map(|((_, (_, v1)), gm)| v1 * *gm)
            .sum::<Vector3<f64>>()
            / GMS;
        let paths = ends
            .iter()
            .zip(&self.gms)
            .map(|(((q0, v0), (q1, v1)), gm)| {
                BodyPath::new(*q0, v0 - sun_start, *q1, v1 - sun_end, GMS + gm, dt)
            })
            .collect::<KeteResult<Vec<_>>>()?;

        // One entry per particle, with the bodies it passes near.
        let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
        for &(idx, j) in flagged {
            match groups.last_mut() {
                Some((last, bodies)) if *last == idx => bodies.push(j),
                _ => groups.push((idx, vec![j])),
            }
        }

        let ids: Vec<i32> = self
            .desigs
            .iter()
            .map(|d| d.clone().naif_id().unwrap_or(-1))
            .collect();
        let (include_gr, solar_j2) = (self.include_gr, self.solar_j2);
        let (peri_bound, dist_bound) = (self.peri_bound, self.dist_bound);
        let (gms, radii) = (&self.gms, &self.radii);
        let (tp_q, tp_v, tp_forces) = (&self.tp_q, &self.tp_v, &self.tp_forces);
        let sun_radius2 = SUN_RADIUS_AU.powi(2);
        let outcomes: Vec<(usize, EncounterOutcome)> = groups
            .par_iter()
            .map(|(idx, near_bodies)| {
                let idx = *idx;
                let mut near = vec![false; paths.len()];
                for &j in near_bodies {
                    near[j] = true;
                }
                let force = tp_forces.get(idx).and_then(Option::as_ref);
                let mu = match force {
                    Some(NonGrav::Dust { beta }) => (1.0 - beta) * GMS,
                    _ => GMS,
                };
                let Ok((q0, v0)) =
                    drift_vec(-0.5 * dt, &(tp_q[idx].val - shift_pre), &tp_v[idx].val, mu)
                else {
                    return (
                        idx,
                        EncounterOutcome::Lost(LostReason::KeplerFailure, start),
                    );
                };
                let u0 = v0 - sun_start;
                let impact = Cell::new(None);
                // The perihelion bound follows the integrated path: a passage
                // is a turn from inward to outward in the heliocentric `r . u`
                // between accepted states, inside the bound. A turn across
                // the step's ends alone can be a flyby's deflection.
                let sign = dt.signum();
                let last_rv = Cell::new(f64::NAN);
                let passage = Cell::new(None);
                let func = |time: Time<TDB>,
                            pos: &Vector3<f64>,
                            u: &Vector3<f64>,
                            (): &mut (),
                            exact: bool|
                 -> KeteResult<Vector3<f64>> {
                    // Impacts are tested at accepted states only, as in the
                    // Radau N-body propagation; a trial state inside a body
                    // is not an impact.
                    let r2 = pos.norm_squared();
                    if exact && r2 < sun_radius2 {
                        impact.set(Some((LostReason::SunImpact, time)));
                        return Err(Error::Impact(10, time));
                    }
                    if exact {
                        let rv = pos.dot(u);
                        if let Some(bound) = peri_bound
                            && passage.get().is_none()
                            && sign * last_rv.get() < 0.0
                            && sign * rv >= 0.0
                            && r2 < bound * bound
                        {
                            passage.set(Some(time));
                        }
                        last_rv.set(rv);
                    }
                    let mut accel = -pos * (mu / (r2 * r2.sqrt()));
                    let tau = (time - start).elapsed;
                    for (j, path) in paths.iter().enumerate() {
                        // The two-body path for the bodies the particle passes
                        // near; the plain cubic for the rest, whose pull is
                        // too weak for the difference to matter.
                        let body = if near[j] {
                            path.position(tau)?
                        } else {
                            path.cubic(tau)
                        };
                        let sep = body - pos;
                        let d2 = sep.norm_squared();
                        if exact && d2 < radii[j] * radii[j] {
                            impact.set(Some((LostReason::PlanetImpact, time)));
                            return Err(Error::Impact(ids[j], time));
                        }
                        let body_r2 = body.norm_squared();
                        // Direct pull, minus the body's pull on the Sun (the
                        // indirect term of the heliocentric frame).
                        accel += sep * (gms[j] / (d2 * d2.sqrt()))
                            - body * (gms[j] / (body_r2 * body_r2.sqrt()));
                    }
                    accel += field_accel(pos, u, include_gr, solar_j2.as_ref());
                    match force {
                        Some(NonGrav::Dust { beta }) => {
                            // Poynting-Robertson drag, the force the map's
                            // drag flow integrates exactly.
                            let k = beta * GMS / (r2 * C_AU_PER_DAY);
                            let r_hat = pos / r2.sqrt();
                            accel -= (u + r_hat * u.dot(&r_hat)) * k;
                        }
                        Some(force) => accel += non_grav_accel(force, pos, u),
                        None => {}
                    }
                    Ok(accel)
                };
                let result =
                    RadauIntegrator::integrate(&func, q0, u0, start, start + dt, (), None, None);
                let Ok((q1, u1, ())) = result else {
                    let (reason, at) = impact
                        .get()
                        .unwrap_or((LostReason::EncounterFailure, start));
                    return (idx, EncounterOutcome::Lost(reason, at));
                };
                // The removal bounds, over the step just integrated.
                if let Some(at) = passage.get() {
                    return (idx, EncounterOutcome::Lost(LostReason::PerihelionBound, at));
                }
                if let Some(bound) = dist_bound
                    && q1.norm_squared() > bound * bound
                {
                    return (
                        idx,
                        EncounterOutcome::Lost(LostReason::DistanceBound, start),
                    );
                }
                // Back to where the map's remaining jump and half drift land on it.
                match drift_vec(-0.5 * dt, &q1, &(u1 + sun_end), mu) {
                    Ok((q_back, v_back)) => (
                        idx,
                        EncounterOutcome::Done {
                            pos: q_back - shift_post,
                            vel: v_back,
                        },
                    ),
                    Err(_) => (
                        idx,
                        EncounterOutcome::Lost(LostReason::KeplerFailure, start),
                    ),
                }
            })
            .collect();

        let mut lost = vec![None; self.tp_q.len()];
        for (idx, outcome) in outcomes {
            self.close_encounter_steps += 1;
            match outcome {
                EncounterOutcome::Done { pos, vel } => {
                    self.tp_q[idx] = CompVec3::new(pos);
                    self.tp_v[idx] = CompVec3::new(vel);
                    // The encounter tested its own step against the bounds,
                    // and a flyby can turn the particle without a passage.
                    self.tp_rv[idx] = f64::NAN;
                }
                EncounterOutcome::Lost(reason, at) => lost[idx] = Some((reason, at)),
            }
        }
        if lost.iter().any(Option::is_some) {
            self.remove_lost(&lost);
        }
        Ok(())
    }

    /// Record an approach as the closest seen so far, if it beats the current
    /// record. `first` indexes the massive body involved.
    fn record_encounter(&mut self, hill_ratio: f64, first: usize, second: Desig, epoch: Time<TDB>) {
        if self
            .closest
            .as_ref()
            .is_none_or(|e| hill_ratio < e.hill_ratio)
        {
            self.closest = Some(Encounter {
                hill_ratio,
                first: self.desigs[first].clone(),
                second,
                epoch,
            });
        }
    }

    /// Solar drift: every heliocentric position shifts by the total massive
    /// momentum divided by the solar mass; velocities unchanged.
    fn jump(&mut self, dt: f64) {
        let mut momentum = Vector3::zeros();
        for (gm, vel) in self.gms.iter().zip(&self.v) {
            momentum += vel.val * *gm;
        }
        let shift = momentum * (dt / GMS);
        for pos in &mut self.q {
            pos.add(&shift);
        }
        let chunk = tp_chunk(self.tp_q.len(), 256);
        let skip = &self.corrector_skip;
        self.tp_q
            .par_iter_mut()
            .enumerate()
            .with_min_len(chunk)
            .for_each(|(idx, pos)| {
                if !matches!(skip.get(idx), Some(Some(_))) {
                    pos.add(&shift);
                }
            });
    }

    /// Kepler drift: every body moves on its exact two-body orbit about the
    /// solar GM for `dt`.
    ///
    /// # Errors
    ///
    /// A massive body which is lost during the drift raises an error; test
    /// particles are moved to [`Self::lost_particles`] instead.
    ///
    /// `map_start` is the time a drift of the map's steps starts, with which
    /// its removals are recorded. It is `None` for the drifts inside the
    /// symplectic corrector, which are not motion of the run: they skip the
    /// removal bounds, and record their removals at the current epoch.
    fn kepler(&mut self, dt: f64, map_start: Option<Time<TDB>>) -> KeteResult<()> {
        let map_step = map_start.is_some();
        let epoch = map_start.unwrap_or_else(|| self.epoch());

        let lost: Vec<Option<LostReason>> = if self.gms.len() > PARALLEL_THRESHOLD {
            self.q
                .par_iter_mut()
                .zip(self.v.par_iter_mut())
                .with_min_len(16)
                .map(|(pos, vel)| drift(dt, pos, vel, GMS))
                .collect()
        } else {
            self.q
                .iter_mut()
                .zip(&mut self.v)
                .map(|(pos, vel)| drift(dt, pos, vel, GMS))
                .collect()
        };
        for (idx, reason) in lost.iter().enumerate() {
            match reason {
                Some(LostReason::SunImpact) => Err(Error::ValueError(format!(
                    "Massive body {:?} hit the Sun at jd = {}, the simulation cannot continue.",
                    self.desigs[idx],
                    epoch.jd()
                )))?,
                Some(LostReason::KeplerFailure) => Err(Error::Convergence(format!(
                    "Kepler drift failed to converge for massive body {:?} at jd = {}. \
                     The state is likely extreme, such as a deep close encounter; \
                     a smaller dt may help.",
                    self.desigs[idx],
                    epoch.jd()
                )))?,
                // The removal bounds and encounter outcomes apply to test particles only.
                Some(
                    LostReason::PerihelionBound
                    | LostReason::DistanceBound
                    | LostReason::PlanetImpact
                    | LostReason::EncounterFailure,
                )
                | None => {}
            }
        }

        let chunk = tp_chunk(self.tp_q.len(), 64);
        let tp_forces = &self.tp_forces;
        let (peri_bound, dist_bound) = (self.peri_bound, self.dist_bound);
        let skip = &self.corrector_skip;
        let tp_lost: Vec<Option<LostReason>> = self
            .tp_q
            .par_iter_mut()
            .zip(self.tp_v.par_iter_mut())
            .zip(self.tp_rv.par_iter_mut())
            .enumerate()
            .with_min_len(chunk)
            .map(|(idx, ((pos, vel), rv_end))| {
                if let Some(Some(_)) = skip.get(idx) {
                    return None;
                }
                // Dust drifts on its radiation-reduced orbit, mu = (1-beta) GMS;
                // `get` covers the gravity-only case, where `tp_forces` is empty.
                let mu = match tp_forces.get(idx) {
                    Some(Some(NonGrav::Dust { beta })) => (1.0 - beta) * GMS,
                    _ => GMS,
                };
                let rv_before = pos.val.dot(&vel.val);
                let lost = drift(dt, pos, vel, mu);
                if lost.is_some() || !map_step {
                    return lost;
                }
                // A turn from inward to outward in the jump and kick since the
                // last drift is a passage too: count this drift as starting
                // inward if the last one ended inward.
                let rv_start = if dt.signum() * *rv_end < 0.0 {
                    *rv_end
                } else {
                    rv_before
                };
                *rv_end = pos.val.dot(&vel.val);
                outside_bounds(dt, rv_start, &pos.val, &vel.val, mu, peri_bound, dist_bound)
            })
            .collect();
        if tp_lost.iter().any(Option::is_some) {
            let lost: Vec<_> = tp_lost.iter().map(|r| r.map(|r| (r, epoch))).collect();
            self.remove_lost(&lost);
        }
        Ok(())
    }

    /// Record and remove the test particles marked in `lost`, each with the
    /// reason and time of its removal.
    fn remove_lost(&mut self, lost: &[Option<(LostReason, Time<TDB>)>]) {
        for (idx, entry) in lost.iter().enumerate() {
            if let Some((reason, epoch)) = entry {
                self.lost.push(LostParticle {
                    desig: self.tp_desigs[idx].clone(),
                    epoch: *epoch,
                    reason: *reason,
                });
            }
        }
        retain_by_flags(&mut self.tp_desigs, lost);
        retain_by_flags(&mut self.tp_q, lost);
        retain_by_flags(&mut self.tp_v, lost);
        // Stays empty when unused; `retain` never calls the closure on an empty
        // vector, so the gravity-only case needs no special case here.
        retain_by_flags(&mut self.tp_forces, lost);
        retain_by_flags(&mut self.corrector_skip, lost);
        retain_by_flags(&mut self.tp_rv, lost);
    }

    /// Elapsed time in days since the initial epoch, computed from the step
    /// count rather than accumulated.
    fn elapsed(&self) -> f64 {
        #[allow(
            clippy::cast_precision_loss,
            reason = "step counts stay far below 2^52"
        )]
        {
            (self.steps as f64) * self.dt
        }
    }

    /// The Sun's velocity in the internal (center of mass) frame.
    fn sun_internal_vel(&self) -> Vector3<f64> {
        let mut momentum = Vector3::zeros();
        for (gm, vel) in self.gms.iter().zip(&self.v) {
            momentum += vel.val * *gm;
        }
        -momentum / GMS
    }

    /// Reconstruct the Sun's barycentric position and velocity in the frame
    /// and origin of the input states.
    fn sun_pos_vel(&self) -> (Vector3<f64>, Vector3<f64>) {
        let mut weighted_q = Vector3::zeros();
        for (gm, pos) in self.gms.iter().zip(&self.q) {
            weighted_q += pos.val * *gm;
        }
        let com_pos = self.com_pos0 + self.com_vel * self.elapsed();
        let sun_pos = com_pos - weighted_q / self.total_gm;
        let sun_vel = self.com_vel + self.sun_internal_vel();
        (sun_pos, sun_vel)
    }
}

/// A 3-vector whose in-place additions use Neumaier compensated summation.
///
/// Every state update in the map is an addition of a small increment onto a
/// larger value; retaining the roundoff of each addition keeps the accumulated
/// state error a random walk instead of a linear drift over billions of steps.
#[derive(Debug, Clone, Copy)]
struct CompVec3 {
    /// Current value.
    val: Vector3<f64>,
    /// Roundoff retained from previous additions.
    comp: Vector3<f64>,
}

impl CompVec3 {
    fn new(val: Vector3<f64>) -> Self {
        Self {
            val,
            comp: Vector3::zeros(),
        }
    }

    /// Add `delta` to the vector, retaining the roundoff of the addition.
    fn add(&mut self, delta: &Vector3<f64>) {
        for k in 0..3 {
            let x = self.val[k];
            let d = delta[k] + self.comp[k];
            let sum = x + d;
            self.comp[k] = if x.abs() >= d.abs() {
                (x - sum) + d
            } else {
                (d - sum) + x
            };
            self.val[k] = sum;
        }
    }
}

/// Destructure a frozen non-gravitational force into the map's internal
/// per-particle binding, validating it in the process.
///
/// The surface description is rebuilt through [`FarnocchiaNonGrav::new`],
/// since its fields are public and a struct literal could otherwise carry an
/// unnormalized spin pole or an invalid scalar past the constructor's checks.
/// The validated equatorial spin pole is rotated into the map's frame `T`
/// once here, so the kick's inner loop needs no rotation.
///
/// # Errors
///
/// Returns an error if any parameter is left free, any fixed value is non-finite or out of range (a NaN left free for orbit
/// fitting cannot be simulated), the surface or `g(r)` description is
/// invalid, or a [`NonGravKind::JplComet`] carries a nonzero time lag `dt`.
fn bind_non_grav<T: StaticFrame>(frozen: &NonGravMask, desig: &Desig) -> KeteResult<NonGrav> {
    let values = frozen
        .fixed_values()
        .map_err(|err| Error::ValueError(format!("Test particle {desig:?}: {err}")))?;
    match frozen.inner() {
        NonGravKind::Farnocchia(force) => {
            let &[a_over_m, lambda_0] = values else {
                return Err(Error::ValueError(format!(
                    "Test particle {desig:?}: a frozen Farnocchia force requires exactly 2 \
                     values (a_over_m, lambda_0), found {}.",
                    values.len()
                )));
            };
            for (name, value) in [("a_over_m", a_over_m), ("lambda_0", lambda_0)] {
                if !value.is_finite() || value < 0.0 {
                    return Err(Error::ValueError(format!(
                        "Test particle {desig:?}: '{name}' must be finite and >= 0, found \
                         {value}. A NaN (fit-free) parameter cannot be simulated.",
                    )));
                }
            }
            let validated = FarnocchiaNonGrav::new(
                force.albedo,
                force.absorptivity,
                force.flattening,
                force.spin_pole,
            )
            .map_err(|err| Error::ValueError(format!("Test particle {desig:?}: {err}")))?;
            Ok(NonGrav::Yarkovsky(Yarkovsky {
                albedo: validated.albedo,
                absorptivity: validated.absorptivity,
                flattening: validated.flattening,
                spin_pole: validated.spin_pole.into_frame::<T>().into(),
                a_over_m,
                lambda_0,
            }))
        }
        NonGravKind::Dust(_) => {
            let &[beta] = values else {
                return Err(Error::ValueError(format!(
                    "Test particle {desig:?}: a frozen dust force requires exactly 1 value \
                     (beta), found {}.",
                    values.len()
                )));
            };
            if !(beta.is_finite() && (0.0..1.0).contains(&beta)) {
                return Err(Error::ValueError(format!(
                    "Test particle {desig:?}: beta = {beta} is outside [0, 1); grains with \
                     beta >= 1 are unbound and not supported.",
                )));
            }
            Ok(NonGrav::Dust { beta })
        }
        NonGravKind::JplComet(force) => {
            let &[a1, a2, a3] = values else {
                return Err(Error::ValueError(format!(
                    "Test particle {desig:?}: a frozen A1/A2/A3 force requires exactly 3 \
                     values, found {}.",
                    values.len()
                )));
            };
            for (name, value) in [("a1", a1), ("a2", a2), ("a3", a3)] {
                if !value.is_finite() {
                    return Err(Error::ValueError(format!(
                        "Test particle {desig:?}: '{name}' must be finite, found {value}. \
                         A NaN (fit-free) parameter cannot be simulated.",
                    )));
                }
            }
            for (name, value) in [
                ("alpha", force.alpha),
                ("r_0", force.r_0),
                ("m", force.m),
                ("n", force.n),
                ("k", force.k),
            ] {
                if !value.is_finite() {
                    return Err(Error::ValueError(format!(
                        "Test particle {desig:?}: g(r) shape parameter '{name}' must be \
                         finite, found {value}.",
                    )));
                }
            }
            if force.r_0 <= 0.0 {
                return Err(Error::ValueError(format!(
                    "Test particle {desig:?}: g(r) reference distance r_0 must be positive, \
                     found {}.",
                    force.r_0
                )));
            }
            if force.dt != 0.0 {
                return Err(Error::ValueError(format!(
                    "Test particle {desig:?}: the time-lagged (dt != 0) outgassing form is \
                     not supported by the symplectic map; propagate such objects with the \
                     Radau N-body propagator.",
                )));
            }
            Ok(NonGrav::JplComet {
                force: force.clone(),
                a1,
                a2,
                a3,
            })
        }
    }
}

/// Kepler drift a single body about the central `mu` for `dt`, returning why
/// the body was lost, if it was. Used for massive bodies and plain test
/// particles (`mu = GMS`) and for dust grains (`mu = (1 - beta) GMS`, gravity
/// minus radiation pressure).
fn drift(dt: f64, pos: &mut CompVec3, vel: &mut CompVec3, mu: f64) -> Option<LostReason> {
    let rv_before = pos.val.dot(&vel.val);
    match analytic_2_body_delta(dt, &pos.val, &vel.val, mu) {
        Ok((d_pos, d_vel)) => {
            pos.add(&d_pos);
            vel.add(&d_vel);
            hit_the_sun(dt, rv_before, &pos.val, &vel.val, mu).then_some(LostReason::SunImpact)
        }
        // A solver failure on an orbit whose perihelion is inside the Sun is an
        // impact, not a numerical mystery.
        Err(_) if compute_peri_dist(&pos.val, &vel.val, mu) < *SUN_RADIUS_AU => {
            Some(LostReason::SunImpact)
        }
        Err(_) => Some(LostReason::KeplerFailure),
    }
}

/// Exact velocity increment over `dt` for the Poynting-Robertson drag kick.
///
/// With the position frozen, the heliocentric-velocity ODE is linear,
/// `u_dot = a_pos - k (u + (u . r_hat) r_hat)`, whose operator `k(I + r_hat
/// r_hat^T)` has eigenvalue `2k` along `r_hat` and `k` perpendicular. The exact
/// flow is therefore two scalar exponential decays applied to the radial and
/// tangential parts (Strang-exact for constant `k` over the step). `u` is the
/// heliocentric velocity; the returned increment is added to the barycentric
/// velocity, and the frozen Sun velocity cancels in the difference.
///
/// The `(1 - e^{-x})/rate` coefficients use `exp_m1` since `k dt` is tiny
/// (~1e-6); they tend to `dt` as `k -> 0`, so `k = 0` (a `beta = 0` grain)
/// recovers the plain additive kick `a_pos dt` exactly.
fn drag_flow(
    dt: f64,
    k: f64,
    pos: &Vector3<f64>,
    u: &Vector3<f64>,
    a_pos: &Vector3<f64>,
) -> Vector3<f64> {
    if k <= 0.0 {
        return a_pos * dt;
    }
    let r_hat = pos.normalize();
    let u_r = u.dot(&r_hat);
    let a_r = a_pos.dot(&r_hat);
    let u_perp = u - u_r * r_hat;
    let a_perp = a_pos - a_r * r_hat;
    let decay_r = (-2.0 * k * dt).exp();
    let decay_t = (-k * dt).exp();
    let ramp_r = -(-2.0 * k * dt).exp_m1() / (2.0 * k); // (1 - decay_r) / (2k)
    let ramp_t = -(-k * dt).exp_m1() / k; // (1 - decay_t) / k
    let u_r_new = decay_r * u_r + ramp_r * a_r;
    let u_perp_new = decay_t * u_perp + ramp_t * a_perp;
    (u_r_new * r_hat + u_perp_new) - u
}

/// Whether a Kepler drift from `rv_before = r.v` to the given end state hit the
/// Sun: either it ended inside the solar radius, or it crossed perihelion during
/// the step (radial velocity flipped inbound to outbound, in the direction of
/// `dt`) with the orbit's perihelion inside the solar radius. The perihelion
/// distance is exact for the drift since it is a conserved property of the
/// orbit about `mu` (the radiation-reduced gravity for a dust grain).
fn hit_the_sun(dt: f64, rv_before: f64, pos: &Vector3<f64>, vel: &Vector3<f64>, mu: f64) -> bool {
    let sun_radius = *SUN_RADIUS_AU;
    if pos.norm_squared() < sun_radius * sun_radius {
        return true;
    }
    let rv_after = pos.dot(vel);
    let (rv_in, rv_out) = if dt > 0.0 {
        (rv_before, rv_after)
    } else {
        (rv_after, rv_before)
    };
    rv_in < 0.0 && rv_out > 0.0 && compute_peri_dist(pos, vel, mu) < sun_radius
}

/// The test of whether a test particle passes near a massive body during a
/// step, made at one instant of the step with half a step either side.
///
/// The relative motion is taken as a straight line, and the encounter radius
/// is widened by how far the Sun's differential pull on the pair can bend it
/// in half a step, `|a_sun(body) - a_sun(particle)| (dt/2)^2 / 2`, which
/// matters for fast passages near the inner planets. A bound on the distance
/// the pair can close in half a step rejects most pairs before the full test.
#[derive(Debug)]
struct EncounterCheck {
    /// Encounter radius per massive body, AU.
    radius: Vec<f64>,
    /// Per massive body, its share of the distance bound: the encounter
    /// radius, the distance it moves in half a step, and how far the Sun can
    /// bend its path, AU.
    reach: Vec<f64>,
    /// Solar acceleration of each massive body, AU/day^2.
    sun_accel: Vec<Vector3<f64>>,
    /// Half the step squared, over 2: the bending factor, day^2.
    bend: f64,
    /// Half the step, days.
    half: f64,
}

/// Per-particle terms of an [`EncounterCheck`].
#[derive(Debug)]
struct ParticleTerms {
    /// The particle's share of the distance bound, AU.
    reach: f64,
    /// Solar acceleration of the particle, AU/day^2.
    sun_accel: Vector3<f64>,
}

impl EncounterCheck {
    /// The test for encounters within `hill_radii` of each body's Hill radius
    /// `hill_r`, with the bodies at the given positions and velocities.
    fn new(q: &[CompVec3], v: &[CompVec3], hill_r: &[f64], hill_radii: f64, dt: f64) -> Self {
        let half = 0.5 * dt.abs();
        let bend = 0.5 * half * half;
        let radius: Vec<f64> = hill_r.iter().map(|r| hill_radii * r).collect();
        let sun_accel: Vec<Vector3<f64>> = q.iter().map(|pos| solar_accel(&pos.val)).collect();
        let reach = radius
            .iter()
            .zip(v)
            .zip(&sun_accel)
            .map(|((r, vel), acc)| r + vel.val.norm() * half + bend * acc.norm())
            .collect();
        Self {
            radius,
            reach,
            sun_accel,
            bend,
            half,
        }
    }

    fn particle(&self, pos: &Vector3<f64>, vel: &Vector3<f64>) -> ParticleTerms {
        let sun_accel = solar_accel(pos);
        ParticleTerms {
            reach: vel.norm() * self.half + self.bend * sun_accel.norm(),
            sun_accel,
        }
    }

    /// Whether a particle at squared distance `r2` from body `j` could come
    /// within its encounter radius in half a step; necessary for
    /// [`Self::close`].
    fn within_reach(&self, j: usize, r2: f64, terms: &ParticleTerms) -> bool {
        let bound = self.reach[j] + terms.reach;
        r2 < bound * bound
    }

    /// Whether the particle passes within the encounter radius of body `j`
    /// during the step, given the body's position `sep` relative to it, its
    /// square `r2`, and the body's velocity `rel_vel` relative to it.
    fn flags(
        &self,
        j: usize,
        sep: &Vector3<f64>,
        r2: f64,
        rel_vel: &Vector3<f64>,
        terms: &ParticleTerms,
    ) -> bool {
        self.within_reach(j, r2, terms) && self.close(j, sep, rel_vel, terms)
    }

    /// Whether the particle passes within the encounter radius of body `j`,
    /// given the body's position `sep` and velocity `rel_vel` relative to it.
    fn close(
        &self,
        j: usize,
        sep: &Vector3<f64>,
        rel_vel: &Vector3<f64>,
        terms: &ParticleTerms,
    ) -> bool {
        let w2 = rel_vel.norm_squared();
        let tau = if w2 > 0.0 {
            (-sep.dot(rel_vel) / w2).clamp(-self.half, self.half)
        } else {
            0.0
        };
        let widened = self.radius[j] + self.bend * (self.sun_accel[j] - terms.sun_accel).norm();
        (sep + rel_vel * tau).norm_squared() < widened * widened
    }
}

/// Impact radius (AU) of the body with NAIF id `id`: its radius among the
/// registered masses `selected`, else in the mass table `known`, or 0 if
/// neither gives a positive one. The Earth-Moon barycenter's entry spans the
/// lunar orbit, so it takes the Earth's radius.
fn impact_radius(id: i32, selected: &[GravParams], known: &[GravParams]) -> f64 {
    let id = if id == 3 { 399 } else { id };
    selected
        .iter()
        .chain(known)
        .find(|p| p.naif_id == id && p.radius > 0.0)
        .map_or(0.0, |p| f64::from(p.radius))
}

/// Solar point-mass acceleration at heliocentric position `pos`.
fn solar_accel(pos: &Vector3<f64>) -> Vector3<f64> {
    -pos * (GMS / pos.norm().powi(3))
}

/// Kepler drift of a plain position and velocity about `mu` for `dt`.
fn drift_vec(
    dt: f64,
    pos: &Vector3<f64>,
    vel: &Vector3<f64>,
    mu: f64,
) -> KeteResult<(Vector3<f64>, Vector3<f64>)> {
    let (d_pos, d_vel) = analytic_2_body_delta(dt, pos, vel, mu)?;
    Ok((pos + d_pos, vel + d_vel))
}

/// Accelerations a test particle feels beyond point-mass gravity and its
/// non-gravitational force: the optional solar GR and J2 terms. `u` is the
/// heliocentric velocity.
fn field_accel(
    pos: &Vector3<f64>,
    u: &Vector3<f64>,
    include_gr: bool,
    solar_j2: Option<&(f64, Vector3<f64>)>,
) -> Vector3<f64> {
    let mut accel = Vector3::zeros();
    if include_gr {
        apply_gr_correction(&mut accel, pos, u, GMS);
    }
    if let Some((j2, pole)) = solar_j2 {
        accel += j2_correction(pos, pole, *SUN_RADIUS_AU, *j2, GMS);
    }
    accel
}

/// Acceleration of a particle's non-gravitational force at heliocentric
/// position `pos` and velocity `u`. Zero for a dust grain, whose radiation
/// pressure is in its reduced `mu` and whose drag each caller applies.
fn non_grav_accel(force: &NonGrav, pos: &Vector3<f64>, u: &Vector3<f64>) -> Vector3<f64> {
    match force {
        // Evaluated directly on the heliocentric position, with the pole
        // already in this frame.
        NonGrav::Yarkovsky(yark) => radiation_accel(
            pos,
            &yark.spin_pole,
            yark.albedo,
            yark.absorptivity,
            yark.flattening,
            yark.a_over_m,
            yark.lambda_0,
        ),
        // A1/A2/A3 on the RTN axes of the instantaneous orbit.
        NonGrav::JplComet { force, a1, a2, a3 } => force.accel_no_lag(pos, u, *a1, *a2, *a3),
        NonGrav::Dust { .. } => Vector3::zeros(),
    }
}

/// Heliocentric path of a massive body across one step, which a resolved
/// close encounter integrates against: the body's two-body orbit about
/// `GMS + gm` from its state at the start of the step, plus the cubic
/// Hermite blend that takes that orbit onto the body's state at the end of
/// the step, position and velocity both. The blend is zero with zero slope
/// at the start, so the path matches the map's states at both ends. Bodies
/// the particle does not pass near use the plain cubic Hermite of the end
/// states instead ([`Self::cubic`]), which needs no Kepler solve.
#[derive(Debug)]
struct BodyPath {
    /// Heliocentric position at the start of the step, AU.
    pos: Vector3<f64>,
    /// Heliocentric velocity at the start of the step, AU/day.
    vel: Vector3<f64>,
    /// Two-body `GMS + gm`.
    mu: f64,
    /// Step, days.
    dt: f64,
    /// Heliocentric position at the end of the step, AU.
    end_pos: Vector3<f64>,
    /// Heliocentric velocity at the end of the step, AU/day.
    end_vel: Vector3<f64>,
    /// End-of-step position minus the two-body orbit's.
    d_pos: Vector3<f64>,
    /// End-of-step velocity minus the two-body orbit's.
    d_vel: Vector3<f64>,
}

impl BodyPath {
    fn new(
        pos: Vector3<f64>,
        vel: Vector3<f64>,
        end_pos: Vector3<f64>,
        end_vel: Vector3<f64>,
        mu: f64,
        dt: f64,
    ) -> KeteResult<Self> {
        let (kep_pos, kep_vel) = drift_vec(dt, &pos, &vel, mu)?;
        Ok(Self {
            pos,
            vel,
            mu,
            dt,
            end_pos,
            end_vel,
            d_pos: end_pos - kep_pos,
            d_vel: end_vel - kep_vel,
        })
    }

    /// Heliocentric position `tau` days after the start of the step from the
    /// cubic Hermite interpolant of the two end states alone: no Kepler solve,
    /// and an error of order the orbit's curvature times `(dt / P)^3`.
    fn cubic(&self, tau: f64) -> Vector3<f64> {
        let s = tau / self.dt;
        let (s2, s3) = (s * s, s * s * s);
        self.pos * (2.0 * s3 - 3.0 * s2 + 1.0)
            + self.vel * (self.dt * (s3 - 2.0 * s2 + s))
            + self.end_pos * (3.0 * s2 - 2.0 * s3)
            + self.end_vel * (self.dt * (s3 - s2))
    }

    /// Heliocentric position `tau` days after the start of the step.
    fn position(&self, tau: f64) -> KeteResult<Vector3<f64>> {
        let (d_pos, _) = analytic_2_body_delta(tau, &self.pos, &self.vel, self.mu)?;
        let s = tau / self.dt;
        let (s2, s3) = (s * s, s * s * s);
        Ok(self.pos
            + d_pos
            + self.d_pos * (3.0 * s2 - 2.0 * s3)
            + self.d_vel * (self.dt * (s3 - s2)))
    }
}

/// Result of a resolved close encounter for one test particle.
#[derive(Debug)]
enum EncounterOutcome {
    /// The particle's state at the kick, to be put back in the map.
    Done {
        pos: Vector3<f64>,
        vel: Vector3<f64>,
    },
    /// The particle is removed, for this reason at this time.
    Lost(LostReason, Time<TDB>),
}

/// Whether a drifted test particle crossed a removal bound: beyond the
/// distance bound, or past perihelion with its perihelion inside the
/// perihelion bound. A drift has passed perihelion if its radial velocity
/// turns from inward to outward, if it is longer than half an orbit with both
/// ends moving the same way (it then passed aphelion and perihelion both), or
/// if it spans a whole orbit.
fn outside_bounds(
    dt: f64,
    rv_before: f64,
    pos: &Vector3<f64>,
    vel: &Vector3<f64>,
    mu: f64,
    peri_bound: Option<f64>,
    dist_bound: Option<f64>,
) -> Option<LostReason> {
    if let Some(bound) = dist_bound
        && pos.norm_squared() > bound * bound
    {
        return Some(LostReason::DistanceBound);
    }
    if let Some(bound) = peri_bound {
        let peri = compute_peri_dist(pos, vel, mu);
        if peri < bound {
            let rv_after = pos.dot(vel);
            let (rv_in, rv_out) = if dt > 0.0 {
                (rv_before, rv_after)
            } else {
                (rv_after, rv_before)
            };
            let semi_major = compute_semi_major(pos, vel, mu);
            let long_enough = semi_major > 0.0 && {
                let period = TAU * (semi_major.powi(3) / mu).sqrt();
                dt.abs() >= period || (dt.abs() > 0.5 * period && rv_in * rv_out > 0.0)
            };
            if (rv_in < 0.0 && rv_out > 0.0) || long_enough {
                return Some(LostReason::PerihelionBound);
            }
        }
    }
    None
}

/// Keep only the entries of `values` where `flags` is [`None`].
fn retain_by_flags<U, F>(values: &mut Vec<U>, flags: &[Option<F>]) {
    let mut idx = 0;
    values.retain(|_| {
        let keep = flags[idx].is_none();
        idx += 1;
        keep
    });
}
