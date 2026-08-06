//! # Wisdom-Holman Symplectic Integrator
//! A fixed-step symplectic N-body map for long term orbital evolution.
//!
//! Unlike the general purpose ODE integrators in this module, this is a specialized
//! N-body map. It advances the whole system at once, and the gravitational dynamics
//! are part of the method itself rather than a user supplied acceleration function.
//
// BSD 3-Clause License
//
// Copyright (c) 2026, Dar Dahlen
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
use crate::analysis::hill_radius;
use crate::constants::{C_AU_PER_DAY, C_AU_PER_DAY_INV_SQUARED, GMS, SUN_J2};
use crate::desigs::Desig;
use crate::errors::{Error, KeteResult};
use crate::forces::{
    FarnocchiaNonGrav, FrozenNonGrav, GravParams, JplCometNonGrav, NonGravKind,
    apply_gr_correction, j2_correction, radiation_accel,
};
use crate::frames::{Ecliptic, InertialFrame, SSB, Vector};
use crate::kepler::analytic_2_body_delta;
use crate::kepler::{compute_peri_dist, compute_semi_major};
use crate::state::State;
use crate::time::{TDB, Time};
use nalgebra::Vector3;
use rayon::prelude::*;
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
/// [`FrozenNonGrav`] inputs of [`WisdomHolman::new`] by [`bind_non_grav`].
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
}

/// Record of a test particle removed during integration.
#[derive(Debug, Clone)]
pub struct LostParticle {
    /// Designation of the removed particle.
    pub desig: Desig,
    /// Epoch of the step during which the particle was removed.
    pub epoch: Time<TDB>,
    /// Why the particle was removed.
    pub reason: LostReason,
}

/// The closest approach between two bodies seen during a run.
///
/// Only approaches within 3 Hill radii are recorded, since the map's accuracy
/// degrades inside that region. If this is reported by a simulation, the
/// trajectories of the involved bodies should be treated with suspicion.
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
/// - There is no close-encounter regularization; the splitting assumes the
///   interaction term is small compared to the Kepler term. The map tracks the
///   closest approach seen, in Hill-radius units, via
///   [`Self::closest_encounter`] so runs can be audited.
/// - Test particles which fall into the Sun, or whose Kepler drift fails to
///   converge, are removed and recorded in [`Self::lost_particles`]. The same
///   situations raise an error for a massive body.
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
///   with the Radau N-body path ([`crate::constants::SUN_J2`], with the solar
///   pole approximated by the ecliptic pole). It is a position-only potential,
///   so the map remains symplectic; it drives the secular nodal regression
///   `-(3/2) n J2 (R/p)^2 cos(i)` and the corresponding apsidal precession.
/// - The optional order-17 symplectic corrector (Wisdom 2006) removes the
///   dominant oscillating error of the map at a fixed cost per integration
///   call.
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
pub struct WisdomHolman<T: InertialFrame> {
    /// Fixed step size in days, sign gives the direction of integration.
    dt: f64,
    /// Epoch of the initial conditions.
    epoch0: Time<TDB>,
    /// Number of steps taken so far.
    steps: i64,
    /// Whether the GR potential correction is applied.
    include_gr: bool,
    /// Whether the solar J2 oblateness term is applied.
    include_j2: bool,
    /// Whether the order-17 symplectic corrector wraps each integration call.
    use_correctors: bool,
    /// The ecliptic pole (the solar spin axis approximation shared with the
    /// Radau N-body path) expressed on the axes of the map's frame.
    solar_pole: Vector3<f64>,
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

impl<T: InertialFrame> WisdomHolman<T> {
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
    /// * `non_gravs` - Optional frozen non-gravitational force per test
    ///   particle: either empty (every particle feels gravity alone) or one
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
    /// * `include_j2` - Apply the solar J2 oblateness term, the same
    ///   [`crate::constants::SUN_J2`] and ecliptic-pole approximation used by
    ///   the Radau N-body path. This is a position-only potential, so the map
    ///   remains symplectic; it drives the small secular nodal regression and
    ///   apsidal precession of low-`a` orbits. When enabled, only the
    ///   component of [`Self::angular_momentum`] along the ecliptic pole is
    ///   conserved (the reaction torque on the solar spin is not modeled).
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
    /// binding the non-gravitational force (wrong frozen-value count, non-finite or
    /// out-of-range values, invalid surface description, or an unsupported
    /// kind), or `dt` is zero or non-finite.
    #[allow(
        clippy::too_many_arguments,
        clippy::fn_params_excessive_bools,
        reason = "parallel per-object lists plus independent physics toggles"
    )]
    pub fn new(
        massive: &[State<T, SSB>],
        gms: &[f64],
        test_particles: &[State<T, SSB>],
        non_gravs: &[Option<FrozenNonGrav>],
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
            if (state.epoch.jd - epoch0.jd).abs() > EPOCH_TOL {
                Err(Error::ValueError(format!(
                    "State {:?} has epoch {} which does not match the Sun's epoch {}.",
                    state.desig, state.epoch.jd, epoch0.jd
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

        // The merged Earth-Moon barycenter is recognized by its NAIF id so it
        // can receive the orbit-averaged lunar quadrupole correction.
        let emb_idx = desigs.iter().position(|d| d.clone().naif_id() == Some(3));

        Ok(Self {
            dt,
            epoch0,
            steps: 0,
            include_gr,
            include_j2,
            use_correctors,
            solar_pole: Vector::<Ecliptic>::new([0.0, 0.0, 1.0])
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
            self.apply_corrector(1.0)?;
        }
        self.kepler(0.5 * dt)?;
        for i in 0..n {
            self.jump(0.5 * dt);
            self.kick(dt);
            self.jump(0.5 * dt);
            if i + 1 < n {
                self.kepler(dt)?;
            } else {
                self.kepler(0.5 * dt)?;
            }
            self.steps += 1;
        }
        if self.use_correctors {
            self.apply_corrector(-1.0)?;
        }
        Ok(())
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
        let n_steps = (time.jd - self.epoch().jd) / self.dt;
        if n_steps < -0.5 {
            Err(Error::ValueError(format!(
                "Target time {} is behind the current epoch {} for dt = {}; construct the \
                 integrator with the opposite sign of dt to integrate in that direction.",
                time.jd,
                self.epoch().jd,
                self.dt
            )))?;
        }
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "clamped non-negative and bounded by f64 integer range"
        )]
        let n_steps = n_steps.round().max(0.0) as u64;
        self.integrate_n_steps(n_steps)
    }

    /// Current epoch, computed as `epoch0 + steps * dt`.
    pub fn epoch(&self) -> Time<TDB> {
        Time::new(self.epoch0.jd + self.elapsed())
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
        self.include_j2
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

    /// The closest approach within 3 Hill radii seen so far, if any occurred.
    ///
    /// The map is not designed to resolve close encounters; if this returns
    /// [`Some`], the subsequent trajectories of the involved bodies (and the
    /// quantitative accuracy of the run) should be treated with suspicion.
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
            if self.include_j2 {
                let u = self.q[i].val.dot(&self.solar_pole) / r;
                energy += 0.5 * GMS * SUN_J2 * sun_r2 * gm * (3.0 * u * u - 1.0) / (r * r * r);
            }
            if self.emb_idx == Some(i) {
                let u = self.q[i].val.dot(&self.solar_pole) / r;
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
    /// separately constant). With the solar J2 term enabled only the
    /// component along the ecliptic pole is conserved; the transverse
    /// components precess, since the reaction torque on the solar spin is
    /// not modeled. With the GR term enabled the magnitude oscillates at the
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
        self.kepler(a)?;
        self.perturbation(-b);
        self.kepler(-2.0 * a)?;
        self.perturbation(b);
        self.kepler(a)
    }

    /// The full perturbation flow (everything except the Kepler part) for
    /// time `b`, as the symmetric split `jump(b/2) kick(b) jump(b/2)`. The
    /// jump/kick composition error is one order higher in the planet masses
    /// than anything the corrector corrects.
    fn perturbation(&mut self, b: f64) {
        self.jump(0.5 * b);
        self.kick(b);
        self.jump(0.5 * b);
    }

    /// Interaction kick: velocity updates from all non-solar pairwise gravity
    /// (plus the optional GR term), positions unchanged. Also tracks
    /// sub-3-Hill-radius approaches.
    fn kick(&mut self, dt: f64) {
        let epoch = self.epoch();
        let n = self.gms.len();

        // Sun velocity frozen at the start of the kick, used to form the
        // heliocentric velocity the dust drag depends on. Captured before the
        // massive velocities are updated below.
        let sun_vel = self.sun_internal_vel();

        for i in 0..n {
            self.hill_r[i] = self.q[i].val.norm() * self.hill_fac[i];
            self.hill_thresh[i] = 9.0 * self.hill_r[i] * self.hill_r[i];
        }
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
        if self.include_j2 {
            let sun_radius = *SUN_RADIUS_AU;
            for i in 0..n {
                self.accel[i] +=
                    j2_correction(&self.q[i].val, &self.solar_pole, sun_radius, SUN_J2, GMS);
            }
        }
        // Orbit-averaged lunar quadrupole on the Earth-Moon barycenter (Quinn,
        // Tremaine & Duncan 1991): the J2 form about the ecliptic pole, with
        // the effective `J2 R^2` carried entirely by the coefficient (radius 1).
        if let Some(i) = self.emb_idx {
            self.accel[i] +=
                j2_correction(&self.q[i].val, &self.solar_pole, 1.0, *EMB_QUAD_J2R2, GMS);
        }
        for i in 0..n {
            self.v[i].add(&(self.accel[i] * dt));
        }

        // Test particles, in parallel.
        let (q, gms, hill_r, hill_thresh) = (&self.q, &self.gms, &self.hill_r, &self.hill_thresh);
        let include_gr = self.include_gr;
        let include_j2 = self.include_j2;
        let solar_pole = self.solar_pole;
        let sun_radius = *SUN_RADIUS_AU;
        let tp_forces = &self.tp_forces;
        let tp_min_chunk = tp_chunk(self.tp_q.len(), 32);
        let tp_worst = self
            .tp_q
            .par_iter()
            .zip(self.tp_v.par_iter_mut())
            .enumerate()
            .with_min_len(tp_min_chunk)
            .map(|(idx, (pos, vel))| {
                let mut accel = Vector3::zeros();
                let mut best = NO_ENCOUNTER;
                for (j, (body_pos, gm)) in q.iter().zip(gms).enumerate() {
                    let sep = body_pos.val - pos.val;
                    let r2 = sep.norm_squared();
                    let inv_r3 = (r2 * r2.sqrt()).recip();
                    accel += sep * (*gm * inv_r3);
                    if r2 < hill_thresh[j] {
                        let ratio = r2.sqrt() / hill_r[j];
                        if ratio < best.0 {
                            best = (ratio, j, idx);
                        }
                    }
                }
                if include_gr {
                    apply_gr_correction(&mut accel, &pos.val, &(vel.val - sun_vel), GMS);
                }
                if include_j2 {
                    accel += j2_correction(&pos.val, &solar_pole, sun_radius, SUN_J2, GMS);
                }
                // `get` covers the gravity-only case, where `tp_forces` is empty.
                match tp_forces.get(idx) {
                    // Radiation pressure and thermal recoil, evaluated directly
                    // on the heliocentric position with the pole already in
                    // this frame.
                    Some(Some(NonGrav::Yarkovsky(yark))) => {
                        accel += radiation_accel(
                            &pos.val,
                            &yark.spin_pole,
                            yark.albedo,
                            yark.absorptivity,
                            yark.flattening,
                            yark.a_over_m,
                            yark.lambda_0,
                        );
                        vel.add(&(accel * dt));
                    }
                    // Dust: radiation pressure is folded into the reduced-mu
                    // drift; the kick applies the exact Poynting-Robertson drag
                    // flow, linear in the heliocentric velocity `vel - sun_vel`.
                    Some(Some(NonGrav::Dust { beta })) => {
                        let k = beta * GMS / (pos.val.norm_squared() * C_AU_PER_DAY);
                        let delta = drag_flow(dt, k, &pos.val, &(vel.val - sun_vel), &accel);
                        vel.add(&delta);
                    }
                    // A1/A2/A3 on the RTN axes of the instantaneous orbit,
                    // evaluated with the pre-kick heliocentric velocity.
                    Some(Some(NonGrav::JplComet { force, a1, a2, a3 })) => {
                        accel += force.accel_no_lag(&pos.val, &(vel.val - sun_vel), *a1, *a2, *a3);
                        vel.add(&(accel * dt));
                    }
                    _ => vel.add(&(accel * dt)),
                }
                best
            })
            .reduce(|| NO_ENCOUNTER, |a, b| if a.0 <= b.0 { a } else { b });

        if worst.0.is_finite() {
            let (ratio, i, j) = worst;
            self.record_encounter(ratio, i, self.desigs[j].clone(), epoch);
        }
        if tp_worst.0.is_finite() {
            let (ratio, j, idx) = tp_worst;
            self.record_encounter(ratio, j, self.tp_desigs[idx].clone(), epoch);
        }
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
        self.tp_q
            .par_iter_mut()
            .with_min_len(chunk)
            .for_each(|pos| pos.add(&shift));
    }

    /// Kepler drift: every body moves on its exact two-body orbit about the
    /// solar GM for `dt`.
    ///
    /// # Errors
    ///
    /// A massive body which is lost during the drift raises an error; test
    /// particles are moved to [`Self::lost_particles`] instead.
    fn kepler(&mut self, dt: f64) -> KeteResult<()> {
        let epoch = self.epoch();

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
                    self.desigs[idx], epoch.jd
                )))?,
                Some(LostReason::KeplerFailure) => Err(Error::Convergence(format!(
                    "Kepler drift failed to converge for massive body {:?} at jd = {}. \
                     The state is likely extreme, such as a deep close encounter; \
                     a smaller dt may help.",
                    self.desigs[idx], epoch.jd
                )))?,
                None => {}
            }
        }

        let chunk = tp_chunk(self.tp_q.len(), 64);
        let tp_forces = &self.tp_forces;
        let tp_lost: Vec<Option<LostReason>> = self
            .tp_q
            .par_iter_mut()
            .zip(self.tp_v.par_iter_mut())
            .enumerate()
            .with_min_len(chunk)
            .map(|(idx, (pos, vel))| {
                // Dust drifts on its radiation-reduced orbit, mu = (1-beta) GMS;
                // `get` covers the gravity-only case, where `tp_forces` is empty.
                let mu = match tp_forces.get(idx) {
                    Some(Some(NonGrav::Dust { beta })) => (1.0 - beta) * GMS,
                    _ => GMS,
                };
                drift(dt, pos, vel, mu)
            })
            .collect();
        if tp_lost.iter().any(Option::is_some) {
            self.remove_lost(&tp_lost, epoch);
        }
        Ok(())
    }

    /// Record and remove the test particles marked in `lost`.
    fn remove_lost(&mut self, lost: &[Option<LostReason>], epoch: Time<TDB>) {
        for (idx, reason) in lost.iter().enumerate() {
            if let Some(reason) = reason {
                self.lost.push(LostParticle {
                    desig: self.tp_desigs[idx].clone(),
                    epoch,
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
/// Returns an error if the frozen value count does not match the kind, any
/// frozen value is non-finite or out of range (a NaN left free for orbit
/// fitting cannot be simulated), the surface or `g(r)` description is
/// invalid, or a [`NonGravKind::JplComet`] carries a nonzero time lag `dt`,
/// which this map does not support.
fn bind_non_grav<T: InertialFrame>(frozen: &FrozenNonGrav, desig: &Desig) -> KeteResult<NonGrav> {
    let values = frozen.values();
    match &frozen.inner {
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

/// Keep only the entries of `values` where `flags` is [`None`].
fn retain_by_flags<U>(values: &mut Vec<U>, flags: &[Option<LostReason>]) {
    let mut idx = 0;
    values.retain(|_| {
        let keep = flags[idx].is_none();
        idx += 1;
        keep
    });
}
