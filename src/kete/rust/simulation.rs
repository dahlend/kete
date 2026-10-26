// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Python bindings for the Wisdom-Holman symplectic N-body simulation.
use kete_core::constants::GMS;
use kete_core::errors::Error;
use kete_core::forces::{GravParams, NonGravMask};
use kete_core::frames::{Equatorial, SSB};
use kete_core::integrators::{LostReason, WisdomHolman};
use kete_core::state::State;
use kete_spice::prelude::LOADED_SPK;
use pyo3::prelude::*;

use crate::frame::PyFrames;
use crate::nongrav::PyNonGravModel;
use crate::state::PyState;
use crate::time::PyTime;
use crate::vector::{PyVector, VectorLike};
use nalgebra::Vector3;

/// Convert optional per-particle `NonGravModel`s into the fully fixed forces the
/// integrator accepts. NaN values become 0, as in every other propagation.
/// Validation (value counts, ranges, supported kinds, list length) happens in the
/// integrator's constructor.
fn freeze_non_gravs(non_gravs: Option<Vec<Option<PyNonGravModel>>>) -> Vec<Option<NonGravMask>> {
    non_gravs
        .unwrap_or_default()
        .into_iter()
        .map(|model| model.map(|model| model.to_fixed()))
        .collect()
}

/// A long-term symplectic N-body simulation of the solar system.
///
/// This is a fixed-step Wisdom-Holman integrator (democratic heliocentric
/// splitting) intended for million-to-billion year orbital evolution of the
/// planets, massive asteroids, and massless test particles. Unlike
/// :func:`~kete.propagation.propagate_n_body`, which follows the SPICE kernels and is
/// limited to their time span, this evolves the entire system
/// self-consistently and has no time limit.
///
/// Because the step size is fixed, it should be at most 1/20th of the
/// shortest orbital period in the simulation (about 4 days when Mercury is
/// included). Close encounters of test particles with massive bodies are
/// resolved when ``close_encounters`` is given; encounters between massive
/// bodies never are. The simulation records the closest approach seen so
/// results can be audited (see :attr:`closest_encounter`).
///
/// Test particles which fall into the Sun are removed and recorded in
/// :attr:`lost_particles`.
///
/// Test particles may optionally carry a :class:`~kete.propagation.NonGravModel`, the
/// same per-object non-gravitational description accepted by
/// :func:`~kete.propagation.propagate_n_body`. A Farnocchia model gives the particle
/// radiation pressure and thermal recoil (Yarkovsky), whose along-track
/// component drifts the semi-major axis at a rate set by the angle between
/// the spin pole and the orbit normal; over millions of years this is what
/// spreads a collisional family, so a population should be given randomly
/// oriented poles (see :meth:`~kete.propagation.NonGravModel.new_farnocchia_from_h_mag`).
/// A dust model makes the particle a dust grain: it moves on the
/// radiation-reduced two-body orbit ``(1 - beta) * GM_sun`` and is damped by
/// Poynting-Robertson drag, so its orbit decays secularly. A JPL-style
/// ``A1/A2/A3`` model applies the fitted radial/transverse/normal
/// accelerations with their ``g(r)`` scaling -- the form JPL orbit solutions
/// use both for comet outgassing and for asteroid Yarkovsky detections via
/// ``A2``, so covariance samples from
/// :meth:`~kete.horizons.HorizonsProperties.sample` feed directly into ``non_gravs``.
/// The fitted constants are extrapolated unchanged over the whole
/// integration, and the time-lagged (``dt != 0``) outgassing variant is
/// rejected.
///
/// Total energy of the massive system oscillates within a bounded band rather
/// than trending, which is the usual quality diagnostic for a run; see
/// :attr:`energy`.
///
/// This is a reduced dynamical model built for long-term statistical
/// dynamics, not short-term ephemeris prediction: point-mass Newtonian
/// gravity plus the optional GR and J2 terms, a limited asteroid list, the
/// Earth-Moon pair merged into a barycenter carrying its orbit-averaged
/// lunar quadrupole, and no Pluto unless registered. Measured against JPL
/// DE441 over 15 kyr backwards, planet positions drift away roughly
/// linearly in time, dominated by the remaining model truncations rather
/// than by integration error: every planet stays within a few 1e-4 AU over
/// the full 15 kyr except Neptune, which reaches ~1e-2 AU unless the Pluto
/// system is included (``kete.register_mass(9)`` before :meth:`from_spice`
/// reduces Neptune's drift by roughly an order of magnitude). For
/// ephemeris-accurate short-term work use :func:`~kete.propagation.propagate_n_body`.
///
/// Parameters
/// ----------
/// massive_states:
///     Barycentric states of all massive bodies. The first entry must be the
///     Sun. For long integrations use the Earth-Moon barycenter (NAIF id 3)
///     rather than the separate Earth and Moon.
/// masses:
///     Mass of each massive body as a fraction of the Sun's mass, in the same
///     order as ``massive_states`` (the Sun itself is ``1.0``).
/// test_particles:
///     States of massless test particles. They feel every massive body but
///     affect nothing.
/// dt:
///     Fixed step size in days. Negative integrates backwards in time.
/// include_gr:
///     Apply the general relativity correction of the Sun: the first-order
///     Schwarzschild acceleration shared with :func:`~kete.propagation.propagate_n_body`,
///     which reproduces both the secular apsidal precession (43 arcsec/century
///     for Mercury) and the relativistic mean motion. The term is velocity
///     dependent, so with it enabled the integrator is only approximately
///     symplectic; the energy error remains a bounded band with no secular
///     trend.
/// include_j2:
///     Apply the solar J2 oblateness term, with the same coefficient and
///     solar spin pole (the IAU pole, as in DE440) used by
///     :func:`~kete.propagation.propagate_n_body`.
///     This is a position-only potential, so the integrator remains
///     symplectic; it drives the small secular nodal regression and apsidal
///     precession of low semi-major axis orbits.
/// use_correctors:
///     Wrap each integration call in the order-17 symplectic corrector of
///     Wisdom (2006), which removes the dominant oscillating error of the
///     map (roughly a factor of 100 in the energy band). The overhead is a
///     fixed ~30 steps worth of work per :meth:`integrate_n_steps` or
///     :meth:`integrate_to` call, so batch many steps per call rather than
///     looping over :meth:`step`.
/// non_gravs:
///     Optional list of :class:`~kete.propagation.NonGravModel`, one per test particle,
///     with ``None`` entries for particles which feel gravity alone. A
///     ``NaN`` parameter, left free for orbit fitting, is treated as 0, as in
///     every other propagation.
/// close_encounters:
///     Optional encounter radius in Hill radii (3 is typical). A test particle
///     that passes inside this radius of a massive body during a step has that
///     whole step integrated with the adaptive Radau integrator instead of the
///     map, with the same forces and the massive bodies interpolated between
///     the map's states at the two ends of the step; other steps are
///     unchanged. A particle that reaches the radius of a body is removed as
///     ``"planet_impact"``: its registered radius (see
///     :func:`~kete.register_custom_mass`), else its radius in the mass table,
///     the Earth's for the Earth-Moon barycenter. A body with neither cannot
///     be hit. The resolved steps are not symplectic. With ``use_correctors``, a
///     particle in an encounter at the start or end of an integration call is
///     left out of the corrector and moved with the body it is near.
///     :attr:`close_encounter_steps` counts the steps resolved.
#[pyclass(module = "kete", name = "SymplecticSim")]
#[derive(Debug)]
pub struct PySymplecticSim {
    sim: WisdomHolman<Equatorial>,
}

impl PySymplecticSim {
    /// Integrate `n` steps with the GIL released, in chunks of `CHUNK_STEPS` with a
    /// check for a keyboard interrupt before each.
    fn integrate_interruptible(&mut self, py: Python<'_>, n: u64) -> PyResult<()> {
        /// Steps between checks for a pending signal.
        const CHUNK_STEPS: u64 = 10_000;
        let mut left = n;
        while left > 0 {
            py.check_signals()?;
            let chunk = left.min(CHUNK_STEPS);
            py.detach(|| self.sim.integrate_n_steps(chunk))?;
            left -= chunk;
        }
        Ok(())
    }
}

#[pymethods]
impl PySymplecticSim {
    /// Construct the simulation from explicit states and masses.
    #[new]
    #[pyo3(signature = (massive_states, masses, test_particles=vec![], dt=4.0, include_gr=true,
        include_j2=true, use_correctors=true, non_gravs=None, close_encounters=None))]
    #[allow(
        clippy::too_many_arguments,
        reason = "keyword arguments with defaults on a Python constructor"
    )]
    pub fn new(
        massive_states: Vec<PyState>,
        masses: Vec<f64>,
        test_particles: Vec<PyState>,
        dt: f64,
        include_gr: bool,
        include_j2: bool,
        use_correctors: bool,
        non_gravs: Option<Vec<Option<PyNonGravModel>>>,
        close_encounters: Option<f64>,
    ) -> PyResult<Self> {
        if massive_states.len() != masses.len() {
            Err(Error::ValueError(
                "massive_states and masses must have the same length.".into(),
            ))?;
        }
        let gms: Vec<f64> = masses.into_iter().map(|mass| mass * GMS).collect();
        let spk = LOADED_SPK.try_read().map_err(Error::from)?;
        let massive: Vec<State<Equatorial, SSB>> = massive_states
            .into_iter()
            .map(|state| spk.try_to_ssb(state.raw))
            .collect::<Result<_, Error>>()?;
        let particles: Vec<State<Equatorial, SSB>> = test_particles
            .into_iter()
            .map(|state| spk.try_to_ssb(state.raw))
            .collect::<Result<_, Error>>()?;
        drop(spk);
        let mut sim = WisdomHolman::new(
            &massive,
            &gms,
            &particles,
            &freeze_non_gravs(non_gravs),
            dt,
            include_gr,
            include_j2,
            use_correctors,
        )?;
        if let Some(radius) = close_encounters {
            sim = sim.with_close_encounters(radius)?;
        }
        Ok(Self { sim })
    }

    /// Build a simulation from the loaded SPICE kernels.
    ///
    /// The Sun and the 8 planets (Earth and Moon merged into their
    /// barycenter) are always included. The currently registered massive
    /// bodies (by default Ceres, Pallas, Vesta, Hygiea, and Interamnia; see
    /// :func:`~kete.propagation.register_mass` and :func:`~kete.propagation.register_custom_mass` to
    /// add more, for example the Pluto system with
    /// ``kete.register_mass(9)``) may optionally be added as fully massive
    /// bodies.
    ///
    /// Parameters
    /// ----------
    /// jd:
    ///     Epoch at which to pull the initial states from SPICE. Must be
    ///     within the loaded kernels' time span.
    /// dt:
    ///     Fixed step size in days. Negative integrates backwards in time.
    /// include_registered:
    ///     Include the registered massive bodies beyond the planets.
    /// test_particles:
    ///     States of massless test particles.
    /// include_gr:
    ///     Apply the general relativity correction of the Sun.
    /// include_j2:
    ///     Apply the solar J2 oblateness term; see the class documentation.
    /// use_correctors:
    ///     Wrap each integration call in the order-17 symplectic corrector,
    ///     see the class documentation.
    /// non_gravs:
    ///     Optional list of :class:`~kete.propagation.NonGravModel`, one per test
    ///     particle, with ``None`` entries for particles which feel gravity
    ///     alone; see the class documentation.
    /// close_encounters:
    ///     Optional encounter radius in Hill radii inside which test particle
    ///     steps are integrated with Radau; see the class documentation.
    #[staticmethod]
    #[pyo3(signature = (jd, dt=4.0, include_registered=true, test_particles=vec![], include_gr=true,
        include_j2=true, use_correctors=true, non_gravs=None, close_encounters=None))]
    #[allow(
        clippy::too_many_arguments,
        reason = "keyword arguments with defaults on a Python constructor"
    )]
    pub fn from_spice(
        jd: PyTime,
        dt: f64,
        include_registered: bool,
        test_particles: Vec<PyState>,
        include_gr: bool,
        include_j2: bool,
        use_correctors: bool,
        non_gravs: Option<Vec<Option<PyNonGravModel>>>,
        close_encounters: Option<f64>,
    ) -> PyResult<Self> {
        let epoch = jd.0;
        let mut params: Vec<GravParams> = GravParams::simplified_planets().to_vec();
        if include_registered {
            // Everything registered beyond the merged planet list. Skip the
            // ids that list already covers: the barycenters themselves
            // (deduplicated below) and the separate planet-center and Moon
            // entries folded into their barycenters.
            let covered = [199, 299, 301, 399, 499, 599, 699, 799, 899];
            let extra: Vec<GravParams> = GravParams::selected_masses()
                .iter()
                .filter(|p| {
                    !covered.contains(&p.naif_id) && params.iter().all(|q| q.naif_id != p.naif_id)
                })
                .cloned()
                .collect();
            params.extend(extra);
        }

        let spk = LOADED_SPK.try_read().map_err(Error::from)?;
        let mut massive: Vec<State<Equatorial, SSB>> = Vec::with_capacity(params.len());
        let mut gms: Vec<f64> = Vec::with_capacity(params.len());
        for p in &params {
            let state = spk.try_get_state_with_center::<Equatorial>(p.naif_id, epoch, 0)?;
            massive.push(state.try_into()?);
            gms.push(p.mass);
        }
        let particles: Vec<State<Equatorial, SSB>> = test_particles
            .into_iter()
            .map(|state| spk.try_to_ssb(state.raw))
            .collect::<Result<_, Error>>()?;
        drop(spk);
        let mut sim = WisdomHolman::new(
            &massive,
            &gms,
            &particles,
            &freeze_non_gravs(non_gravs),
            dt,
            include_gr,
            include_j2,
            use_correctors,
        )?;
        if let Some(radius) = close_encounters {
            sim = sim.with_close_encounters(radius)?;
        }
        Ok(Self { sim })
    }

    /// Advance the simulation by a single step of ``dt``.
    ///
    /// With correctors enabled every call pays the fixed corrector overhead;
    /// prefer :meth:`integrate_n_steps` or :meth:`integrate_to` for more
    /// than a handful of steps.
    pub fn step(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.sim.step()).map_err(Into::into)
    }

    /// Advance the simulation by the given number of steps.
    ///
    /// The GIL is released for the duration of the integration. A keyboard
    /// interrupt stops it within a few thousand steps, at the end of a step, with
    /// the simulation usable from there.
    pub fn integrate_n_steps(&mut self, py: Python<'_>, n: u64) -> PyResult<()> {
        self.integrate_interruptible(py, n)
    }

    /// Advance the simulation to approximately the target time.
    ///
    /// The step size is fixed, so the integration lands on the whole step
    /// closest to the requested time and no partial step is taken; check
    /// :attr:`jd` for the resulting epoch. The GIL is released for the
    /// duration of the integration. A keyboard interrupt stops it within a few
    /// thousand steps, at the end of a step, with the simulation usable from there.
    ///
    /// Parameters
    /// ----------
    /// jd:
    ///     Target time. Must not be behind the current epoch with respect to
    ///     the sign of ``dt``.
    pub fn integrate_to(&mut self, py: Python<'_>, jd: PyTime) -> PyResult<()> {
        let n = self.sim.steps_to(jd.0)?;
        self.integrate_interruptible(py, n)
    }

    /// Current epoch of the simulation as a TDB scaled JD.
    #[getter]
    pub fn jd(&self) -> f64 {
        self.sim.epoch().jd()
    }

    /// Fixed step size in days.
    #[getter]
    pub fn dt(&self) -> f64 {
        self.sim.dt()
    }

    /// Number of steps taken so far.
    #[getter]
    pub fn steps_taken(&self) -> i64 {
        self.sim.steps_taken()
    }

    /// Whether the GR correction is enabled.
    #[getter]
    pub fn include_gr(&self) -> bool {
        self.sim.include_gr()
    }

    /// Whether the solar J2 oblateness term is enabled.
    #[getter]
    pub fn include_j2(&self) -> bool {
        self.sim.include_j2()
    }

    /// The close encounter radius in Hill radii, or ``None`` when test
    /// particle encounters are not resolved.
    #[getter]
    pub fn close_encounters(&self) -> Option<f64> {
        self.sim.close_encounter_radius()
    }

    /// Number of test particle steps resolved as close encounters so far,
    /// including those ending in a removal.
    #[getter]
    pub fn close_encounter_steps(&self) -> u64 {
        self.sim.close_encounter_steps()
    }

    /// Whether the order-17 symplectic corrector is enabled.
    #[getter]
    pub fn use_correctors(&self) -> bool {
        self.sim.use_correctors()
    }

    /// Current barycentric states of the Sun and all massive bodies, in the
    /// order they were provided (the Sun first).
    #[getter]
    pub fn massive_states(&self) -> Vec<PyState> {
        self.sim
            .massive_states()
            .into_iter()
            .map(|s| State::<Equatorial>::from(s).into())
            .collect()
    }

    /// Current barycentric states of the active test particles, including any
    /// carrying non-gravitational forces.
    #[getter]
    pub fn test_particle_states(&self) -> Vec<PyState> {
        self.sim
            .test_particle_states()
            .into_iter()
            .map(|s| State::<Equatorial>::from(s).into())
            .collect()
    }

    /// Designations of the active test particles, in the order of
    /// :attr:`test_particle_states`.
    #[getter]
    pub fn test_particle_desigs(&self) -> Vec<String> {
        self.sim
            .test_particle_desigs()
            .iter()
            .map(|d| d.clone().try_naif_id_to_name().to_string())
            .collect()
    }

    /// Heliocentric positions (AU) and velocities (AU/day) of the active test
    /// particles, as two lists of ``[x, y, z]`` on the axes of ``frame``, in
    /// the order of :attr:`test_particle_states`.
    ///
    /// This skips building a :class:`~kete.State` per particle, so it is the
    /// fast way to read a large population.
    ///
    /// Parameters
    /// ----------
    /// frame:
    ///     Frame of the returned vectors.
    #[pyo3(signature = (frame=PyFrames::Ecliptic))]
    pub fn test_particle_heliocentric(&self, frame: PyFrames) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
        self.sim
            .test_particle_heliocentric()
            .into_iter()
            .map(|(pos, vel)| {
                (
                    PyVector::new(pos.into(), frame).raw(),
                    PyVector::new(vel.into(), frame).raw(),
                )
            })
            .unzip()
    }

    /// Remove the test particles at the given indices (into the order of
    /// :attr:`test_particle_states`) and return their barycentric states, in
    /// the order the indices were given. The remaining particles keep their
    /// relative order.
    ///
    /// Parameters
    /// ----------
    /// indices:
    ///     Indices of the particles to remove; each at most once.
    pub fn remove_test_particles(&mut self, indices: Vec<usize>) -> PyResult<Vec<PyState>> {
        Ok(self
            .sim
            .remove_test_particles(&indices)?
            .into_iter()
            .map(|s| State::<Equatorial>::from(s).into())
            .collect())
    }

    /// Append test particles at the current epoch.
    ///
    /// Parameters
    /// ----------
    /// states:
    ///     States of the new particles, at the simulation's current epoch.
    /// non_gravs:
    ///     Optional list of :class:`~kete.propagation.NonGravModel`, one per
    ///     state, with ``None`` entries for particles which feel gravity alone.
    #[pyo3(signature = (states, non_gravs=None))]
    pub fn add_test_particles(
        &mut self,
        states: Vec<PyState>,
        non_gravs: Option<Vec<Option<PyNonGravModel>>>,
    ) -> PyResult<()> {
        let spk = LOADED_SPK.try_read().map_err(Error::from)?;
        let particles: Vec<State<Equatorial, SSB>> = states
            .into_iter()
            .map(|state| spk.try_to_ssb(state.raw))
            .collect::<Result<_, Error>>()?;
        drop(spk);
        self.sim
            .add_test_particles(&particles, &freeze_non_gravs(non_gravs))?;
        Ok(())
    }

    /// Add a velocity change (AU/day) to every active test particle: an
    /// instantaneous impulse, such as from a passing star.
    ///
    /// Parameters
    /// ----------
    /// dv:
    ///     One ``[vx, vy, vz]`` per particle, in the order of
    ///     :attr:`test_particle_states`, on the axes of ``frame``.
    /// frame:
    ///     Frame of ``dv``.
    #[pyo3(signature = (dv, frame=PyFrames::Ecliptic))]
    pub fn kick_test_particles(&mut self, dv: Vec<[f64; 3]>, frame: PyFrames) -> PyResult<()> {
        let dv: Vec<Vector3<f64>> = dv
            .into_iter()
            .map(|d| VectorLike::Arr(d).into_vector(frame).into())
            .collect();
        self.sim.kick_test_particles(&dv)?;
        Ok(())
    }

    /// Remove test particles that pass perihelion inside ``min_perihelion`` AU
    /// (an absorbing boundary, such as the planetary region for comets), or
    /// that move beyond ``max_distance`` AU from the Sun. ``None`` disables a
    /// bound. Removed particles appear in :attr:`lost_particles` with reason
    /// ``"perihelion_bound"`` or ``"distance_bound"``.
    ///
    /// Parameters
    /// ----------
    /// min_perihelion:
    ///     Perihelion distance in AU below which a particle passing perihelion
    ///     is removed.
    /// max_distance:
    ///     Heliocentric distance in AU beyond which a particle is removed.
    #[pyo3(signature = (min_perihelion=None, max_distance=None))]
    pub fn set_removal_bounds(
        &mut self,
        min_perihelion: Option<f64>,
        max_distance: Option<f64>,
    ) -> PyResult<()> {
        self.sim.set_removal_bounds(min_perihelion, max_distance)?;
        Ok(())
    }

    /// Number of currently active test particles.
    #[getter]
    pub fn n_test_particles(&self) -> usize {
        self.sim.n_test_particles()
    }

    /// Test particles removed from the simulation, as a list of
    /// ``(designation, jd, reason)`` tuples. ``reason`` is one of
    /// ``"sun_impact"``, ``"kepler_failure"``, ``"perihelion_bound"``,
    /// ``"distance_bound"`` (see :meth:`set_removal_bounds`),
    /// ``"planet_impact"`` and ``"encounter_failure"`` (a resolved close
    /// encounter whose integration failed; see ``close_encounters``).
    #[getter]
    pub fn lost_particles(&self) -> Vec<(String, f64, String)> {
        self.sim
            .lost_particles()
            .iter()
            .map(|lost| {
                let reason = match lost.reason {
                    LostReason::SunImpact => "sun_impact",
                    LostReason::KeplerFailure => "kepler_failure",
                    LostReason::PerihelionBound => "perihelion_bound",
                    LostReason::DistanceBound => "distance_bound",
                    LostReason::PlanetImpact => "planet_impact",
                    LostReason::EncounterFailure => "encounter_failure",
                };
                (
                    lost.desig.clone().try_naif_id_to_name().to_string(),
                    lost.epoch.jd(),
                    reason.into(),
                )
            })
            .collect()
    }

    /// The closest approach between two bodies seen so far, as a
    /// ``(designation, designation, hill_ratio, jd)`` tuple, or ``None``.
    ///
    /// Only unresolved approaches within 3 Hill radii of a massive body are
    /// recorded; test particle steps resolved with ``close_encounters`` are
    /// not. If this is not ``None``, the trajectories of the involved bodies
    /// should be treated with suspicion, since the fixed-step map does not
    /// resolve close encounters.
    #[getter]
    pub fn closest_encounter(&self) -> Option<(String, String, f64, f64)> {
        self.sim.closest_encounter().map(|enc| {
            (
                enc.first.clone().try_naif_id_to_name().to_string(),
                enc.second.clone().try_naif_id_to_name().to_string(),
                enc.hill_ratio,
                enc.epoch.jd(),
            )
        })
    }

    /// The shortest osculating orbital period among the massive bodies, in
    /// days. ``dt`` should be at most 1/20th of this.
    #[getter]
    pub fn shortest_period(&self) -> f64 {
        self.sim.shortest_period()
    }

    /// Total energy of the massive system multiplied by the gravitational
    /// constant, in AU^5 / Day^4.
    ///
    /// For a symplectic map this oscillates within a narrow bounded band;
    /// its relative change is the standard quality diagnostic for a run.
    #[getter]
    pub fn energy(&self) -> f64 {
        self.sim.energy()
    }

    /// Total angular momentum of the massive system multiplied by the
    /// gravitational constant, in AU^5 / Day^3, on equatorial axes.
    #[getter]
    pub fn angular_momentum(&self) -> [f64; 3] {
        self.sim.angular_momentum().into()
    }

    fn __repr__(&self) -> String {
        format!(
            "SymplecticSim(jd={:.2}, dt={}, n_massive={}, n_test_particles={}, \
             include_gr={}, include_j2={}, use_correctors={})",
            self.sim.epoch().jd(),
            self.sim.dt(),
            self.sim.n_massive(),
            self.sim.n_test_particles(),
            self.sim.include_gr(),
            self.sim.include_j2(),
            self.sim.use_correctors(),
        )
    }
}
