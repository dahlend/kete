//! Python bindings for the Wisdom-Holman symplectic N-body simulation.
use kete_core::constants::GMS;
use kete_core::errors::Error;
use kete_core::forces::{FrozenNonGrav, GravParams};
use kete_core::frames::{Equatorial, SSB};
use kete_core::integrators::{LostReason, WisdomHolman};
use kete_core::state::State;
use kete_spice::prelude::LOADED_SPK;
use pyo3::prelude::*;

use crate::nongrav::PyNonGravModel;
use crate::state::PyState;
use crate::time::PyTime;

/// Convert optional per-particle `NonGravModel`s into the frozen forces the
/// integrator accepts. Validation (value counts, ranges, supported kinds,
/// list length) happens in the integrator's constructor.
fn freeze_non_gravs(non_gravs: Option<Vec<Option<PyNonGravModel>>>) -> Vec<Option<FrozenNonGrav>> {
    non_gravs
        .unwrap_or_default()
        .into_iter()
        .map(|model| model.map(|model| model.to_frozen()))
        .collect()
}

/// A long-term symplectic N-body simulation of the solar system.
///
/// This is a fixed-step Wisdom-Holman integrator (democratic heliocentric
/// splitting) intended for million-to-billion year orbital evolution of the
/// planets, massive asteroids, and massless test particles. Unlike
/// :func:`~kete.propagate_n_body`, which follows the SPICE kernels and is
/// limited to their time span, this evolves the entire system
/// self-consistently and has no time limit.
///
/// Because the step size is fixed, it should be at most 1/20th of the
/// shortest orbital period in the simulation (about 4 days when Mercury is
/// included). Close encounters are not resolved by this method; the
/// simulation records the closest approach seen so results can be audited
/// (see :attr:`closest_encounter`).
///
/// Test particles which fall into the Sun are removed and recorded in
/// :attr:`lost_particles`.
///
/// Test particles may optionally carry a :class:`~kete.NonGravModel`, the
/// same per-object non-gravitational description accepted by
/// :func:`~kete.propagate_n_body`. A Farnocchia model gives the particle
/// radiation pressure and thermal recoil (Yarkovsky), whose along-track
/// component drifts the semi-major axis at a rate set by the angle between
/// the spin pole and the orbit normal; over millions of years this is what
/// spreads a collisional family, so a population should be given randomly
/// oriented poles (see :meth:`~kete.NonGravModel.new_farnocchia_from_h_mag`).
/// A dust model makes the particle a dust grain: it moves on the
/// radiation-reduced two-body orbit ``(1 - beta) * GM_sun`` and is damped by
/// Poynting-Robertson drag, so its orbit decays secularly. A JPL-style
/// ``A1/A2/A3`` model applies the fitted radial/transverse/normal
/// accelerations with their ``g(r)`` scaling -- the form JPL orbit solutions
/// use both for comet outgassing and for asteroid Yarkovsky detections via
/// ``A2``, so covariance samples from
/// :meth:`~kete.HorizonsProperties.sample` feed directly into ``non_gravs``.
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
/// ephemeris-accurate short-term work use :func:`~kete.propagate_n_body`.
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
///     Schwarzschild acceleration shared with :func:`~kete.propagate_n_body`,
///     which reproduces both the secular apsidal precession (43 arcsec/century
///     for Mercury) and the relativistic mean motion. The term is velocity
///     dependent, so with it enabled the integrator is only approximately
///     symplectic; the energy error remains a bounded band with no secular
///     trend.
/// include_j2:
///     Apply the solar J2 oblateness term, with the same coefficient and
///     ecliptic-pole approximation used by :func:`~kete.propagate_n_body`.
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
///     Optional list of :class:`~kete.NonGravModel`, one per test particle,
///     with ``None`` entries for particles which feel gravity alone. All
///     model parameters must be concrete values (a ``NaN`` left free for
///     orbit fitting cannot be simulated).
#[pyclass(module = "kete", name = "SymplecticSim")]
#[derive(Debug)]
pub struct PySymplecticSim {
    sim: WisdomHolman<Equatorial>,
}

#[pymethods]
impl PySymplecticSim {
    /// Construct the simulation from explicit states and masses.
    #[new]
    #[pyo3(signature = (massive_states, masses, test_particles=vec![], dt=4.0, include_gr=true,
        include_j2=true, use_correctors=true, non_gravs=None))]
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
        Ok(Self {
            sim: WisdomHolman::new(
                &massive,
                &gms,
                &particles,
                &freeze_non_gravs(non_gravs),
                dt,
                include_gr,
                include_j2,
                use_correctors,
            )?,
        })
    }

    /// Build a simulation from the loaded SPICE kernels.
    ///
    /// The Sun and the 8 planets (Earth and Moon merged into their
    /// barycenter) are always included. The currently registered massive
    /// bodies (by default Ceres, Pallas, Vesta, Hygiea, and Interamnia; see
    /// :func:`~kete.register_mass` and :func:`~kete.register_custom_mass` to
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
    ///     Optional list of :class:`~kete.NonGravModel`, one per test
    ///     particle, with ``None`` entries for particles which feel gravity
    ///     alone; see the class documentation.
    #[staticmethod]
    #[pyo3(signature = (jd, dt=4.0, include_registered=true, test_particles=vec![], include_gr=true,
        include_j2=true, use_correctors=true, non_gravs=None))]
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
    ) -> PyResult<Self> {
        let epoch = jd.0;
        let mut params: Vec<GravParams> = GravParams::simplified_planets().clone();
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
                .copied()
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
        Ok(Self {
            sim: WisdomHolman::new(
                &massive,
                &gms,
                &particles,
                &freeze_non_gravs(non_gravs),
                dt,
                include_gr,
                include_j2,
                use_correctors,
            )?,
        })
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
    /// The GIL is released for the duration of the integration.
    pub fn integrate_n_steps(&mut self, py: Python<'_>, n: u64) -> PyResult<()> {
        py.detach(|| self.sim.integrate_n_steps(n))
            .map_err(Into::into)
    }

    /// Advance the simulation to approximately the target time.
    ///
    /// The step size is fixed, so the integration lands on the whole step
    /// closest to the requested time and no partial step is taken; check
    /// :attr:`jd` for the resulting epoch. The GIL is released for the
    /// duration of the integration.
    ///
    /// Parameters
    /// ----------
    /// jd:
    ///     Target time. Must not be behind the current epoch with respect to
    ///     the sign of ``dt``.
    pub fn integrate_to(&mut self, py: Python<'_>, jd: PyTime) -> PyResult<()> {
        py.detach(|| self.sim.integrate_to(jd.0))
            .map_err(Into::into)
    }

    /// Current epoch of the simulation as a TDB scaled JD.
    #[getter]
    pub fn jd(&self) -> f64 {
        self.sim.epoch().jd
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

    /// Number of currently active test particles.
    #[getter]
    pub fn n_test_particles(&self) -> usize {
        self.sim.n_test_particles()
    }

    /// Test particles removed from the simulation, as a list of
    /// ``(designation, jd, reason)`` tuples. ``reason`` is either
    /// ``"sun_impact"`` or ``"kepler_failure"``.
    #[getter]
    pub fn lost_particles(&self) -> Vec<(String, f64, String)> {
        self.sim
            .lost_particles()
            .iter()
            .map(|lost| {
                let reason = match lost.reason {
                    LostReason::SunImpact => "sun_impact",
                    LostReason::KeplerFailure => "kepler_failure",
                };
                (
                    lost.desig.clone().try_naif_id_to_name().to_string(),
                    lost.epoch.jd,
                    reason.into(),
                )
            })
            .collect()
    }

    /// The closest approach between two bodies seen so far, as a
    /// ``(designation, designation, hill_ratio, jd)`` tuple, or ``None``.
    ///
    /// Only approaches within 3 Hill radii of a massive body are recorded.
    /// If this is not ``None``, the trajectories of the involved bodies
    /// should be treated with suspicion, since the fixed-step map does not
    /// resolve close encounters.
    #[getter]
    pub fn closest_encounter(&self) -> Option<(String, String, f64, f64)> {
        self.sim.closest_encounter().map(|enc| {
            (
                enc.first.clone().try_naif_id_to_name().to_string(),
                enc.second.clone().try_naif_id_to_name().to_string(),
                enc.hill_ratio,
                enc.epoch.jd,
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
            self.sim.epoch().jd,
            self.sim.dt(),
            self.sim.n_massive(),
            self.sim.n_test_particles(),
            self.sim.include_gr(),
            self.sim.include_j2(),
            self.sim.use_correctors(),
        )
    }
}
