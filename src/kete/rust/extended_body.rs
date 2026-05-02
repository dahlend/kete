//! Python bindings for the extended-gravity machinery.
//!
//! Exposes [`Polyhedron`], [`RotationModel`], [`ExtendedBody`] and the
//! body-centric integrator wrapper [`propagate_near_body`] under
//! `kete.extended_body`.  All Python-facing units follow the kete
//! convention: lengths in AU, angles in degrees.  Body-natural units
//! (the internal scaling used by [`kete_core::shape::BodyUnits`]) are
//! handled inside the bindings; the user does not need to think about
//! them.

use kete_core::errors::Error;
use kete_core::forces::GravParams;
use kete_core::frames::Equatorial;
use kete_core::shape::{
    BodyUnits, ExtendedBody, ExtendedGravity, Polyhedron, RotationModel,
    small_body_accel as core_small_body_accel,
};
use kete_core::state::State;
use kete_spice::prelude::{
    BodyRelativeState, LOADED_SPK, Perturber, body_relative_to_heliocentric,
    heliocentric_to_body_relative, is_inside_proximity, propagate_near_body,
};
use nalgebra::Vector3;
use pyo3::exceptions::PyValueError;
use pyo3::{PyResult, pyclass, pyfunction, pymethods};

use crate::desigs::NaifIDLike;
use crate::nongrav::PyNonGravModel;
use crate::state::PyState;
use crate::time::PyTime;

// ---------------------------------------------------------------------
// Polyhedron
// ---------------------------------------------------------------------

/// Werner-Scheeres polyhedron gravity model.
///
/// Vertices are given in AU in the body-fixed frame.  The mesh must be
/// closed (every edge shared by exactly two faces) and have outward-
/// facing CCW-from-outside winding.  Both convex and non-convex
/// polyhedra are supported (the closed-form Werner-Scheeres expressions
/// are exact for any closed orientable mesh).
///
/// Construct with either:
///
/// - :py:meth:`Polyhedron` (the constructor) - pass `gm` directly in
///   solar units (AU^3 / day^2).
/// - :py:meth:`Polyhedron.from_density` - pass a uniform mass density
///   in kg / m^3; `gm` is derived from the volume of the mesh.
///
/// Parameters
/// ----------
/// vertices :
///     Sequence of `(x, y, z)` triples in AU, body-fixed coordinates.
/// faces :
///     Sequence of `(i, j, k)` integer index triples (CCW from outside).
/// gm :
///     Gravitational parameter `G * M` in AU^3 / day^2 (solar units).
#[pyclass(
    module = "kete.extended_body",
    name = "Polyhedron",
    frozen,
    from_py_object
)]
#[derive(Debug, Clone)]
pub struct PyPolyhedron {
    pub(crate) inner: Polyhedron,
}

#[pymethods]
impl PyPolyhedron {
    /// Construct a polyhedron from explicit vertices, face indices and `gm`.
    #[new]
    pub fn new(vertices: Vec<[f64; 3]>, faces: Vec<[u32; 3]>, gm: f64) -> PyResult<Self> {
        let verts: Vec<Vector3<f64>> = vertices.into_iter().map(Vector3::from).collect();
        let inner = Polyhedron::try_new(verts, &faces, gm).map_err(map_err)?;
        Ok(Self { inner })
    }

    /// Construct a polyhedron from a uniform mass density in kg / m^3.
    ///
    /// `gm` is derived as `G * density * volume`, where `volume` is
    /// computed from the mesh in AU^3 and converted to SI internally.
    /// Vertex units remain AU.
    ///
    /// Parameters
    /// ----------
    /// vertices :
    ///     Sequence of `(x, y, z)` triples in AU.
    /// faces :
    ///     Sequence of `(i, j, k)` integer index triples.
    /// density_kg_m3 :
    ///     Uniform mass density in SI (kg / m^3).
    #[staticmethod]
    pub fn from_density(
        vertices: Vec<[f64; 3]>,
        faces: Vec<[u32; 3]>,
        density_kg_m3: f64,
    ) -> PyResult<Self> {
        // G in AU^3 / (kg * day^2) so that
        //   gm[AU^3/day^2] = G * density[kg/m^3] * volume[AU^3]
        // requires converting density to kg/AU^3:
        //   1 AU = 1.495978707e11 m  ->  1 m^3 = (1/1.495978707e11)^3 AU^3
        //   density[kg/AU^3] = density[kg/m^3] * (1.495978707e11)^3
        // Then GM = G_SI[m^3/(kg*s^2)] * density[kg/m^3] * volume[m^3]
        // converted to AU^3/day^2 by:
        //   1 day = 86400 s, 1 m^3 = (AU/m)^-3 AU^3
        //
        // Net: the simplest path is compute `g_sigma = G_SI * density`
        // in SI (m^3/s^2 per m^3 = 1/s^2), convert s^-2 to day^-2 by
        // multiplying by 86400^2, and that *is* the body-natural
        // density-times-G factor expected by Polyhedron::try_new_density.
        // No length conversion is needed because the Werner-Scheeres
        // formulas are dimensionally homogeneous in the vertex length
        // unit.  Volume[AU^3] * g_sigma[1/s^2 * day^2] = AU^3/day^2.
        let g_si = 6.674_30e-11_f64; // m^3 / (kg * s^2)
        let day_seconds = 86400.0_f64;
        let g_sigma = g_si * density_kg_m3 * day_seconds * day_seconds;
        let verts: Vec<Vector3<f64>> = vertices.into_iter().map(Vector3::from).collect();
        let inner = Polyhedron::try_new_density(verts, &faces, g_sigma).map_err(map_err)?;
        Ok(Self { inner })
    }

    /// Vertices in the body-fixed frame, in AU.  Shape `(n, 3)`.
    #[getter]
    pub fn vertices(&self) -> Vec<[f64; 3]> {
        self.inner.vertices.iter().map(|v| (*v).into()).collect()
    }

    /// Total volume of the polyhedron in AU^3.
    #[getter]
    pub fn volume(&self) -> f64 {
        self.inner.volume
    }

    /// Largest distance from the body-fixed origin to any vertex (AU).
    #[getter]
    pub fn bounding_radius(&self) -> f64 {
        self.inner.bounding_radius
    }

    /// Gravitational parameter `G * M` in AU^3 / day^2.
    #[getter]
    pub fn gm(&self) -> f64 {
        self.inner.gm
    }

    /// Number of vertices.
    #[getter]
    pub fn n_vertices(&self) -> usize {
        self.inner.n_vertices()
    }

    /// Number of faces.
    #[getter]
    pub fn n_faces(&self) -> usize {
        self.inner.n_faces()
    }

    /// Number of unique edges.
    #[getter]
    pub fn n_edges(&self) -> usize {
        self.inner.n_edges()
    }

    /// Constant-density center of mass in body-fixed AU coordinates.
    pub fn center_of_mass(&self) -> [f64; 3] {
        self.inner.center_of_mass().into()
    }

    /// Gravitational potential at `point` (body-fixed AU), units
    /// AU^2 / day^2.  Raises :py:class:`ValueError` if `point` lies
    /// exactly on the polyhedron surface.
    pub fn potential(&self, point: [f64; 3]) -> PyResult<f64> {
        self.inner.potential(Vector3::from(point)).map_err(map_err)
    }

    /// Gravitational acceleration at `point` (body-fixed AU), units
    /// AU / day^2.  Raises :py:class:`ValueError` if `point` lies
    /// exactly on the polyhedron surface.
    pub fn acceleration(&self, point: [f64; 3]) -> PyResult<[f64; 3]> {
        self.inner
            .acceleration(Vector3::from(point))
            .map(|v| v.into())
            .map_err(map_err)
    }

    /// True if `point` (body-fixed AU) is inside the polyhedron.
    pub fn contains(&self, point: [f64; 3]) -> bool {
        self.inner.contains(Vector3::from(point))
    }

    /// String representation.
    pub fn __repr__(&self) -> String {
        format!(
            "Polyhedron(n_vertices={}, n_faces={}, volume={:e}, gm={:e})",
            self.inner.n_vertices(),
            self.inner.n_faces(),
            self.inner.volume,
            self.inner.gm
        )
    }
}

// ---------------------------------------------------------------------
// RotationModel
// ---------------------------------------------------------------------

/// Analytic rotation model relating the inertial frame (Equatorial /
/// J2000) to the body-fixed frame.
///
/// Construct with either of the two classmethods:
///
/// - :py:meth:`RotationModel.fixed` - non-rotating body (identity by
///   default).
/// - :py:meth:`RotationModel.constant_spin` - IAU "Cartographic
///   Elements" form: fixed pole and linear prime meridian
///   `W(t) = w0 + w_dot * (t - epoch)`.
///
/// All angle inputs are in degrees; the rate `w_dot` is in degrees /
/// day.  These are converted to radians internally.
#[pyclass(
    module = "kete.extended_body",
    name = "RotationModel",
    frozen,
    from_py_object
)]
#[derive(Debug, Clone, Copy)]
pub struct PyRotationModel {
    pub(crate) inner: RotationModel,
}

#[pymethods]
impl PyRotationModel {
    /// Identity rotation (body-fixed frame coincides with the inertial
    /// frame at all times).
    #[staticmethod]
    pub fn fixed() -> Self {
        Self {
            inner: RotationModel::identity(),
        }
    }

    /// Construct an IAU constant-spin rotation model.
    ///
    /// Parameters
    /// ----------
    /// pole_ra :
    ///     Right ascension of the body's north pole in degrees.
    /// pole_dec :
    ///     Declination of the body's north pole in degrees.
    /// w0 :
    ///     Prime meridian angle at `epoch` in degrees.
    /// w_dot :
    ///     Prime meridian rate in degrees / day.
    /// epoch :
    ///     Reference epoch (TDB) at which the prime meridian equals
    ///     `w0`.  Accepts a JD float or a :py:class:`kete.Time`.
    #[staticmethod]
    pub fn constant_spin(pole_ra: f64, pole_dec: f64, w0: f64, w_dot: f64, epoch: PyTime) -> Self {
        Self {
            inner: RotationModel::ConstantSpin {
                pole_ra: pole_ra.to_radians(),
                pole_dec: pole_dec.to_radians(),
                w0: w0.to_radians(),
                w_dot: w_dot.to_radians(),
                epoch_jd: epoch.jd(),
            },
        }
    }

    /// String representation.
    pub fn __repr__(&self) -> String {
        match self.inner {
            RotationModel::Fixed { .. } => "RotationModel.fixed()".to_string(),
            RotationModel::ConstantSpin {
                pole_ra,
                pole_dec,
                w0,
                w_dot,
                epoch_jd,
            } => format!(
                "RotationModel.constant_spin(pole_ra={:.6} deg, pole_dec={:.6} deg, w0={:.6} deg, w_dot={:.6} deg/day, epoch={:.6})",
                pole_ra.to_degrees(),
                pole_dec.to_degrees(),
                w0.to_degrees(),
                w_dot.to_degrees(),
                epoch_jd,
            ),
        }
    }
}

// ---------------------------------------------------------------------
// ExtendedBody
// ---------------------------------------------------------------------

/// An extended body for proximity-regime gravity work.
///
/// Bundles a list of [`Polyhedron`] components, a [`RotationModel`],
/// and the proximity-regime radius (in AU) at which the body-centric
/// integrator is the appropriate evaluator.  The internal
/// :py:class:`BodyUnits` (length scale, GM scale, time scale) is
/// derived automatically from `length_au` (typically the bounding
/// radius of the largest component) and the aggregate `gm`.
///
/// Parameters
/// ----------
/// components :
///     One or more :py:class:`Polyhedron` components in a shared
///     body-fixed frame.  Total `gm` is the sum of component `gm`s.
/// rotation :
///     A :py:class:`RotationModel` relating the inertial (Equatorial)
///     and body-fixed frames.
/// length_au :
///     Length unit used for body-natural scaling.  Typical value is
///     the bounding radius of the largest component, in AU.
/// proximity_radius_au :
///     Radius (AU) outside of which the caller should switch to the
///     standard heliocentric N-body propagator.  Used by
///     :py:func:`is_inside_proximity` and as a safety check inside
///     :py:func:`propagate_near_body`.
#[pyclass(
    module = "kete.extended_body",
    name = "ExtendedBody",
    frozen,
    from_py_object
)]
#[derive(Debug, Clone)]
pub struct PyExtendedBody {
    pub(crate) inner: ExtendedBody,
}

#[pymethods]
impl PyExtendedBody {
    /// Construct an :py:class:`ExtendedBody` from one or more polyhedron
    /// components, a rotation model, and a proximity-regime radius.
    #[new]
    pub fn new(
        components: Vec<PyPolyhedron>,
        rotation: PyRotationModel,
        length_au: f64,
        proximity_radius_au: f64,
    ) -> PyResult<Self> {
        if components.is_empty() {
            return Err(PyValueError::new_err(
                "ExtendedBody requires at least one component",
            ));
        }
        let gm_total: f64 = components.iter().map(|c| c.inner.gm).sum();
        let units = BodyUnits::try_new(length_au, gm_total).map_err(map_err)?;
        let inner = ExtendedBody::try_new(
            components.into_iter().map(|c| c.inner).collect(),
            rotation.inner,
            units,
            proximity_radius_au,
        )
        .map_err(map_err)?;
        Ok(Self { inner })
    }

    /// Aggregate gravitational parameter in AU^3 / day^2.
    #[getter]
    pub fn gm(&self) -> f64 {
        self.inner.gm
    }

    /// Body-natural length scale in AU.
    #[getter]
    pub fn length_au(&self) -> f64 {
        self.inner.units.length_au
    }

    /// Proximity-regime radius in AU.
    #[getter]
    pub fn proximity_radius_au(&self) -> f64 {
        self.inner.proximity_radius_au
    }

    /// Number of polyhedron components.
    #[getter]
    pub fn n_components(&self) -> usize {
        self.inner.components.len()
    }

    /// True if `point` (body-fixed AU) is inside any component.
    pub fn contains(&self, point: [f64; 3]) -> bool {
        self.inner.contains(Vector3::from(point))
    }

    /// Aggregate body-frame acceleration at `point` (body-fixed AU),
    /// summed over all components and including no tidal terms.
    /// Units: AU / day^2.
    pub fn acceleration(&self, point: [f64; 3]) -> PyResult<[f64; 3]> {
        self.inner
            .body_acceleration(Vector3::from(point))
            .map(|v| v.into())
            .map_err(map_err)
    }

    /// String representation.
    pub fn __repr__(&self) -> String {
        format!(
            "ExtendedBody(n_components={}, gm={:e} AU^3/day^2, length={:e} AU, proximity={:e} AU)",
            self.inner.components.len(),
            self.inner.gm,
            self.inner.units.length_au,
            self.inner.proximity_radius_au,
        )
    }
}

// ---------------------------------------------------------------------
// propagate_near_body
// ---------------------------------------------------------------------

/// Propagate a particle in the body-centric proximity regime of an
/// :py:class:`ExtendedBody` from `state.epoch` to `jd_final`.
///
/// The particle :py:class:`~kete.State` must be expressed with respect
/// to the central body's NAIF id (use :py:meth:`~kete.State.change_center`
/// to convert if necessary).  Internally the state is converted to the
/// body-fixed frame, integrated with body-natural lengths and
/// TDB-day time, and the resulting state is converted back to an
/// inertial Equatorial state centered on the same body NAIF id at
/// `jd_final`.
///
/// Perturbers are specified as a list of `(naif_id, gm)` tuples; their
/// states are queried from the loaded SPK kernels at every integration
/// evaluation.  Pass an empty list to integrate only the body's own
/// extended-gravity field.
///
/// Parameters
/// ----------
/// body :
///     The :py:class:`ExtendedBody` providing the gravity model and
///     rotation.
/// body_naif_id :
///     SPK NAIF id of the central body (used to query the body's own
///     ephemeris and to center the input/output particle state).
/// particle :
///     :py:class:`kete.State` of the particle, centered on
///     `body_naif_id`.
/// jd_final :
///     Final TDB epoch.  Accepts a JD float or :py:class:`kete.Time`.
/// perturbers :
///     Optional list of `(naif_id, gm_solar)` pairs added to the set
///     of registered massive bodies, where `gm_solar` is in
///     AU^3 / day^2.  Use this to include a body that is not in the
///     registered mass list (see :py:func:`kete.spice.register_mass`)
///     or to override its `gm`.  Defaults to no extra perturbers.
/// include_asteroids :
///     If `True`, draw third-body perturbers from
///     :py:meth:`GravParams.selected_masses` (planets plus the
///     extended asteroid set); otherwise from
///     :py:meth:`GravParams.planets` (Sun, planets, Moon).  In both
///     cases the central body itself is filtered out so it is not
///     double-counted with its own polyhedron field.  Mirrors
///     `propagate_n_body`'s argument of the same name.
/// non_grav :
///     Optional :class:`~kete.propagation.NonGravModel` describing
///     non-gravitational forces (JPL comet model, dust SRP/PR, or
///     Farnocchia radiation model).  Evaluated in the inertial
///     Sun-relative frame at every integrator step and transformed
///     back into body-fixed coordinates.
///
/// Returns
/// -------
/// kete.State
///     Particle state at `jd_final`, Equatorial frame, centered on
///     `body_naif_id`.
#[pyfunction]
#[pyo3(name = "propagate_near_body", signature = (body, body_naif_id, particle, jd_final, perturbers=None, include_asteroids=false, non_grav=None))]
pub fn propagate_near_body_py(
    body: &PyExtendedBody,
    body_naif_id: NaifIDLike,
    particle: PyState,
    jd_final: PyTime,
    perturbers: Option<Vec<(NaifIDLike, f64)>>,
    include_asteroids: bool,
    non_grav: Option<PyNonGravModel>,
) -> PyResult<PyState> {
    let (_, body_id): (String, i32) = body_naif_id.try_into()?;

    // Start from the registered mass list (planets, or planets +
    // extended asteroids) just like `propagate_n_body_spk`, dropping
    // the central body itself so it is not double-counted with its
    // own polyhedron field.  Then append any extra caller-supplied
    // perturbers verbatim.
    let mut perturber_list: Vec<Perturber> = if include_asteroids {
        GravParams::selected_masses()
            .iter()
            .filter(|p| p.naif_id != body_id)
            .map(|p| Perturber {
                naif_id: p.naif_id,
                gm_solar: p.mass,
            })
            .collect()
    } else {
        GravParams::planets()
            .into_iter()
            .filter(|p| p.naif_id != body_id)
            .map(|p| Perturber {
                naif_id: p.naif_id,
                gm_solar: p.mass,
            })
            .collect()
    };
    if let Some(extra) = perturbers {
        for (id, gm) in extra {
            let (_, nid): (String, i32) = id.try_into()?;
            perturber_list.push(Perturber {
                naif_id: nid,
                gm_solar: gm,
            });
        }
    }
    let perturbers = perturber_list;

    let mut particle_raw = particle.raw.clone();
    let particle_frame = particle.frame;
    let spk = LOADED_SPK.try_read().map_err(Error::from)?;

    // Re-center the particle on the body if it isn't already.
    if particle_raw.center_id() != body_id {
        spk.try_change_center(&mut particle_raw, body_id)
            .map_err(map_err)?;
    }
    // Body's heliocentric (body-centered, trivially zero) state at the
    // initial and final epochs - except we need a state in some
    // *common* center to drive the conversion helpers.  The natural
    // choice is to use the SSB (id 0) as the conversion center.
    let particle_ssb: State<Equatorial> = {
        let mut s = particle_raw.clone();
        spk.try_change_center(&mut s, 0).map_err(map_err)?;
        s
    };
    let jd_final_tdb: kete_core::time::Time<kete_core::time::TDB> = jd_final.into();
    let body_state_initial: State<Equatorial> = spk
        .try_get_state_with_center::<Equatorial>(body_id, particle_raw.epoch, 0)
        .map_err(map_err)?;
    let body_state_final: State<Equatorial> = spk
        .try_get_state_with_center::<Equatorial>(body_id, jd_final_tdb, 0)
        .map_err(map_err)?;
    drop(spk);

    let body_relative_initial =
        heliocentric_to_body_relative(&particle_ssb, &body.inner, &body_state_initial);

    let spk = LOADED_SPK.try_read().map_err(Error::from)?;
    let non_grav_inner = non_grav.as_ref().map(|n| &n.0);
    let body_relative_final = propagate_near_body(
        &body.inner,
        body_id,
        &perturbers,
        non_grav_inner,
        &body_relative_initial,
        jd_final_tdb,
        &spk,
    )
    .map_err(map_err)?;
    drop(spk);

    let final_ssb =
        body_relative_to_heliocentric(&body_relative_final, &body.inner, &body_state_final);

    // Re-center the output back onto the body NAIF id (so the user
    // gets back a state in the same convention they passed in).
    let mut final_state = final_ssb;
    if final_state.center_id() != body_id {
        let spk = LOADED_SPK.try_read().map_err(Error::from)?;
        spk.try_change_center(&mut final_state, body_id)
            .map_err(map_err)?;
    }

    Ok(PyState {
        raw: final_state,
        frame: particle_frame,
        elements: None,
    })
}

/// True if `particle` lies inside `factor * body.proximity_radius_au`.
///
/// Hysteresis is implemented by the caller picking a tighter `factor`
/// for the regime-entry condition than for the exit condition.
///
/// Parameters
/// ----------
/// particle :
///     :py:class:`kete.State` of the particle (any frame, any center).
/// body :
///     :py:class:`ExtendedBody`.
/// body_naif_id :
///     SPK NAIF id of the central body, used to compute the body-
///     relative position via SPK if `particle` is not already centered
///     on the body.
/// factor :
///     Multiplier on `body.proximity_radius_au`.  Defaults to 1.0.
#[pyfunction]
#[pyo3(name = "is_inside_proximity", signature = (particle, body, body_naif_id, factor=1.0))]
pub fn is_inside_proximity_py(
    particle: PyState,
    body: &PyExtendedBody,
    body_naif_id: NaifIDLike,
    factor: f64,
) -> PyResult<bool> {
    let (_, body_id): (String, i32) = body_naif_id.try_into()?;
    let mut particle_raw = particle.raw;
    if particle_raw.center_id() != body_id {
        let spk = LOADED_SPK.try_read().map_err(Error::from)?;
        spk.try_change_center(&mut particle_raw, body_id)
            .map_err(map_err)?;
    }
    // Build a BodyRelativeState by hand for the predicate (no SPK
    // call needed once we are centered on the body).
    let r_au = Vector3::from(particle_raw.pos);
    let r_body_units = body
        .inner
        .rotation
        .rotate_to_body(particle_raw.epoch.jd, r_au)
        / body.inner.units.length_au;
    let v_au = Vector3::from(particle_raw.vel);
    let v_body_units = body
        .inner
        .rotation
        .rotate_to_body(particle_raw.epoch.jd, v_au)
        / body.inner.units.length_au;
    let brs = BodyRelativeState {
        desig: Some(particle_raw.desig.clone()),
        pos: r_body_units,
        vel: v_body_units,
        epoch: particle_raw.epoch,
    };
    Ok(is_inside_proximity(&brs, &body.inner, factor))
}

/// Helper: gravitational acceleration for the body alone (no tidal
/// terms) at an inertial point, in AU / day^2, returned in the body-
/// fixed frame.  Useful for sanity-checking the gravity model from
/// Python without invoking the integrator.
#[pyfunction]
#[pyo3(name = "body_acceleration", signature = (body, point_body_au, jd_tdb, perturbers=None))]
pub fn body_acceleration_py(
    body: &PyExtendedBody,
    point_body_au: [f64; 3],
    jd_tdb: f64,
    perturbers: Option<Vec<(f64, [f64; 3])>>,
) -> PyResult<[f64; 3]> {
    let perturbers: Vec<(f64, Vector3<f64>)> = perturbers
        .unwrap_or_default()
        .into_iter()
        .map(|(gm, p)| (gm, Vector3::from(p)))
        .collect();
    // small_body_accel returns body_length / day^2, so multiply back
    // by length_au to give AU / day^2 for a Python user.
    let r_body_units = Vector3::from(point_body_au) / body.inner.units.length_au;
    let a_natural =
        core_small_body_accel(&body.inner, jd_tdb, r_body_units, &perturbers).map_err(map_err)?;
    let a_au_per_day2 = a_natural * body.inner.units.length_au;
    Ok(a_au_per_day2.into())
}

// ---------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------

fn map_err<E: Into<Error>>(e: E) -> pyo3::PyErr {
    let e: Error = e.into();
    e.into()
}
