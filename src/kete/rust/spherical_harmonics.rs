//! Python bindings for spherical harmonic gravity fields.
use std::sync::Arc;

use kete_core::constants::{AU_KM, GMS};
use kete_core::forces::{GravParams, Shape, SphericalHarmonics};
use kete_core::prelude::Error;
use pyo3::{PyResult, pyclass, pyfunction, pymethods};

use crate::polyhedron::{PyPolyhedron, orientation};

/// Gravity field given by fully normalized spherical harmonic coefficients.
///
/// ``U = GM / R sum_n sum_m (C_nm V_nm + S_nm W_nm)`` with
/// ``V_nm = (R / r)^(n+1) P_nm(sin lat) cos(m lon)`` (``W_nm`` with ``sin``), ``P_nm``
/// the associated Legendre functions without the Condon-Shortley phase, and the
/// coefficients fully normalized (the geodesy convention): an unnormalized J2 is
/// ``C_20 = -J2 / sqrt(5)``. Positions are relative to the expansion origin, in the
/// body frame, in the units of ``radius``; ``gm`` sets the units of the results.
///
/// The series converges only outside the smallest sphere about the origin that
/// contains all of the body's mass (its Brillouin sphere). Given ``min_radius`` (the
/// Brillouin radius or larger), an evaluation inside it raises; without it, only
/// the origin does. The reference radius ``radius`` is a normalization choice and
/// is not that limit.
///
/// Parameters
/// ----------
/// gm :
///     Gravitational parameter.
/// radius :
///     Reference radius of the coefficients.
/// c :
///     ``C_nm``, one row per degree ``n`` with ``n + 1`` entries.
/// s :
///     ``S_nm``, the same shape; ``S_n0`` must be 0.
/// min_radius :
///     Distance from the origin inside which an evaluation raises.
#[pyclass(
    module = "kete._core",
    name = "SphericalHarmonics",
    frozen,
    from_py_object
)]
#[derive(Clone, Debug)]
pub struct PySphericalHarmonics(pub Arc<SphericalHarmonics>);

#[pymethods]
impl PySphericalHarmonics {
    /// Build a field from normalized coefficients.
    #[new]
    #[pyo3(signature = (gm, radius, c, s, min_radius=None))]
    pub fn new(
        gm: f64,
        radius: f64,
        c: Vec<Vec<f64>>,
        s: Vec<Vec<f64>>,
        min_radius: Option<f64>,
    ) -> PyResult<Self> {
        Ok(Self(Arc::new(SphericalHarmonics::new(
            gm, radius, &c, &s, min_radius,
        )?)))
    }

    /// The exact field of a constant-density :class:`Polyhedron` to ``degree``,
    /// expanded about the mesh origin, with the reference and minimum radius the
    /// mesh's bounding radius; units are those of the polyhedron.
    #[staticmethod]
    pub fn from_polyhedron(poly: PyPolyhedron, degree: usize) -> PyResult<Self> {
        Ok(Self(Arc::new(SphericalHarmonics::from_polyhedron(
            &poly.0, degree,
        )?)))
    }

    /// The same field cut at ``degree``.
    pub fn truncated(&self, degree: usize) -> PyResult<Self> {
        Ok(Self(Arc::new(self.0.truncated(degree)?)))
    }

    /// Gravitational potential at a position, positive, so that the acceleration
    /// is its gradient.
    pub fn potential(&self, pos: [f64; 3]) -> PyResult<f64> {
        Ok(self.0.potential(&pos.into())?)
    }

    /// Acceleration at a position.
    pub fn field(&self, pos: [f64; 3]) -> PyResult<[f64; 3]> {
        Ok(self.0.field(&pos.into())?.into())
    }

    /// Acceleration and its gradient with respect to position (``grad[i][j]`` is
    /// ``d accel_i / d pos_j``) at a position.
    pub fn field_and_gradient(&self, pos: [f64; 3]) -> PyResult<([f64; 3], [[f64; 3]; 3])> {
        let (accel, grad) = self.0.field_and_gradient(&pos.into())?;
        let rows = [0, 1, 2].map(|i| [grad[(i, 0)], grad[(i, 1)], grad[(i, 2)]]);
        Ok((accel.into(), rows))
    }

    /// Gravitational parameter.
    #[getter]
    pub fn gm(&self) -> f64 {
        self.0.gm()
    }

    /// Reference radius.
    #[getter]
    pub fn radius(&self) -> f64 {
        self.0.radius()
    }

    /// Distance from the origin inside which an evaluation raises, or None.
    #[getter]
    pub fn min_radius(&self) -> Option<f64> {
        self.0.min_radius()
    }

    /// Largest degree.
    #[getter]
    pub fn degree(&self) -> usize {
        self.0.degree()
    }

    /// ``C_nm``, one row per degree.
    #[getter]
    pub fn c(&self) -> Vec<Vec<f64>> {
        self.rows(|(c, _)| c)
    }

    /// ``S_nm``, one row per degree.
    #[getter]
    pub fn s(&self) -> Vec<Vec<f64>> {
        self.rows(|(_, s)| s)
    }

    /// String representation.
    pub fn __repr__(&self) -> String {
        format!(
            "SphericalHarmonics(degree={}, gm={}, radius={}, min_radius={:?})",
            self.0.degree(),
            self.0.gm(),
            self.0.radius(),
            self.0.min_radius()
        )
    }
}

impl PySphericalHarmonics {
    fn rows(&self, pick: impl Fn((f64, f64)) -> f64) -> Vec<Vec<f64>> {
        (0..=self.0.degree())
            .map(|n| {
                (0..=n)
                    .filter_map(|m| self.0.coefficient(n, m).map(&pick))
                    .collect()
            })
            .collect()
    }
}

/// Register a massive body whose gravity near it is a spherical harmonic field.
///
/// Registered bodies are part of the force model when ``include_asteroids=True`` is
/// passed to :func:`propagate_n_body` or orbit fitting; the default force model is a
/// fixed list of planets and the Moon, which a registration does not change. Within
/// ``switch_radius`` of the body those functions use the series in place of the
/// point mass; beyond it, the point mass. The body
/// must have an SPK ephemeris loaded; the expansion origin is the body's SPK
/// position. Inside ``min_radius`` (the body's Brillouin radius, or larger) the
/// series is not valid, and a propagation that reaches it raises as an impact with
/// the body. A body already registered with the same NAIF ID is replaced.
///
/// The coefficients are fully normalized, without the Condon-Shortley phase (see
/// :class:`~kete.shape.SphericalHarmonics`); check the convention of a published
/// field before using it. ``C_00`` must be 1, since the series' mass is the body's,
/// which is also the point mass beyond ``switch_radius``.
///
/// Exactly one orientation is required: ``frame_id``, a body frame read at each
/// evaluation inside ``switch_radius`` from the loaded PCK files or, when they have
/// no frame with that id, the CK files with their clock, where a time without
/// orientation is an error, or ``rotation``, a fixed matrix taking body-frame
/// vectors to equatorial (J2000) axes.
///
/// Parameters
/// ----------
/// naif_id :
///     NAIF ID of the body.
/// c :
///     Normalized ``C_nm``, one row per degree ``n`` with ``n + 1`` entries.
/// s :
///     Normalized ``S_nm``, the same shape; ``S_n0`` must be 0.
/// radius :
///     Reference radius of the coefficients, in km.
/// min_radius :
///     Distance from the body in km inside which the series is not valid.
/// switch_radius :
///     Distance from the body in km inside which the series is used; at least
///     ``min_radius``.
/// mass :
///     Mass of the body as a fraction of the Sun's mass. Defaults to the value in
///     the built-in mass table.
/// frame_id :
///     SPICE frame ID or frame name of the body frame.
/// rotation :
///     Fixed rotation matrix from the body frame to equatorial axes, shape
///     ``(3, 3)``.
///
/// Returns
/// -------
/// SphericalHarmonics
///     The registered field, in AU and AU^3/day^2.
#[pyfunction]
#[allow(
    clippy::too_many_arguments,
    reason = "Python keyword arguments, the last three with defaults"
)]
#[pyo3(signature = (naif_id, c, s, radius, min_radius, switch_radius, mass=None, frame_id=None, rotation=None))]
pub fn register_spherical_harmonics(
    naif_id: i32,
    c: Vec<Vec<f64>>,
    s: Vec<Vec<f64>>,
    radius: f64,
    min_radius: f64,
    switch_radius: f64,
    mass: Option<f64>,
    frame_id: Option<crate::spice::FrameLike>,
    rotation: Option<[[f64; 3]; 3]>,
) -> PyResult<PySphericalHarmonics> {
    let orientation = orientation(frame_id, rotation)?;
    if !switch_radius.is_finite() || switch_radius < min_radius {
        Err(Error::ValueError(format!(
            "switch_radius must be finite and at least min_radius, found {switch_radius} km \
             and {min_radius} km."
        )))?;
    }
    if c.first()
        .and_then(|row| row.first())
        .is_none_or(|c00| (c00 - 1.0).abs() > 1e-12)
    {
        Err(Error::ValueError(
            "C_00 must be 1: the series' mass is the body's mass.".into(),
        ))?;
    }
    let gm = match mass {
        Some(m) => m * GMS,
        None => GravParams::try_mass_from_naif_id(naif_id)?,
    };
    let model = Arc::new(SphericalHarmonics::new(
        gm,
        radius / AU_KM,
        &c,
        &s,
        Some(min_radius / AU_KM),
    )?);
    let mut params = GravParams::new(naif_id, gm, (min_radius / AU_KM) as f32);
    params.shape = Shape::SphericalHarmonics {
        model: Arc::clone(&model),
        orientation,
        switch_radius: switch_radius / AU_KM,
    };
    params.register();
    Ok(PySphericalHarmonics(model))
}
