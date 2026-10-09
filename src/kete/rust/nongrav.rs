// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-FileCopyrightText: 2025 California Institute of Technology
// SPDX-License-Identifier: BSD-3-Clause

//! Python wrapper for non-gravitational force models.
//!
//! Exposes the non-grav variants to Python as a single class
//! (`NonGravModel`): dust radiation pressure, JPL comet outgassing, the
//! Farnocchia thermal recoil model. Each variant stores the physical
//! inputs given at construction time (e.g. `beta` for dust, `a1/a2/a3`
//! for comets) and converts them to the underlying Rust force type on
//! demand.
use std::collections::HashMap;

use kete_core::{
    constants::C_V,
    errors::Error,
    forces::{
        DustNonGrav, FarnocchiaNonGrav, JplCometNonGrav, NonGravKind, NonGravMask, ParameterMask,
        ParameterizedForce, a_over_m_from_physical, density_from_a_over_m, lambda_0_from_physical,
        thermal_inertia_from_lambda_0,
    },
};
use kete_flux::diam_from_h_mag_albedo;
use pyo3::{PyResult, exceptions::PyValueError, pyclass, pyfunction, pymethods};

use crate::frame::PyFrames;
use crate::vector::VectorLike;

/// Radiation-pressure coefficient in kg/m^2, the constant in the
/// Burns, Lamy & Soter (1979) form of beta:
///
/// ```text
/// beta = C_PR * q_pr / (density * diameter)
/// ```
///
/// Shared by [`PyNonGravModel::new_dust`] and [`PyNonGravModel::diameter`],
/// which are inverses of one another, so that their defaults cannot drift
/// apart. With the default density of 1000 kg/m^3 and `q_pr = 1`, `beta = 1`
/// falls at a diameter of about 1.2 um.
const C_PR: f64 = 1.19e-3;

/// Non-gravitational force model for n-body propagation.
///
/// The n-body propagation functions accept these models to include forces other
/// than gravity. The constructors are:
///
/// - :py:meth:`NonGravModel.new_dust`: solar radiation pressure and
///   Poynting-Robertson drag on dust.
/// - :py:meth:`NonGravModel.new_comet`: the functional form of the JPL Horizons comet
///   model; see that method for the formula.
/// - :py:meth:`NonGravModel.new_asteroid`: the comet model with a 1/r^2 falloff,
///   for asteroids with the Yarkovsky effect.
/// - :py:meth:`NonGravModel.new_farnocchia`: the Farnocchia et al. radiation
///   model.
///
/// The orbit fitting tools read the fittable parameters of each model (for example
/// A1, A2 and A3 of the comet model) with a NaN convention. A NaN value marks the
/// parameter as free, and the fit starts it from 0. A concrete value holds the
/// parameter fixed at that value. :py:meth:`NonGravModel.with_free` frees parameters
/// while keeping their values as the starting point of the fit. Propagation treats
/// NaN values as 0.
#[pyclass(
    frozen,
    module = "kete.propagation",
    name = "NonGravModel",
    from_py_object
)]
#[derive(Debug, Clone)]
pub struct PyNonGravModel {
    /// The force, with its fixed physical constants.
    force: NonGravKind,
    /// Values of the fittable parameters, in the force's parameter order. NaN marks a
    /// parameter as free for orbit fitting.
    values: Vec<f64>,
    /// Parameters freed at their value by [`with_free`](Self::with_free); empty when
    /// none are.
    freed: Vec<bool>,
}

impl PyNonGravModel {
    /// Wrap a force and its parameter values, with no parameter freed by
    /// [`with_free`](Self::with_free).
    fn from_parts(force: NonGravKind, values: Vec<f64>) -> Self {
        Self {
            force,
            values,
            freed: Vec::new(),
        }
    }

    /// Whether each fittable parameter is free: NaN, or freed at its value.
    fn free_flags(&self) -> Vec<bool> {
        self.values
            .iter()
            .enumerate()
            .map(|(i, v)| v.is_nan() || self.freed.get(i).copied().unwrap_or(false))
            .collect()
    }

    /// Return the starting values for the free parameters: 0 for a NaN, the value
    /// for a parameter freed by [`with_free`](Self::with_free).
    pub fn initial_values(&self) -> Vec<f64> {
        self.values
            .iter()
            .zip(self.free_flags())
            .filter(|(_, free)| *free)
            .map(|(v, _)| if v.is_nan() { 0.0 } else { *v })
            .collect()
    }

    /// Starting values of the free parameters for orbit fitting, when at least one
    /// of them carries a value (a parameter freed by
    /// [`with_free`](Self::with_free)); `None` when every free parameter is NaN.
    pub fn start_values(&self) -> Option<Vec<f64>> {
        let warm = self
            .values
            .iter()
            .zip(self.free_flags())
            .any(|(v, free)| free && !v.is_nan());
        warm.then(|| self.initial_values())
    }

    /// Return a [`NonGravMask`] with every parameter fixed.
    ///
    /// NaN values are mapped to 0.0, freed parameters keep their values. For
    /// propagation only - use
    /// [`to_mask`](Self::to_mask) when setting up orbit fitting.
    pub fn to_fixed(&self) -> NonGravMask {
        let values = self
            .values
            .iter()
            .map(|v| if v.is_nan() { 0.0 } else { *v })
            .collect();
        ParameterMask::all_fixed(self.force.clone(), values).expect("one value per force parameter")
    }

    /// Return a [`NonGravMask`] derived from the NaN sentinel and the freed
    /// parameters.
    ///
    /// NaN or freed -> `None` (free); any other value v -> `Some(v)` (fixed at v).
    pub fn to_mask(&self) -> NonGravMask {
        let mask = self
            .values
            .iter()
            .zip(self.free_flags())
            .map(|(v, free)| if free { None } else { Some(*v) })
            .collect();
        ParameterMask::new(self.force.clone(), mask).expect("one entry per force parameter")
    }

    /// Reconstruct a Python wrapper from a [`NonGravKind`] template and
    /// its concrete parameter values (one per `inner.n_free_params()`). Returns `None`
    /// if the number of values does not match the model.
    pub fn from_force(template: &NonGravKind, values: &[f64]) -> Option<Self> {
        (values.len() == template.n_free_params())
            .then(|| Self::from_parts(template.clone(), values.to_vec()))
    }
}

#[pymethods]
impl PyNonGravModel {
    /// Unused constructor; use the static factory methods.
    #[allow(clippy::new_without_default)]
    #[new]
    pub fn new() -> PyResult<Self> {
        Err(Error::ValueError(
            "Non-gravitational force models need to be constructed using new_dust, new_comet, \
             new_asteroid, or new_farnocchia."
                .into(),
        ))?
    }

    /// Create a new non-gravitational forces Dust model.
    ///
    /// This implements the radiative force model presented in:
    /// "Radiation forces on small particles in the solar system"
    /// Icarus, Vol 40, Issue 1, Pages 1-48, 1979 Oct
    /// https://doi.org/10.1016/0019-1035(79)90050-2
    ///
    ///
    /// The model calculated has the acceleration of the form:
    ///
    /// .. math::
    ///     
    ///     \text{accel} = \frac{L_0 A Q_{pr}}{r^2 c m} \bigg((1 - \frac{\dot{r}}{c}) \vec{S} - \vec{v} / c \bigg)
    ///
    /// Where :math:`L_0` is the luminosity of the Sun, `A` is the effective cross
    /// sectional area of the dust, :math:`Q_{pr}` is a scattering coefficient (~1 for
    /// dust larger than about 0.1 micron), `m` mass, `c` speed of light, and
    /// `r` heliocentric distance.
    ///
    /// The vectors on the right are :math:`\vec{S}` the position with respect to the
    /// Sun. :math:`\vec{v}` the velocity with respect to the Sun. :math:`\dot{r}` is
    /// the radial velocity toward the sun.
    ///
    /// This equation includes both the effects from solar radiation pressure in
    /// addition to the Poynting-Robertson effect. By neglecting the Poynting-Robertson
    /// components of the above formula, it is possible to find a mapping from the
    /// standard :math:`\beta` formalism to the above coefficient:
    ///
    /// .. math::
    ///     
    ///     \beta = \frac{L_0 A Q_{pr}}{c m G}
    ///
    /// Where `G` is the solar standard gravitational parameter (GM).
    /// Making the above equation equivalent to:
    ///
    /// .. math::
    ///     
    ///     \text{accel} = \frac{\beta G}{r^2} \bigg((1 - \frac{\dot{r}}{c}) \vec{S} - \vec{v} / c \bigg)
    ///
    /// :py:meth:`NonGravModel.diameter` is the inverse conversion and shares these
    /// defaults, so ``new_dust(diameter=d).diameter() == d``.
    ///
    /// Parameters
    /// ==========
    /// beta:
    ///     Beta value of the dust, if this is specified, all other inputs are ignored.
    ///     If this value is specified, diameter cannot be specified. Pass
    ///     ``float("nan")`` to leave beta free during orbit fitting.
    /// diameter :
    ///     Diameter of the dust particle in meters, this uses the following parameters to estimate
    ///     the beta value. If beta is specified, this cannot be specified.
    /// density:
    ///     Density in kg/m^3, defaults to 1000 kg/m^3
    /// c_pr:
    ///     Radiation pressure coefficient, defaults to 1.19e-3 kg/m^2
    /// q_pr:
    ///     Scattering efficiency for radiation pressure, defaults to 1.0
    ///     1.0 is a good estimate for particles larger than 1um (Burns, Lamy & Soter 1979)
    #[staticmethod]
    #[pyo3(signature=(beta=None, diameter=None, density=1000.0, c_pr=C_PR, q_pr=1.0))]
    pub fn new_dust(
        beta: Option<f64>,
        diameter: Option<f64>,
        density: f64,
        c_pr: f64,
        q_pr: f64,
    ) -> PyResult<Self> {
        let beta_value = match (beta, diameter) {
            (None, None) => Err(PyValueError::new_err("Must specify beta or diameter."))?,
            (Some(_), Some(_)) => Err(PyValueError::new_err(
                "Cannot specify both beta and diameter.",
            ))?,
            (Some(b), None) => b,
            (None, Some(d)) => (c_pr * q_pr) / (d * density),
        };
        Ok(Self::from_parts(
            NonGravKind::Dust(DustNonGrav),
            vec![beta_value],
        ))
    }

    /// Get the beta value for this dust model.
    #[getter]
    pub fn beta(&self) -> f64 {
        match self.force {
            NonGravKind::Dust(_) => self.values[0],
            _ => f64::NAN,
        }
    }

    /// Estimate the diameter of the dust particle in meters.
    ///
    /// This inverts the beta relation used by
    /// :py:meth:`NonGravModel.new_dust` and takes the same defaults, so a
    /// diameter passed to that constructor is returned unchanged here. Since
    /// only beta is stored, the density, `c_pr` and `q_pr` used to build the
    /// model are not recovered with it and must be supplied again to get back
    /// the same diameter.
    ///
    /// Only works for dust models, returns NaN for asteroid/comet models.
    ///
    /// Parameters
    /// ==========
    /// density:
    ///     Density in kg/m^3, defaults to 1000 kg/m^3
    /// c_pr:
    ///     Radiation pressure coefficient, defaults to 1.19e-3 kg/m^2
    /// q_pr:
    ///     Scattering efficiency for radiation pressure, defaults to 1.0
    ///     1.0 is a good estimate for particles larger than 1um (Burns, Lamy & Soter 1979)
    #[pyo3(signature=(density=1000.0, c_pr=C_PR, q_pr=1.0))]
    pub fn diameter(&self, density: f64, c_pr: f64, q_pr: f64) -> f64 {
        match self.force {
            NonGravKind::Dust(_) => (c_pr * q_pr) / (self.values[0] * density),
            _ => f64::NAN,
        }
    }

    /// JPL's non-gravitational forces are modeled as defined on page 139 of the
    /// Comets II textbook.
    ///
    /// This model adds 3 "A" terms to the acceleration which the object feels. These
    /// A terms represent additional radial, tangential, and normal forces on the
    /// object.
    ///
    /// The defaults of this method are the defaults that JPL Horizons uses for comets
    /// when they are not otherwise specified.
    ///
    /// .. math::
    ///     
    ///     \text{accel} = A_1 g(r) \vec{r} + A_2 g(r) \vec{t} + A_3 g(r) \vec{n}
    ///
    /// Where :math:`\vec{r}`, :math:`\vec{t}`, :math:`\vec{n}` are the radial,
    /// tangential, and normal unit vectors for the object.
    ///
    /// The :math:`g(r)` function is defined by the equation:
    ///
    /// .. math::
    ///
    ///     g(r) = \alpha \big(\frac{r}{r_0}\big) ^ {-m} \bigg(1 + \big(\frac{r}{r_0}\big) ^ n\bigg) ^ {-k}
    ///
    /// When alpha=1.0, n=0.0, k=0.0, r0=1.0, and m=2.0, this is equivalent to a
    /// :math:`1/r^2` correction.
    ///
    /// This includes an optional time delay, which the non-gravitational forces are
    /// time delayed.
    ///
    /// The A1, A2, and A3 terms follow the NaN convention: a NaN value marks
    /// the parameter as free when the model is passed to the orbit fitting
    /// tools (the fit starts it at 0), while a concrete value freezes it at
    /// that value. Propagation treats NaN as 0.0. The defaults leave all
    /// three terms free, so ``new_comet()`` requests a fit of A1/A2/A3;
    /// pass explicit values to freeze them instead.
    ///
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (a1=f64::NAN, a2=f64::NAN, a3=f64::NAN, alpha=0.1112620426, r_0=2.808, m=2.15, n=5.093, k=4.6142, dt=0.0))]
    #[staticmethod]
    pub fn new_comet(
        a1: f64,
        a2: f64,
        a3: f64,
        alpha: f64,
        r_0: f64,
        m: f64,
        n: f64,
        k: f64,
        dt: f64,
    ) -> Self {
        Self::from_parts(
            NonGravKind::JplComet(JplCometNonGrav::new(alpha, r_0, m, n, k, dt)),
            vec![a1, a2, a3],
        )
    }

    /// This is the same as :py:meth:`NonGravModel.new_comet`, but with default values
    /// set so that :math:`g(r) = 1/r^2`.
    ///
    /// See :py:meth:`NonGravModel.new_comet` for more details, including the
    /// NaN convention: pass NaN for any of A1/A2/A3 to leave that parameter
    /// free during orbit fitting; concrete values are frozen.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (a1, a2, a3, alpha=1.0, r_0=1.0, m= 2.0, n=1.0, k=0.0, dt=0.0))]
    #[staticmethod]
    pub fn new_asteroid(
        a1: f64,
        a2: f64,
        a3: f64,
        alpha: f64,
        r_0: f64,
        m: f64,
        n: f64,
        k: f64,
        dt: f64,
    ) -> Self {
        Self::from_parts(
            NonGravKind::JplComet(JplCometNonGrav::new(alpha, r_0, m, n, k, dt)),
            vec![a1, a2, a3],
        )
    }

    /// Construct a physical radiation force model from Farnocchia et al. 2025.
    ///
    /// Models the body as an oblate spheroid with a fixed spin pole and
    /// computes solar radiation pressure plus thermal recoil (Yarkovsky)
    /// acceleration.
    ///
    /// The two fittable parameters are taken in the form used by the paper:
    /// ``a_over_m`` (Eq. 6) and ``lambda_0`` (Eq. 12). Pass ``float("nan")``
    /// for either to leave it free during orbit fitting; concrete values are
    /// frozen. Use the helpers
    /// :func:`kete.propagation.a_over_m_from_physical` and
    /// :func:`kete.propagation.lambda_0_from_physical` to compute them
    /// from physical surface inputs (density, thermal inertia, diameter,
    /// rotation period, etc.).
    ///
    /// Parameters
    /// ----------
    /// a_over_m :
    ///     Area-to-mass ratio in ``m^2 / kg`` (Eq. 6:
    ///     ``A/M = 3 / (4 * rho * R_P)``).
    /// lambda_0 :
    ///     Dimensionless thermal lag parameter at 1 AU (Eq. 12). At ``0``
    ///     (zero thermal lag) the transverse Yarkovsky component vanishes
    ///     but the radial recoil from instantaneous re-emission remains;
    ///     set ``absorptivity`` to ``0`` to disable the thermal terms
    ///     entirely.
    /// albedo :
    ///     Geometric (Lambert) albedo, enters SRP only.
    /// absorptivity :
    ///     ``alpha = 1 - A_B`` where ``A_B`` is the Bond albedo. Multiplies
    ///     the thermal terms.
    /// flattening :
    ///     Axis ratio ``e = R_P / R_E``, in ``(0, 1]``. Use ``1.0`` for a sphere.
    /// spin_pole :
    ///     Spin pole unit vector (any :class:`~kete.Vector` or length-3
    ///     sequence). Must be fixed in inertial space.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``a_over_m`` or ``lambda_0`` is neither NaN nor a finite value
    ///     ``>= 0``, if ``albedo`` or ``absorptivity`` is negative or not finite,
    ///     if ``flattening`` is outside ``(0, 1]``, or if ``spin_pole`` is zero.
    #[staticmethod]
    #[pyo3(signature = (a_over_m, lambda_0, albedo, absorptivity, flattening, spin_pole))]
    pub fn new_farnocchia(
        a_over_m: f64,
        lambda_0: f64,
        albedo: f64,
        absorptivity: f64,
        flattening: f64,
        spin_pole: VectorLike,
    ) -> PyResult<Self> {
        // NaN marks a free parameter. A concrete value must be physical.
        for (name, value) in [("a_over_m", a_over_m), ("lambda_0", lambda_0)] {
            if !value.is_nan() && !(value.is_finite() && value >= 0.0) {
                return Err(PyValueError::new_err(format!(
                    "'{name}' must be NaN (free) or finite and >= 0, got {value}"
                )));
            }
        }
        let pole = spin_pole.into_vector(PyFrames::Equatorial);
        let force = FarnocchiaNonGrav::new(albedo, absorptivity, flattening, pole)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self::from_parts(
            NonGravKind::Farnocchia(force),
            vec![a_over_m, lambda_0],
        ))
    }

    /// Construct a Farnocchia radiation force model from an absolute
    /// magnitude H and assumed physical properties.
    ///
    /// This is the usual entry point for a collisional family, where H is
    /// measured and the rest is assumed. It composes the standard chain:
    ///
    /// 1. ``H`` and ``albedo`` give the diameter,
    ///    ``D = 1329 / sqrt(albedo) * 10 ** (-H / 5)`` km (see
    ///    :func:`~kete.conversion.compute_diameter`).
    /// 2. the diameter and ``density`` give the area-to-mass ratio,
    ///    ``A/M = 3 / (4 * density * R)``, which is the radiation pressure
    ///    coupling (see :func:`~kete.propagation.a_over_m_from_physical`).
    ///    Since ``A/M`` scales as ``1 / (density * D)``, the drift rate
    ///    scales as ``1 / D``: five magnitudes fainter is ten times the
    ///    drift.
    /// 3. ``thermal_inertia`` and ``rotation_period`` give the thermal lag
    ///    (see :func:`~kete.propagation.lambda_0_from_physical`).
    ///
    /// The result is an ordinary Farnocchia :class:`NonGravModel`, usable
    /// with both :func:`~kete.propagation.propagate_n_body` and
    /// :class:`~kete.SymplecticSim`; the stored :attr:`a_over_m` and
    /// :attr:`lambda_0` are readable so the chain can be checked, and
    /// :meth:`bulk_density` / :meth:`thermal_inertia` invert it.
    ///
    /// Parameters
    /// ----------
    /// h_mag :
    ///     Absolute magnitude H.
    /// spin_pole :
    ///     Spin pole (any :class:`~kete.Vector` or length-3 sequence), fixed
    ///     in inertial space. A collisional family should be given randomly
    ///     oriented poles rather than one shared pole.
    /// albedo :
    ///     Geometric albedo. Sets the diameter along with H, and enters
    ///     radiation pressure.
    /// density :
    ///     Bulk density in ``kg / m^3``.
    /// thermal_inertia :
    ///     Thermal inertia in SI units (``J m^-2 K^-1 s^-1/2``).
    /// rotation_period :
    ///     Rotation period in hours.
    /// emissivity :
    ///     Surface emissivity.
    /// absorptivity :
    ///     ``alpha = 1 - A_B`` where ``A_B`` is the Bond albedo.
    /// flattening :
    ///     Axis ratio ``e = R_P / R_E``. Use ``1.0`` for a sphere.
    #[staticmethod]
    #[pyo3(signature = (h_mag, spin_pole, albedo=0.15, density=2500.0, thermal_inertia=200.0,
        rotation_period=6.0, emissivity=0.9, absorptivity=0.9, flattening=1.0))]
    #[allow(
        clippy::too_many_arguments,
        reason = "physical surface properties, all keyword arguments with defaults"
    )]
    pub fn new_farnocchia_from_h_mag(
        h_mag: f64,
        spin_pole: VectorLike,
        albedo: f64,
        density: f64,
        thermal_inertia: f64,
        rotation_period: f64,
        emissivity: f64,
        absorptivity: f64,
        flattening: f64,
    ) -> PyResult<Self> {
        if !(albedo > 0.0 && albedo.is_finite()) {
            Err(PyValueError::new_err(format!(
                "albedo must be finite and positive to convert H to a diameter, found {albedo}."
            )))?;
        }
        if !(density > 0.0 && density.is_finite()) {
            Err(PyValueError::new_err(format!(
                "density must be finite and positive, found {density}."
            )))?;
        }
        if !(rotation_period > 0.0 && rotation_period.is_finite()) {
            Err(PyValueError::new_err(format!(
                "rotation_period must be finite and positive, found {rotation_period}."
            )))?;
        }
        let diameter = diam_from_h_mag_albedo(h_mag, albedo, C_V);
        let a_over_m = a_over_m_from_physical(density, diameter, flattening);
        let lambda_0 = lambda_0_from_physical(
            thermal_inertia,
            emissivity,
            absorptivity,
            flattening,
            rotation_period,
        );
        Self::new_farnocchia(
            a_over_m,
            lambda_0,
            albedo,
            absorptivity,
            flattening,
            spin_pole,
        )
    }

    /// Stored area-to-mass ratio ``A/M`` (``m^2 / kg``) for a
    /// ``FarnocchiaModel`` (Eq. 6).
    ///
    /// Returns ``NaN`` unless this is a ``FarnocchiaModel``.
    #[getter]
    pub fn a_over_m(&self) -> f64 {
        match self.force {
            NonGravKind::Farnocchia(_) => self.values[0],
            _ => f64::NAN,
        }
    }

    /// Stored thermal parameter ``lambda_0`` (dimensionless, Eq. 12) for a
    /// ``FarnocchiaModel``.
    ///
    /// Returns ``NaN`` unless this is a ``FarnocchiaModel``.
    #[getter]
    pub fn lambda_0(&self) -> f64 {
        match self.force {
            NonGravKind::Farnocchia(_) => self.values[1],
            _ => f64::NAN,
        }
    }

    /// Recover bulk density (``kg / m^3``) of this ``FarnocchiaModel`` given
    /// the auxiliary inputs that were collapsed away at construction time.
    ///
    /// Returns ``NaN`` unless this is a ``FarnocchiaModel``.
    pub fn bulk_density(&self, diameter: f64) -> f64 {
        match self.force {
            NonGravKind::Farnocchia(ref f) => {
                density_from_a_over_m(self.values[0], diameter, f.flattening)
            }
            _ => f64::NAN,
        }
    }

    /// Recover surface thermal inertia ``Gamma`` (SI units) of this
    /// ``FarnocchiaModel`` given the auxiliary inputs that were collapsed
    /// away at construction time. ``rotation_period`` is in hours.
    ///
    /// Returns ``NaN`` unless this is a ``FarnocchiaModel``.
    pub fn thermal_inertia(&self, emissivity: f64, rotation_period: f64) -> f64 {
        match self.force {
            NonGravKind::Farnocchia(ref f) => thermal_inertia_from_lambda_0(
                self.values[1],
                emissivity,
                f.absorptivity,
                f.flattening,
                rotation_period,
            ),
            _ => f64::NAN,
        }
    }

    /// Return a dictionary of the values used in this non-grav model.
    #[getter]
    pub fn items(&self) -> HashMap<String, f64> {
        // The fittable parameters under their own names, then the fixed constants.
        let mut values: HashMap<String, f64> = self
            .force
            .free_param_names()
            .into_iter()
            .map(str::to_string)
            .zip(self.values.iter().copied())
            .collect();
        let mut constant = |name: &str, value: f64| {
            let _ = values.insert(name.to_string(), value);
        };
        match self.force {
            NonGravKind::Dust(_) => {}
            NonGravKind::JplComet(ref f) => {
                constant("alpha", f.alpha);
                constant("r_0", f.r_0);
                constant("m", f.m);
                constant("n", f.n);
                constant("k", f.k);
                constant("dt", f.dt);
            }
            NonGravKind::Farnocchia(ref f) => {
                let raw: [f64; 3] = f.spin_pole.into();
                constant("albedo", f.albedo);
                constant("absorptivity", f.absorptivity);
                constant("flattening", f.flattening);
                constant("spin_pole_x", raw[0]);
                constant("spin_pole_y", raw[1]);
                constant("spin_pole_z", raw[2]);
            }
        }
        values
    }

    /// Text representation of this object.
    pub fn __repr__(&self) -> String {
        // NaN (a free parameter) has no Python literal; keep the repr
        // eval-able.
        fn f(v: f64) -> String {
            if v.is_nan() {
                "float(\"nan\")".into()
            } else {
                format!("{v:?}")
            }
        }
        let v = &self.values;
        let base = match self.force {
            NonGravKind::Dust(_) => {
                format!("kete.propagation.NonGravModel.new_dust(beta={})", f(v[0]))
            }
            NonGravKind::JplComet(ref c) => format!(
                "kete.propagation.NonGravModel.new_comet(a1={}, a2={}, a3={}, alpha={:?}, r_0={:?}, m={:?}, n={:?}, k={:?}, dt={:?})",
                f(v[0]),
                f(v[1]),
                f(v[2]),
                c.alpha,
                c.r_0,
                c.m,
                c.n,
                c.k,
                c.dt,
            ),
            NonGravKind::Farnocchia(ref c) => {
                let raw: [f64; 3] = c.spin_pole.into();
                format!(
                    "kete.propagation.NonGravModel.new_farnocchia(a_over_m={}, lambda_0={}, albedo={:?}, absorptivity={:?}, flattening={:?}, spin_pole={raw:?})",
                    f(v[0]),
                    f(v[1]),
                    c.albedo,
                    c.absorptivity,
                    c.flattening,
                )
            }
        };
        let freed: Vec<String> = self
            .force
            .free_param_names()
            .into_iter()
            .zip(self.values.iter().zip(self.free_flags()))
            .filter(|(_, (v, free))| *free && !v.is_nan())
            .map(|(name, _)| format!("{name:?}"))
            .collect();
        if freed.is_empty() {
            base
        } else {
            format!("{base}.with_free({})", freed.join(", "))
        }
    }

    /// Free parameters for orbit fitting, starting from their current values.
    ///
    /// Returns a copy of this model in which the named fittable parameters (every
    /// one when none are named) are free, like a NaN, but keep their values. An orbit
    /// fit starts those parameters there, and takes ``initial_state`` as the
    /// matching starting point for the joint fit: the gravity-only pass that
    /// otherwise precedes the non-gravitational fit is skipped. This is how a fit
    /// is continued from an earlier one::
    ///
    ///     fit = kete.orbit_fitting.fit_orbit(fit.state, obs, non_grav=fit.non_grav.with_free())
    ///
    /// Propagation uses the values as given. Parameters that are NaN stay free and
    /// start from 0.
    ///
    /// Parameters
    /// ----------
    /// names :
    ///     Names of fittable parameters, as in :py:attr:`free_parameters` of a model
    ///     with every parameter NaN (for example ``"a1"``).
    #[pyo3(signature = (*names))]
    pub fn with_free(&self, names: Vec<String>) -> PyResult<Self> {
        let all = self.force.free_param_names();
        let mut flags = self.free_flags();
        for name in &names {
            let Some(i) = all.iter().position(|n| n == name) else {
                Err(PyValueError::new_err(format!(
                    "{name:?} is not a fittable parameter of this model; they are {all:?}."
                )))?
            };
            flags[i] = true;
        }
        if names.is_empty() {
            flags.fill(true);
        }
        Ok(Self {
            freed: flags,
            ..self.clone()
        })
    }

    /// The names of the parameters an orbit fit would fit. These are the NaN values
    /// and those freed by :py:meth:`with_free`.
    #[getter]
    pub fn free_parameters(&self) -> Vec<String> {
        self.force
            .free_param_names()
            .into_iter()
            .zip(self.free_flags())
            .filter(|(_, free)| *free)
            .map(|(name, _)| name.to_string())
            .collect()
    }
}

/// Compute ``A/M`` (``m^2 / kg``) from physical surface inputs
/// (Farnocchia 2025 Eq. 6).
///
/// Parameters
/// ----------
/// density :
///     Bulk density in ``kg / m^3``.
/// diameter :
///     Volume-equivalent diameter in km.
/// flattening :
///     Axis ratio ``e = R_P / R_E``. Use ``1.0`` for a sphere.
#[pyfunction]
#[pyo3(name = "a_over_m_from_physical")]
pub fn py_a_over_m_from_physical(density: f64, diameter: f64, flattening: f64) -> f64 {
    a_over_m_from_physical(density, diameter, flattening)
}

/// Inverse of :func:`a_over_m_from_physical`: solve for bulk density
/// (``kg / m^3``) given ``A/M``, ``diameter`` (km), and ``flattening``.
#[pyfunction]
#[pyo3(name = "density_from_a_over_m")]
pub fn py_density_from_a_over_m(a_over_m: f64, diameter: f64, flattening: f64) -> f64 {
    density_from_a_over_m(a_over_m, diameter, flattening)
}

/// Compute ``lambda_0`` (dimensionless, Farnocchia 2025 Eq. 12) from
/// physical surface inputs.
///
/// Parameters
/// ----------
/// thermal_inertia :
///     Surface thermal inertia ``Gamma`` in SI units
///     (``J m^-2 s^-1/2 K^-1``).
/// emissivity :
///     Thermal emissivity.
/// absorptivity :
///     ``alpha = 1 - A_B`` where ``A_B`` is the Bond albedo.
/// flattening :
///     Axis ratio ``e = R_P / R_E``.
/// rotation_period :
///     Rotation period in hours.
#[pyfunction]
#[pyo3(name = "lambda_0_from_physical")]
pub fn py_lambda_0_from_physical(
    thermal_inertia: f64,
    emissivity: f64,
    absorptivity: f64,
    flattening: f64,
    rotation_period: f64,
) -> f64 {
    lambda_0_from_physical(
        thermal_inertia,
        emissivity,
        absorptivity,
        flattening,
        rotation_period,
    )
}

/// Inverse of :func:`lambda_0_from_physical`: solve for thermal inertia
/// (SI units) given ``lambda_0`` and the auxiliary surface inputs.
/// ``rotation_period`` is in hours.
#[pyfunction]
#[pyo3(name = "thermal_inertia_from_lambda_0")]
pub fn py_thermal_inertia_from_lambda_0(
    lambda_0: f64,
    emissivity: f64,
    absorptivity: f64,
    flattening: f64,
    rotation_period: f64,
) -> f64 {
    thermal_inertia_from_lambda_0(
        lambda_0,
        emissivity,
        absorptivity,
        flattening,
        rotation_period,
    )
}
