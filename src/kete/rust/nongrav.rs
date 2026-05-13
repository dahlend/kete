//! Python wrapper for non-gravitational force models.
//!
//! Exposes three non-grav variants to Python as a single class
//! (`NonGravModel`): dust radiation pressure, JPL comet outgassing, and
//! the Farnocchia thermal recoil model. Each variant stores the physical
//! inputs given at construction time (e.g. `beta` for dust, `a1/a2/a3`
//! for comets) and converts them to the underlying Rust force type on
//! demand.
use std::collections::HashMap;
use std::sync::Arc;

use kete_core::{
    errors::Error,
    forces::{
        DustNonGrav, FarnocchiaNonGrav, FrozenForce, FrozenNonGrav, JplCometNonGrav, NonGravForce,
        ParameterMask, ParameterizedForce, a_over_m_from_physical, density_from_a_over_m,
        lambda_0_from_physical, thermal_inertia_from_lambda_0,
    },
    frames::{Equatorial, Vector},
};
use pyo3::{PyResult, exceptions::PyValueError, pyclass, pyfunction, pymethods};

use crate::frame::PyFrames;
use crate::vector::VectorLike;

/// Per-variant data stored on the Python wrapper.
#[derive(Debug, Clone)]
enum NonGravData {
    Dust {
        beta: f64,
    },
    JplComet {
        a1: f64,
        a2: f64,
        a3: f64,
        alpha: f64,
        r_0: f64,
        m: f64,
        n: f64,
        k: f64,
        dt: f64,
    },
    Farnocchia {
        a_over_m: f64,
        lambda_0: f64,
        albedo: f64,
        absorptivity: f64,
        flattening: f64,
        spin_pole: Vector<Equatorial>,
    },
}

/// Non-gravitational force models for n-body propagation.
///
/// The closed-set hierarchy exposed to Python is unchanged from prior
/// releases. Internally the wrapper holds a typed ParameterizedForce impl
/// and produces a FrozenForce or ParameterMask on demand for the
/// propagation machinery.
#[pyclass(
    frozen,
    module = "kete.propagation",
    name = "NonGravModel",
    from_py_object
)]
#[derive(Debug, Clone)]
pub struct PyNonGravModel(NonGravData);

impl PyNonGravModel {
    /// Return the typed ParameterizedForce template for this model.
    pub fn to_force(&self) -> NonGravForce {
        match self.0 {
            NonGravData::Dust { .. } => Arc::new(DustNonGrav),
            NonGravData::JplComet {
                alpha,
                r_0,
                m,
                n,
                k,
                dt,
                ..
            } => Arc::new(JplCometNonGrav::new(alpha, r_0, m, n, k, dt)),
            NonGravData::Farnocchia {
                albedo,
                absorptivity,
                flattening,
                spin_pole,
                ..
            } => Arc::new(
                FarnocchiaNonGrav::new(albedo, absorptivity, flattening, spin_pole)
                    .expect("validated at construction"),
            ),
        }
    }

    /// Return the initial free-parameter values for this model.
    pub fn initial_values(&self) -> Vec<f64> {
        match self.0 {
            NonGravData::Dust { beta } => vec![beta],
            NonGravData::JplComet { a1, a2, a3, .. } => vec![a1, a2, a3],
            NonGravData::Farnocchia {
                a_over_m, lambda_0, ..
            } => vec![a_over_m, lambda_0],
        }
    }

    /// Return a [`FrozenNonGrav`] with the initial parameter values baked in.
    ///
    /// Suitable for plain [`State`](kete_core::state::State) propagation,
    /// batch propagation, and covariance sampling -- anywhere a single concrete
    /// parameter estimate drives the trajectory.
    pub fn to_frozen(&self) -> FrozenNonGrav {
        let force = self.to_force();
        let values = self.initial_values();
        FrozenForce::new(force, values).expect("n matches n_free_params")
    }

    /// Return an all-`None` [`ParameterMask`] wrapping the typed ParameterizedForce template.
    ///
    /// The mask exposes all parameters as free (variational); parameter values
    /// are supplied at integration time from the carrying state's `free_params`.
    /// This is the representation stored on [`PyUncertainState`] and
    /// [`PyDiffuseState`].
    pub fn to_mask(&self) -> ParameterMask<NonGravForce> {
        let force = self.to_force();
        let n = force.n_free_params();
        ParameterMask::new(force, vec![None; n]).expect("n matches n_free_params")
    }

    /// Reconstruct a Python wrapper from a [`ParameterizedForce`] template and
    /// its current free-parameter values.
    ///
    /// Returns `None` if the template is not one of the canonical
    /// `DustNonGrav` / `JplCometNonGrav` / `FarnocchiaNonGrav`.
    pub fn from_force(template: &NonGravForce, values: &[f64]) -> Option<Self> {
        let any = template.as_any()?;

        if any.downcast_ref::<DustNonGrav>().is_some() {
            return Some(Self(NonGravData::Dust {
                beta: *values.first()?,
            }));
        }
        if let Some(typed) = any.downcast_ref::<JplCometNonGrav>() {
            return Some(Self(NonGravData::JplComet {
                a1: values.first().copied().unwrap_or(f64::NAN),
                a2: values.get(1).copied().unwrap_or(f64::NAN),
                a3: values.get(2).copied().unwrap_or(f64::NAN),
                alpha: typed.alpha,
                r_0: typed.r_0,
                m: typed.m,
                n: typed.n,
                k: typed.k,
                dt: typed.dt,
            }));
        }
        if let Some(typed) = any.downcast_ref::<FarnocchiaNonGrav>() {
            return Some(Self(NonGravData::Farnocchia {
                a_over_m: values.first().copied().unwrap_or(f64::NAN),
                lambda_0: values.get(1).copied().unwrap_or(f64::NAN),
                albedo: typed.albedo,
                absorptivity: typed.absorptivity,
                flattening: typed.flattening,
                spin_pole: typed.spin_pole,
            }));
        }
        None
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
    /// Parameters
    /// ==========
    /// beta:
    ///     Beta value of the dust, if this is specified, all other inputs are ignored.
    ///     If this value is specified, diameter cannot be specified.
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
    #[pyo3(signature=(beta=None, diameter=None, density=1000.0, c_pr=1.19e-3, q_pr=1.0))]
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
        Ok(Self(NonGravData::Dust { beta: beta_value }))
    }

    /// Get the beta value for this dust model.
    #[getter]
    pub fn beta(&self) -> f64 {
        match self.0 {
            NonGravData::Dust { beta } => beta,
            _ => f64::NAN,
        }
    }

    /// Estimate the diameter of the dust particle in meters.
    ///
    /// Only works for dust models, returns NaN for asteroid/comet models.
    ///
    /// Parameters
    /// ==========
    /// density:
    ///     Density in kg/m^3, defaults to 1000 kg/m^3
    /// c_pr:
    ///     Radiation pressure coefficient, defaults to 1.19 kg/m^2
    /// q_pr:
    ///     Scattering efficiency for radiation pressure, defaults to 1.0
    ///     1.0 is a good estimate for particles larger than 1um (Burns, Lamy & Soter 1979)
    #[pyo3(signature=(density=1000.0, c_pr=1.19, q_pr=1.0))]
    pub fn diameter(&self, density: f64, c_pr: f64, q_pr: f64) -> f64 {
        match self.0 {
            NonGravData::Dust { beta } => (c_pr * q_pr) / (beta * density),
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
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (a1=0.0, a2=0.0, a3=0.0, alpha=0.1112620426, r_0=2.808, m=2.15, n=5.093, k=4.6142, dt=0.0))]
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
        Self(NonGravData::JplComet {
            a1,
            a2,
            a3,
            alpha,
            r_0,
            m,
            n,
            k,
            dt,
        })
    }

    /// This is the same as :py:meth:`NonGravModel.new_comet`, but with default values
    /// set so that :math:`g(r) = 1/r^2`.
    ///
    /// See :py:meth:`NonGravModel.new_comet` for more details.
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
        Self(NonGravData::JplComet {
            a1,
            a2,
            a3,
            alpha,
            r_0,
            m,
            n,
            k,
            dt,
        })
    }

    /// Construct a physical radiation force model from Farnocchia et al. 2025.
    ///
    /// Models the body as an oblate spheroid with a fixed spin pole and
    /// computes solar radiation pressure plus thermal recoil (Yarkovsky)
    /// acceleration.
    ///
    /// The two fittable parameters are taken in the form used by the paper:
    /// ``a_over_m`` (Eq. 6) and ``lambda_0`` (Eq. 12). Use the helpers
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
    ///     Dimensionless thermal lag parameter at 1 AU (Eq. 12). Set to
    ///     ``0`` to disable the thermal (Yarkovsky) component.
    /// albedo :
    ///     Geometric (Lambert) albedo, enters SRP only.
    /// absorptivity :
    ///     ``alpha = 1 - A_B`` where ``A_B`` is the Bond albedo. Multiplies
    ///     the thermal terms.
    /// flattening :
    ///     Axis ratio ``e = R_P / R_E``. Use ``1.0`` for a sphere.
    /// spin_pole :
    ///     Spin pole unit vector (any :class:`~kete.Vector` or length-3
    ///     sequence). Must be fixed in inertial space.
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
        let pole = spin_pole.into_vector(PyFrames::Equatorial);
        // Validate construction inputs by trying to build the typed
        // FarnocchiaNonGrav force. The actual instance is rebuilt on
        // demand by `to_model`.
        let _ = FarnocchiaNonGrav::new(albedo, absorptivity, flattening, pole)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self(NonGravData::Farnocchia {
            a_over_m,
            lambda_0,
            albedo,
            absorptivity,
            flattening,
            spin_pole: pole,
        }))
    }

    /// Stored area-to-mass ratio ``A/M`` (``m^2 / kg``) for a
    /// ``FarnocchiaModel`` (Eq. 6).
    ///
    /// Returns ``NaN`` unless this is a ``FarnocchiaModel``.
    #[getter]
    pub fn a_over_m(&self) -> f64 {
        match self.0 {
            NonGravData::Farnocchia { a_over_m, .. } => a_over_m,
            _ => f64::NAN,
        }
    }

    /// Stored thermal parameter ``lambda_0`` (dimensionless, Eq. 12) for a
    /// ``FarnocchiaModel``.
    ///
    /// Returns ``NaN`` unless this is a ``FarnocchiaModel``.
    #[getter]
    pub fn lambda_0(&self) -> f64 {
        match self.0 {
            NonGravData::Farnocchia { lambda_0, .. } => lambda_0,
            _ => f64::NAN,
        }
    }

    /// Recover bulk density (``kg / m^3``) of this ``FarnocchiaModel`` given
    /// the auxiliary inputs that were collapsed away at construction time.
    ///
    /// Returns ``NaN`` unless this is a ``FarnocchiaModel``.
    pub fn bulk_density(&self, diameter: f64) -> f64 {
        match self.0 {
            NonGravData::Farnocchia {
                a_over_m,
                flattening,
                ..
            } => density_from_a_over_m(a_over_m, diameter, flattening),
            _ => f64::NAN,
        }
    }

    /// Recover surface thermal inertia ``Gamma`` (SI units) of this
    /// ``FarnocchiaModel`` given the auxiliary inputs that were collapsed
    /// away at construction time. ``rotation_period`` is in hours.
    ///
    /// Returns ``NaN`` unless this is a ``FarnocchiaModel``.
    pub fn thermal_inertia(&self, emissivity: f64, rotation_period: f64) -> f64 {
        match self.0 {
            NonGravData::Farnocchia {
                lambda_0,
                absorptivity,
                flattening,
                ..
            } => thermal_inertia_from_lambda_0(
                lambda_0,
                emissivity,
                absorptivity,
                flattening,
                rotation_period,
            ),
            _ => f64::NAN,
        }
    }

    /// Return a dictionary of the values used in this non-grav model.
    #[getter]
    pub fn items(&self) -> HashMap<String, f64> {
        let mut values = HashMap::new();
        match self.0 {
            NonGravData::Dust { beta } => {
                let _ = values.insert("beta".to_string(), beta);
            }
            NonGravData::JplComet {
                a1,
                a2,
                a3,
                alpha,
                r_0,
                m,
                n,
                k,
                dt,
            } => {
                let _ = values.insert("a1".to_string(), a1);
                let _ = values.insert("a2".to_string(), a2);
                let _ = values.insert("a3".to_string(), a3);
                let _ = values.insert("alpha".to_string(), alpha);
                let _ = values.insert("r_0".to_string(), r_0);
                let _ = values.insert("m".to_string(), m);
                let _ = values.insert("n".to_string(), n);
                let _ = values.insert("k".to_string(), k);
                let _ = values.insert("dt".to_string(), dt);
            }
            NonGravData::Farnocchia {
                a_over_m,
                lambda_0,
                albedo,
                absorptivity,
                flattening,
                spin_pole,
            } => {
                let raw: [f64; 3] = spin_pole.into();
                let _ = values.insert("a_over_m".to_string(), a_over_m);
                let _ = values.insert("lambda_0".to_string(), lambda_0);
                let _ = values.insert("albedo".to_string(), albedo);
                let _ = values.insert("absorptivity".to_string(), absorptivity);
                let _ = values.insert("flattening".to_string(), flattening);
                let _ = values.insert("spin_pole_x".to_string(), raw[0]);
                let _ = values.insert("spin_pole_y".to_string(), raw[1]);
                let _ = values.insert("spin_pole_z".to_string(), raw[2]);
            }
        }
        values
    }

    /// Text representation of this object.
    pub fn __repr__(&self) -> String {
        match self.0 {
            NonGravData::Dust { beta } => {
                format!("kete.propagation.NonGravModel.new_dust(beta={beta:?})")
            }
            NonGravData::JplComet {
                a1,
                a2,
                a3,
                alpha,
                r_0,
                m,
                n,
                k,
                dt,
            } => format!(
                "kete.propagation.NonGravModel.new_comet(a1={a1:?}, a2={a2:?}, a3={a3:?}, alpha={alpha:?}, r_0={r_0:?}, m={m:?}, n={n:?}, k={k:?}, dt={dt:?})",
            ),
            NonGravData::Farnocchia {
                a_over_m,
                lambda_0,
                albedo,
                absorptivity,
                flattening,
                spin_pole,
            } => {
                let raw: [f64; 3] = spin_pole.into();
                format!(
                    "kete.propagation.NonGravModel.new_farnocchia(a_over_m={a_over_m:?}, lambda_0={lambda_0:?}, albedo={albedo:?}, absorptivity={absorptivity:?}, flattening={flattening:?}, spin_pole={raw:?})",
                )
            }
        }
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
