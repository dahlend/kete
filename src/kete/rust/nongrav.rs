//! Python wrapper for non-gravitational force models.
//!
//! Internally stores construction-time data per variant (Dust /
//! JplComet / Farnocchia) and produces a [`NonGravEntry`] (typed ParameterizedForce
//! template plus free-parameter values) on demand for use with the
//! Rust `ParameterizedForce` machinery. The closed-set tagged union exists only at
//! the Python boundary; on the Rust side each variant is a separate
//! typed ParameterizedForce impl.
use std::collections::HashMap;
use std::sync::Arc;

use kete_core::{
    errors::Error,
    forces::{
        ArcForce, DustNonGrav, FarnocchiaNonGrav, FrozenForce, FrozenNonGrav, JplCometNonGrav,
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
/// releases. Internally the wrapper holds typed ParameterizedForce impls and emits
/// a `kete_core` `NonGravEntry` (typed ParameterizedForce template plus free-parameter
/// values) on demand for the propagation machinery.
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
    pub fn to_force(&self) -> ArcForce {
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
    /// batch propagation, and covariance sampling — anywhere a single concrete
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
    pub fn to_mask(&self) -> ParameterMask<ArcForce> {
        let force = self.to_force();
        let n = force.n_free_params();
        ParameterMask::new(force, vec![None; n]).expect("n matches n_free_params")
    }

    /// Reconstruct a Python wrapper from a [`ParameterizedForce`] template and
    /// its current free-parameter values.
    ///
    /// Returns `None` if the template is not one of the canonical
    /// `DustNonGrav` / `JplCometNonGrav` / `FarnocchiaNonGrav`.
    pub fn from_force(template: &ArcForce, values: &[f64]) -> Option<Self> {
        let any = template.as_any()?;

        if any.downcast_ref::<DustNonGrav>().is_some() {
            return Some(Self(NonGravData::Dust {
                beta: *values.first()?,
            }));
        }
        if let Some(typed) = any.downcast_ref::<JplCometNonGrav>() {
            return Some(Self(NonGravData::JplComet {
                a1: values.first().copied().unwrap_or(0.0),
                a2: values.get(1).copied().unwrap_or(0.0),
                a3: values.get(2).copied().unwrap_or(0.0),
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
                a_over_m: values.first().copied().unwrap_or(0.0),
                lambda_0: values.get(1).copied().unwrap_or(0.0),
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
    #[pyo3(signature=(density=1000.0, c_pr=1.19, q_pr=1.0))]
    pub fn diameter(&self, density: f64, c_pr: f64, q_pr: f64) -> f64 {
        match self.0 {
            NonGravData::Dust { beta } => (c_pr * q_pr) / (beta * density),
            _ => f64::NAN,
        }
    }

    /// Construct a JPL Comet RTN-frame non-grav model.
    ///
    /// Defaults match JPL Horizons' standard comet drop-off
    /// coefficients (Marsden et al.).
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

    /// JPL Comet model with `g(r) = 1/r^2` defaults; suitable for
    /// asteroids with a Yarkovsky-style RTN-frame parameterization.
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

    /// Farnocchia 2025 oblate-spheroid radiation model.
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

    /// Stored area-to-mass ratio (Farnocchia only).
    #[getter]
    pub fn a_over_m(&self) -> f64 {
        match self.0 {
            NonGravData::Farnocchia { a_over_m, .. } => a_over_m,
            _ => f64::NAN,
        }
    }

    /// Stored thermal parameter `lambda_0` (Farnocchia only).
    #[getter]
    pub fn lambda_0(&self) -> f64 {
        match self.0 {
            NonGravData::Farnocchia { lambda_0, .. } => lambda_0,
            _ => f64::NAN,
        }
    }

    /// Recover bulk density from a Farnocchia model.
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

    /// Recover surface thermal inertia from a Farnocchia model.
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

/// Compute ``A/M`` from physical surface inputs (Farnocchia 2025 Eq. 6).
#[pyfunction]
#[pyo3(name = "a_over_m_from_physical")]
pub fn py_a_over_m_from_physical(density: f64, diameter: f64, flattening: f64) -> f64 {
    a_over_m_from_physical(density, diameter, flattening)
}

/// Inverse of :func:`a_over_m_from_physical`.
#[pyfunction]
#[pyo3(name = "density_from_a_over_m")]
pub fn py_density_from_a_over_m(a_over_m: f64, diameter: f64, flattening: f64) -> f64 {
    density_from_a_over_m(a_over_m, diameter, flattening)
}

/// Compute ``lambda_0`` from physical surface inputs (Farnocchia Eq. 12).
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

/// Inverse of :func:`lambda_0_from_physical`.
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
