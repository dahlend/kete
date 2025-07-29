//! Python support for non-gravitational forces
use std::collections::HashMap;

use kete_core::{errors::Error, propagation::NonGravModel};
use pyo3::{PyResult, exceptions::PyValueError, pyclass, pymethods};

/// Non-gravitational force models.
///
/// This is used optionally by the N-Body propagation methods to compute orbits
/// including non-gravitational forces, such as solar radiation pressure, or
/// poynting-robertson force.
///
/// There are two generic non-gravitational models available, one is specifically
/// intended for dust modeling, and includes the solar radiation pressure, the other
/// model is a mathematical match to the JPL Horizons comet model.
///
/// See :py:meth:`NonGravModel.new_dust` and :py:meth:`NonGravModel.new_comet` for more
/// details. Note that the Comet model can also represent asteroids which undergo the
/// Yarkovsky effect, see :py:meth:`NonGravModel.new_asteroid`, which is a convenience
/// function over the :py:meth:`NonGravModel.new_comet` method, but with 1/r^2 falloff.
///
#[pyclass(frozen, module = "kete.propagation", name = "NonGravModel")]
#[derive(Debug, Clone)]
pub struct PyNonGravModel(pub NonGravModel);

#[pymethods]
impl PyNonGravModel {
    /// Unused constructor for non-grav models.
    #[allow(clippy::new_without_default)]
    #[new]
    pub fn new() -> PyResult<Self> {
        Err(Error::ValueError("Non-gravitational force models need to be constructed using either the new_dust, new_comet, or new_asteroid methods.".into()))?
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
    ///     Radiation pressure coefficient, defaults to 1.19 kg/m^2
    /// q_pr:
    ///     Scattering efficiency for radiation pressure, defaults to 1.0
    ///     1.0 is a good estimate for particles larger than 1um (Burns, Lamy & Soter 1979)
    #[staticmethod]
    #[pyo3(signature=(beta=None, diameter=None, density=1000.0, c_pr=1.19, q_pr=1.0))]
    pub fn new_dust(
        beta: Option<f64>,
        diameter: Option<f64>,
        density: f64,
        c_pr: f64,
        q_pr: f64,
    ) -> PyResult<Self> {
        match (beta, diameter) {
            (None, None) => Err(PyValueError::new_err("Must specify beta or diameter."))?,
            (Some(_), Some(_)) => Err(PyValueError::new_err(
                "Cannot specify both beta and diameter.",
            ))?,
            (Some(beta), None) => Ok(Self(NonGravModel::Dust { beta })),
            (None, Some(diameter)) => Ok(Self(NonGravModel::Dust {
                beta: (c_pr * q_pr) / (diameter * density),
            })),
        }
    }

    #[getter]
    /// Get the beta value for this dust model.
    pub fn beta(&self) -> f64 {
        match self.0 {
            NonGravModel::Dust { beta } => beta,
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
            NonGravModel::Dust { beta } => (c_pr * q_pr) / (beta * density),
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
        Self(NonGravModel::JplComet {
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
        Self(NonGravModel::JplComet {
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

    #[getter]
    /// Return a dictionary of the values used in this non-grav model.
    pub fn items(&self) -> HashMap<String, f64> {
        match self.0 {
            NonGravModel::Dust { beta } => {
                let mut values = HashMap::new();
                let _ = values.insert("beta".to_string(), beta);
                values
            }
            NonGravModel::JplComet {
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
                let mut values = HashMap::new();
                let _ = values.insert("a1".to_string(), a1);
                let _ = values.insert("a2".to_string(), a2);
                let _ = values.insert("a3".to_string(), a3);
                let _ = values.insert("alpha".to_string(), alpha);
                let _ = values.insert("r_0".to_string(), r_0);
                let _ = values.insert("m".to_string(), m);
                let _ = values.insert("n".to_string(), n);
                let _ = values.insert("k".to_string(), k);
                let _ = values.insert("dt".to_string(), dt);
                values
            }
        }
    }

    /// Text representation of this object
    pub fn __repr__(&self) -> String {
        match self.0 {
            NonGravModel::Dust { beta } => {
                format!("kete.propagation.NonGravModel.new_dust(beta={beta:?})")
            }
            NonGravModel::JplComet {
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
        }
    }
}
