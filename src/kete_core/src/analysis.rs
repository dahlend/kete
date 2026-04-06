//! Orbital analysis tools.
//!
//! Functions for characterizing orbits: Hill radius, sphere of influence,
//! Tisserand parameter, and related quantities.

// Docstrings in this module use NumPy-style formatting (e.g. `param :`) so they
// render correctly in both Rust and Python (via cfg_attr pyfunction/pyclass).
#![allow(clippy::doc_markdown, reason = "NumPy-style Python docstrings")]

use crate::errors::{Error, KeteResult};
use crate::forces::GravParams;
use crate::frames::InertialFrame;
use crate::state::State;
use nalgebra::Vector3;

/// Compute the Hill radius of a body orbiting a more massive central body.
///
///   r_H = a (1 - e) (m / (3 M))^(1/3)
///
/// Parameters
/// ----------
/// semi_major :
///     Semi-major axis of the body's orbit in AU.
/// eccentricity :
///     Eccentricity of the body's orbit.
/// gm_body :
///     Gravitational parameter of the orbiting body (AU^3/day^2).
/// gm_central :
///     Gravitational parameter of the central body (AU^3/day^2).
///
/// Returns
/// -------
/// float
///     Hill radius in AU.
#[must_use]
pub fn hill_radius(semi_major: f64, eccentricity: f64, gm_body: f64, gm_central: f64) -> f64 {
    semi_major * (1.0 - eccentricity) * (gm_body / (3.0 * gm_central)).cbrt()
}

/// Compute the Laplace sphere-of-influence radius.
///
///   r_SOI = a (m / M)^(2/5)
///
/// Parameters
/// ----------
/// semi_major :
///     Semi-major axis of the body's orbit in AU.
/// gm_body :
///     Gravitational parameter of the orbiting body (AU^3/day^2).
/// gm_central :
///     Gravitational parameter of the central body (AU^3/day^2).
///
/// Returns
/// -------
/// float
///     Sphere of influence radius in AU.
#[must_use]
pub fn sphere_of_influence(semi_major: f64, gm_body: f64, gm_central: f64) -> f64 {
    semi_major * (gm_body / gm_central).powf(0.4)
}

/// Compute the Tisserand parameter relative to a perturbing body.
///
///   T = a_P / a + 2 cos(i) sqrt((a / a_P) (1 - e^2))
///
/// Parameters
/// ----------
/// semi_major :
///     Semi-major axis of the small body's orbit in AU.
/// eccentricity :
///     Eccentricity of the small body's orbit.
/// inclination :
///     Inclination of the small body's orbit in radians.
/// a_planet :
///     Semi-major axis of the perturber's orbit in AU.
///
/// Returns
/// -------
/// float
///     Tisserand parameter, dimensionless.
#[cfg_attr(feature = "pyo3", pyo3::pyfunction)]
#[must_use]
pub fn tisserand(semi_major: f64, eccentricity: f64, inclination: f64, a_planet: f64) -> f64 {
    a_planet / semi_major
        + 2.0
            * inclination.cos()
            * ((semi_major / a_planet) * (1.0 - eccentricity * eccentricity)).sqrt()
}

/// Look up the gravitational parameter for a NAIF center ID.
///
/// # Errors
/// Returns an error if the center ID is not in the known masses list.
fn gm_for_center(center_id: i32) -> KeteResult<f64> {
    let known = GravParams::known_masses();
    known
        .iter()
        .find(|p| p.naif_id == center_id)
        .map(|p| p.mass)
        .ok_or_else(|| {
            Error::ValueError(format!(
                "Unknown center_id {center_id}: no gravitational parameter available"
            ))
        })
}

/// Specific orbital energy from a state.
///
///   E = v^2/2 - mu/r
///
/// Negative for bound orbits, positive for hyperbolic.
/// Units: AU^2/day^2. Uses the GM of the state's center body.
///
/// # Errors
/// Returns an error if the center body's GM is unknown.
///
pub fn specific_energy<T: InertialFrame>(state: &State<T>) -> KeteResult<f64> {
    let gm = gm_for_center(state.center_id)?;
    let r = state.pos.norm();
    let v2 = state.vel.norm_squared();
    Ok(0.5 * v2 - gm / r)
}

/// B-plane encounter geometry for a hyperbolic flyby.
///
/// When one body approaches another on a hyperbolic trajectory, it follows a
/// curved path that, far from the encounter, approaches two straight lines
/// called asymptotes. The incoming asymptote is the straight-line path the
/// body would follow if there were no gravity.
///
/// The B-plane is the plane that passes through the center of the target body
/// and is perpendicular to this incoming asymptote. Imagine looking down the
/// barrel of the incoming trajectory: the B-plane is the "target" you see,
/// with the target body at the origin.
///
/// The B-vector points from the center of the target body to the point where
/// the incoming asymptote pierces this plane. Its length is the impact
/// parameter, the distance by which the body would miss if there were no
/// gravity. A larger B-vector means a wider miss; a B-vector of zero means
/// a head-on collision.
///
/// The B-plane is split into two axes:
///
/// - T lies along the intersection of the B-plane with the ecliptic plane.
/// - R is perpendicular to T within the B-plane.
///
/// The components B dot T and B dot R locate where the incoming asymptote
/// pierces the B-plane, giving a 2D coordinate for the encounter geometry.
///
/// Attributes
/// ----------
/// b_t :
///     B dot T component (AU).
/// b_r :
///     B dot R component (AU).
/// b_mag :
///     Magnitude of the B vector (AU).
/// theta :
///     B-plane angle (radians).
/// v_inf :
///     Hyperbolic excess speed (AU/day).
/// closest_approach :
///     Periapsis distance (AU).
#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, module = "kete", name = "BPlane", from_py_object, get_all)
)]
pub struct BPlane {
    /// B dot T component (AU).
    pub b_t: f64,

    /// B dot R component (AU).
    pub b_r: f64,

    /// Magnitude of the B vector (AU).
    pub b_mag: f64,

    /// B-plane angle (radians).
    pub theta: f64,

    /// Hyperbolic excess speed (AU/day).
    pub v_inf: f64,

    /// Periapsis distance (AU).
    pub closest_approach: f64,
}

#[cfg(feature = "pyo3")]
#[pyo3::pymethods]
impl BPlane {
    fn __repr__(&self) -> String {
        format!(
            "BPlane(b_t={:.6}, b_r={:.6}, b_mag={:.6}, theta={:.6}, v_inf={:.6}, closest_approach={:.6})",
            self.b_t, self.b_r, self.b_mag, self.theta, self.v_inf, self.closest_approach
        )
    }
}

/// Compute B-plane parameters from a hyperbolic state.
///
/// The input state must be the relative state of the incoming body with
/// respect to the target, centered on the target body. The orbit must be
/// hyperbolic (positive specific energy). The B-plane coordinates are
/// derived from the two-body conic, so the input state should come from
/// a full N-body propagation to capture real perturbations, but the
/// B-plane decomposition itself is an analytical two-body projection.
///
/// # Errors
/// Returns an error if the orbit is bound (energy < 0).
pub fn compute_b_plane<T: InertialFrame>(state: &State<T>) -> KeteResult<BPlane> {
    let gm = gm_for_center(state.center_id)?;
    let pos: Vector3<f64> = state.pos.into();
    let vel: Vector3<f64> = state.vel.into();
    let r = pos.norm();
    let v2 = vel.norm_squared();
    let energy = 0.5 * v2 - gm / r;

    if energy < 0.0 {
        return Err(Error::ValueError(
            "compute_b_plane requires a hyperbolic orbit (energy >= 0)".into(),
        ));
    }

    let v_inf = (2.0 * energy).sqrt();

    // Angular momentum
    let h = pos.cross(&vel);

    // Eccentricity vector
    let e_vec = vel.cross(&h) / gm - pos / r;
    let ecc = e_vec.norm();

    // Semi-major axis (negative for hyperbola)
    let a = -gm / (2.0 * energy);

    // B-plane miss distance and periapsis
    let b_mag = a.abs() * (ecc * ecc - 1.0).sqrt();
    let closest_approach = a.abs() * (ecc - 1.0);

    // Incoming asymptote direction: S = (e_hat * cos(theta_inf) - p_hat * sin(theta_inf))
    // where theta_inf = acos(-1/e) is the true anomaly at infinity.
    // Equivalently, S = v_inf_hat when the body is far away and approaching.
    // For the B-plane we use: S = (e_vec/ecc * cos(theta_inf) + h x e_vec / (h_mag * ecc) * sin(theta_inf))
    let cos_theta_inf = -1.0 / ecc;
    let sin_theta_inf = (1.0 - cos_theta_inf * cos_theta_inf).sqrt();
    let h_mag = h.norm();
    let e_hat = e_vec / ecc;
    let p_hat = h.cross(&e_vec) / (h_mag * ecc);
    // Incoming asymptote (from infinity toward periapsis)
    let s_hat = e_hat * cos_theta_inf + p_hat * sin_theta_inf;

    // Reference direction: ecliptic pole (k-hat)
    let k_hat = Vector3::new(0.0, 0.0, 1.0);

    // T = S x k / |S x k|, R = S x T
    let s_cross_k = s_hat.cross(&k_hat);
    let s_cross_k_mag = s_cross_k.norm();
    let t_hat = if s_cross_k_mag > 1e-12 {
        s_cross_k / s_cross_k_mag
    } else {
        // S is nearly parallel to k; use x-axis as fallback
        let x_hat = Vector3::new(1.0, 0.0, 0.0);
        let fallback = s_hat.cross(&x_hat);
        fallback / fallback.norm()
    };
    let r_hat = s_hat.cross(&t_hat);

    // B vector: perpendicular from center body to the incoming asymptote line.
    // B = b_mag * (h x S) / |h x S|
    let h_cross_s = h.cross(&s_hat);
    let b_vec = b_mag * h_cross_s / h_cross_s.norm();

    let b_t = b_vec.dot(&t_hat);
    let b_r = b_vec.dot(&r_hat);
    let theta = b_r.atan2(b_t);

    Ok(BPlane {
        b_t,
        b_r,
        b_mag,
        theta,
        v_inf,
        closest_approach,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::GMS;
    use crate::frames::{Ecliptic, Vector};
    use crate::prelude::*;
    use crate::time::TDB;

    // Earth GM from DE441 masses.tsv: 3.00348961546514e-06 * GMS
    const EARTH_GM: f64 = 3.003_489_615_465_14e-06 * GMS;
    // Jupiter GM: 2.82534584083387e-07 / 3.00348961546514e-06 ... actually
    // Jupiter mass fraction from masses.tsv is 9.54790662709902e-04
    const JUPITER_GM: f64 = 9.547_906_627_099_02e-04 * GMS;

    #[test]
    fn test_hill_radius_earth() {
        // Earth: a ~ 1.0 AU, e ~ 0.017
        let r_h = hill_radius(1.0, 0.017, EARTH_GM, GMS);
        // Expected ~ 0.01 AU (1.5 million km / 149.6M km/AU)
        assert!(
            (r_h - 0.01).abs() < 0.002,
            "Earth Hill radius: {r_h:.6} AU, expected ~0.01"
        );
    }

    #[test]
    fn test_hill_radius_jupiter() {
        // Jupiter: a ~ 5.2 AU, e ~ 0.048
        let r_h = hill_radius(5.2, 0.048, JUPITER_GM, GMS);
        // Expected ~ 0.35 AU
        assert!(
            (r_h - 0.35).abs() < 0.02,
            "Jupiter Hill radius: {r_h:.6} AU, expected ~0.35"
        );
    }

    #[test]
    fn test_soi_earth() {
        let r_soi = sphere_of_influence(1.0, EARTH_GM, GMS);
        // Expected ~ 0.006 AU
        assert!(
            (r_soi - 0.006).abs() < 0.002,
            "Earth SOI: {r_soi:.6} AU, expected ~0.006"
        );
    }

    #[test]
    fn test_soi_jupiter() {
        let r_soi = sphere_of_influence(5.2, JUPITER_GM, GMS);
        // Expected ~ 0.32 AU
        assert!(
            (r_soi - 0.32).abs() < 0.05,
            "Jupiter SOI: {r_soi:.6} AU, expected ~0.32"
        );
    }

    #[test]
    fn test_tisserand_main_belt() {
        // Main-belt asteroid: a=2.5 AU, e=0.1, i=5 deg
        let i = 5.0_f64.to_radians();
        let a_jup = 5.2;
        let t = tisserand(2.5, 0.1, i, a_jup);
        assert!(
            (t - 3.4).abs() < 0.1,
            "Main-belt Tisserand: {t:.4}, expected ~3.4"
        );
    }

    #[test]
    fn test_tisserand_jfc() {
        // 67P/Churyumov-Gerasimenko: a~3.46, e~0.64, i~7.04 deg
        let i = 7.04_f64.to_radians();
        let t = tisserand(3.46, 0.64, i, 5.2);
        assert!(
            (t - 2.75).abs() < 0.1,
            "67P Tisserand: {t:.4}, expected ~2.75"
        );
    }

    #[test]
    fn test_specific_energy_circular() {
        // Circular orbit at 1 AU: E = -GMS / (2a)
        let v = (GMS / 1.0_f64).sqrt();
        let state = State::<Ecliptic>::new(
            Desig::Empty,
            Time::<TDB>::new(2451545.0),
            Vector::new([1.0, 0.0, 0.0]),
            Vector::new([0.0, v, 0.0]),
            10,
        );
        let e = specific_energy(&state).unwrap();
        let expected = -GMS / 2.0;
        assert!(
            (e - expected).abs() < 1e-10,
            "Circular energy: {e:.10e}, expected: {expected:.10e}"
        );
    }

    #[test]
    fn test_specific_energy_hyperbolic() {
        // Hyperbolic: speed > escape velocity
        let r = 1.0;
        let v_esc = (2.0 * GMS / r).sqrt();
        let v = v_esc * 1.5;
        let state = State::<Ecliptic>::new(
            Desig::Empty,
            Time::<TDB>::new(2451545.0),
            Vector::new([r, 0.0, 0.0]),
            Vector::new([0.0, v, 0.0]),
            10,
        );
        let e = specific_energy(&state).unwrap();
        assert!(e > 0.0, "Hyperbolic energy should be positive: {e}");
    }

    #[test]
    fn test_b_plane_known_flyby() {
        // Create a hyperbolic orbit around a body.
        // Use Earth as center (center_id = 399), arriving from +x direction.
        // v_inf = 0.0002 AU/day (~2.6 km/s), periapsis at ~0.0001 AU (~15000 km).
        let v_inf = 0.0002;
        let r_p = 0.0001; // periapsis distance in AU

        // At periapsis: position = [r_p, 0, 0], velocity = [0, v_peri, 0]
        // v_peri^2 = v_inf^2 + 2*GM/r_p
        let gm_earth = EARTH_GM;
        let v_peri = (v_inf * v_inf + 2.0 * gm_earth / r_p).sqrt();

        let state = State::<Ecliptic>::new(
            Desig::Empty,
            Time::<TDB>::new(2451545.0),
            Vector::new([r_p, 0.0, 0.0]),
            Vector::new([0.0, v_peri, 0.0]),
            399, // Earth-centered
        );

        let bp = compute_b_plane(&state).unwrap();

        // Check v_inf
        assert!(
            (bp.v_inf - v_inf).abs() < 1e-8,
            "v_inf: {:.8}, expected: {v_inf:.8}",
            bp.v_inf
        );

        // Check closest approach matches periapsis distance
        assert!(
            (bp.closest_approach - r_p).abs() / r_p < 1e-6,
            "closest_approach: {:.8}, expected: {r_p:.8}",
            bp.closest_approach
        );

        // b_mag should be > r_p (impact parameter > periapsis for hyperbola)
        assert!(
            bp.b_mag > r_p,
            "b_mag ({:.8}) should exceed periapsis ({r_p:.8})",
            bp.b_mag
        );
    }

    #[test]
    fn test_b_plane_bound_orbit_rejected() {
        // Circular orbit -> bound -> should error
        let v = (GMS / 1.0_f64).sqrt();
        let state = State::<Ecliptic>::new(
            Desig::Empty,
            Time::<TDB>::new(2451545.0),
            Vector::new([1.0, 0.0, 0.0]),
            Vector::new([0.0, v, 0.0]),
            10,
        );
        assert!(compute_b_plane(&state).is_err());
    }
}
