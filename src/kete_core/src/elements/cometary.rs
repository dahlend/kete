// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-FileCopyrightText: 2025 California Institute of Technology
// SPDX-License-Identifier: BSD-3-Clause

//! # Cometary Orbital Elements
//!
//! Conversion to and from cometary orbital elements and [`State`].
//!
//! The universal Kepler solver carries the two-body state at perihelion to the
//! epoch. The time of perihelion comes from the closed form inverse of the same
//! formulation. These formulas apply from a circular orbit through the
//! parabolic limit. For a circular orbit perihelion is at the ascending node.

use super::gm_sqrt_for_center;
use crate::frames::{CenterBody, DynCenter, Ecliptic};
use crate::kepler::{analytic_2_body_delta, compute_eccentric_anomaly, universal_g};
use crate::prelude::{Desig, KeteResult, State};
use crate::time::{TDB, Time};

use nalgebra::Vector3;
use std::f64::consts::TAU;

/// Cometary Orbital Elements.
///
/// Units are:
/// - Radians
/// - AU
/// - Time is in Days
#[derive(Debug, Clone)]
pub struct CometElements {
    /// Designation of the object
    pub desig: Desig,

    /// Epoch of fit
    pub epoch: Time<TDB>,

    /// Eccentricity
    pub eccentricity: f64,

    /// Inclination away from the frame of reference in radians
    pub inclination: f64,

    /// Longitude of ascending node in radians
    pub lon_of_ascending: f64,

    /// Time of perihelion passage in JD TDB scaled time
    pub peri_time: Time<TDB>,

    /// Argument of perihelion in radians
    pub peri_arg: f64,

    /// Perihelion distance in AU
    pub peri_dist: f64,

    /// NAIF ID of the central body (default: 10 for the Sun)
    pub center_id: i32,

    /// Square root of the gravitational parameter of the central body.
    /// Units: AU^(3/2) / Day
    pub gm_sqrt: f64,
}

impl CometElements {
    /// Create cometary elements from a state.
    ///
    /// # Errors
    /// Fails if the state's center has no known mass. Elements are defined about a
    /// gravitating body, and which body it is sets `mu`; there is no sensible default.
    pub fn from_state<C: CenterBody>(state: &State<Ecliptic, C>) -> KeteResult<Self>
    where
        DynCenter: From<C>,
    {
        let gm_sqrt = gm_sqrt_for_center(state.center_id())?;
        Ok(Self::from_pos_vel(
            state.desig.clone(),
            state.epoch,
            &state.pos.into(),
            &state.vel.into(),
            state.center_id(),
            gm_sqrt,
        ))
    }

    /// Convert cometary elements to an [`State`] if possible.
    ///
    /// # Errors
    /// Returns [`Error::Convergence`](crate::errors::Error::Convergence) if the
    /// elements are not finite, the perihelion distance is not positive, or the
    /// Kepler solver does not converge.
    pub fn try_to_state(&self) -> KeteResult<State<Ecliptic>> {
        let [pos, vel] = self.to_pos_vel()?;
        Ok(State::new(
            self.desig.clone(),
            self.epoch,
            pos,
            vel,
            self.center_id,
        ))
    }

    /// Compute the eccentric anomaly for the cometary elements.
    ///
    /// For an elliptical orbit this is the eccentric anomaly `E` in
    /// `[0, 2 pi)`. For a hyperbolic orbit it is the hyperbolic anomaly `H`.
    /// For a parabolic orbit it is zero, which is the limit of both as the
    /// eccentricity approaches one. This solves Kepler's equation at the
    /// [`Self::mean_anomaly`].
    ///
    /// # Errors
    /// Returns [`Error::ValueError`](crate::errors::Error::ValueError) if the
    /// eccentricity or the mean anomaly is not finite, or the eccentricity is
    /// negative. Returns
    /// [`Error::Convergence`](crate::errors::Error::Convergence) if the
    /// iteration does not converge.
    pub fn eccentric_anomaly(&self) -> KeteResult<f64> {
        compute_eccentric_anomaly(self.eccentricity, self.mean_anomaly())
    }

    /// Compute the semi major axis in AU.
    ///
    /// The value is negative for a hyperbolic orbit and infinite for a
    /// parabolic orbit.
    #[must_use]
    pub fn semi_major(&self) -> f64 {
        self.peri_dist / (1.0 - self.eccentricity)
    }

    /// Compute the orbital period in days.
    ///
    /// Infinity is returned if the orbit is not bound.
    #[must_use]
    pub fn orbital_period(&self) -> f64 {
        if self.eccentricity >= 1.0 {
            return f64::INFINITY;
        }
        TAU / self.mean_motion()
    }

    /// Compute the aphelion distance in AU.
    ///
    /// Infinity is returned if the orbit is not bound.
    #[must_use]
    pub fn aphelion(&self) -> f64 {
        if self.eccentricity >= 1.0 {
            return f64::INFINITY;
        }
        self.peri_dist * (1.0 + self.eccentricity) / (1.0 - self.eccentricity)
    }

    /// Compute the mean motion in radians per day, `sqrt(GM / |a|^3)`.
    ///
    /// The value is zero for a parabolic orbit. This is its limit as the
    /// eccentricity approaches one.
    #[must_use]
    pub fn mean_motion(&self) -> f64 {
        self.gm_sqrt * ((1.0 - self.eccentricity).abs() / self.peri_dist).powf(1.5)
    }

    /// Compute the mean anomaly in radians.
    ///
    /// The value is in `[0, 2 pi)` for an elliptical orbit. An open orbit has
    /// no period, so the value is not reduced. The value is zero for a
    /// parabolic orbit, see [`Self::mean_motion`].
    #[must_use]
    pub fn mean_anomaly(&self) -> f64 {
        let mean_anomaly = (self.epoch - self.peri_time).elapsed * self.mean_motion();
        if self.eccentricity < 1.0 {
            mean_anomaly.rem_euclid(TAU)
        } else {
            mean_anomaly
        }
    }

    /// Compute the true anomaly in radians.
    ///
    /// This is the angle from perihelion to the current position as seen from
    /// the origin, in `[0, 2 pi)`.
    ///
    /// # Errors
    /// Returns [`Error::Convergence`](crate::errors::Error::Convergence) if the
    /// elements are not finite, the perihelion distance is not positive, or the
    /// Kepler solver does not converge.
    pub fn true_anomaly(&self) -> KeteResult<f64> {
        let (pos, _) = self.perifocal()?;
        Ok(pos.y.atan2(pos.x).rem_euclid(TAU))
    }

    /// Construct Cometary Orbital elements from a position and velocity vector.
    ///
    /// The units of the vectors are AU and AU/Day.
    pub(super) fn from_pos_vel(
        desig: Desig,
        epoch: Time<TDB>,
        pos: &Vector3<f64>,
        vel: &Vector3<f64>,
        center_id: i32,
        gm_sqrt: f64,
    ) -> Self {
        let vel_scaled = vel / gm_sqrt;
        let v_mag2 = vel_scaled.norm_squared();
        let p_mag = pos.norm();
        let vp_mag = pos.dot(&vel_scaled);

        // Compute the 3 orthogonal vectors which define the orbit.
        let ecc_vec = (v_mag2 - 1.0 / p_mag) * pos - vp_mag * vel_scaled;
        let ang_vec = pos.cross(&vel_scaled);
        let mut lon_asc_vec = Vector3::new(-ang_vec.y, ang_vec.x, 0.0);

        let ecc = ecc_vec.norm();
        let ang_vec_mag = ang_vec.norm();

        let peri_dist = ang_vec_mag.powi(2) / (1.0 + ecc);
        let incl = (ang_vec.x * ang_vec.x + ang_vec.y * ang_vec.y)
            .sqrt()
            .atan2(ang_vec.z);

        // For a nearly equatorial orbit the node direction is mostly rounding.
        // The argument of perihelion is measured from this same direction, so
        // the state is unaffected. An exactly equatorial orbit has its node on
        // the x axis.
        if lon_asc_vec == Vector3::zeros() {
            lon_asc_vec = Vector3::new(1.0, 0.0, 0.0);
        }
        let lon_of_asc = lon_asc_vec.y.atan2(lon_asc_vec.x);

        // The angle from the node to the eccentricity vector, about the angular
        // momentum. It is zero for a zero eccentricity vector.
        let sin_w = lon_asc_vec.cross(&ecc_vec).dot(&ang_vec) / ang_vec_mag;
        let peri_arg = sin_w.atan2(lon_asc_vec.dot(&ecc_vec)).rem_euclid(TAU);

        // The position in the perifocal frame. Its x axis is the eccentricity
        // vector, the direction the argument of perihelion was measured to, so
        // the anomaly agrees with that angle even where the direction is mostly
        // rounding. A zero eccentricity vector puts perihelion at the node.
        let p_axis = if ecc > 0.0 {
            ecc_vec / ecc
        } else {
            lon_asc_vec.normalize()
        };
        let q_axis = (ang_vec / ang_vec_mag).cross(&p_axis);
        let (x, y) = (pos.dot(&p_axis), pos.dot(&q_axis));
        let mu = gm_sqrt * gm_sqrt;
        let beta = mu * (2.0 / p_mag - v_mag2);
        let peri_time = epoch - time_since_perihelion(x, y, ang_vec_mag, ecc, peri_dist, beta, mu);

        Self {
            desig,
            epoch,
            eccentricity: ecc,
            inclination: incl,
            lon_of_ascending: lon_of_asc,
            peri_time,
            peri_arg,
            peri_dist,
            center_id,
            gm_sqrt,
        }
    }

    /// Convert orbital elements into a cartesian coordinate position and velocity.
    /// Units are in AU and AU/Day.
    pub(super) fn to_pos_vel(&self) -> KeteResult<[[f64; 3]; 2]> {
        let (perifocal_pos, perifocal_vel) = self.perifocal()?;
        let (x, y) = (perifocal_pos.x, perifocal_pos.y);
        let (x_dot, y_dot) = (perifocal_vel.x, perifocal_vel.y);

        let (s_w, c_w) = self.peri_arg.sin_cos();
        let (s_o, c_o) = self.lon_of_ascending.sin_cos();
        let (s_i, c_i) = self.inclination.sin_cos();

        let px = c_w * c_o - s_w * s_o * c_i;
        let py = c_w * s_o + s_w * c_o * c_i;
        let pz = s_w * s_i;
        let qx = -s_w * c_o - c_w * s_o * c_i;
        let qy = -s_w * s_o + c_w * c_o * c_i;
        let qz = c_w * s_i;

        let pos = [x * px + y * qx, x * py + y * qy, x * pz + y * qz];
        let vel = [
            x_dot * px + y_dot * qx,
            x_dot * py + y_dot * qy,
            x_dot * pz + y_dot * qz,
        ];

        Ok([pos, vel])
    }

    /// Compute the perifocal position and velocity in AU and AU/Day.
    ///
    /// The x axis points toward perihelion. The y axis is the direction of
    /// motion at perihelion. The state at perihelion is position `(q, 0, 0)`
    /// and velocity `(0, sqrt(GM (1 + e) / q), 0)`. The universal Kepler solver
    /// carries it to the epoch.
    ///
    /// # Errors
    /// Returns [`Error::Convergence`](crate::errors::Error::Convergence) if the
    /// elements are not finite, the perihelion distance is not positive, or the
    /// Kepler solver does not converge.
    fn perifocal(&self) -> KeteResult<(Vector3<f64>, Vector3<f64>)> {
        let mu = self.gm_sqrt * self.gm_sqrt;
        let peri_pos = Vector3::new(self.peri_dist, 0.0, 0.0);
        let peri_vel = Vector3::new(
            0.0,
            (mu * (1.0 + self.eccentricity) / self.peri_dist).sqrt(),
            0.0,
        );
        let (d_pos, d_vel) = analytic_2_body_delta(
            (self.epoch - self.peri_time).elapsed,
            &peri_pos,
            &peri_vel,
            mu,
        )?;
        Ok((peri_pos + d_pos, peri_vel + d_vel))
    }
}

/// Compute the time since perihelion in days of a two-body orbit.
///
/// `x` and `y` are the position in the perifocal frame, in AU. `h_scaled` is
/// the specific angular momentum divided by `sqrt(mu)`, so its square is the
/// semi-latus rectum `p`. `ecc` and `peri_dist` are the eccentricity and the
/// perihelion distance. `beta = 2 mu / r - v^2`, and `mu` is the gravitational
/// parameter.
///
/// From perihelion at universal anomaly `s`, `G1(s) = y / sqrt(mu p)` and
/// `G2(s) = (r - x) / (mu (1 + e))`, see [`universal_g`]. Neither divides by
/// the eccentricity. Let `z = sqrt(|beta|) s`. On an elliptical orbit
/// `sin(z) = sqrt(beta) G1` and `cos(z) = G0 = 1 - beta G2`. On a hyperbolic
/// orbit `sinh(z) = sqrt(-beta) G1`. These give `s` in closed form, and
/// `s = G1` when `beta = 0`. The time is `q G1(s) + mu G3(s)`.
fn time_since_perihelion(
    x: f64,
    y: f64,
    h_scaled: f64,
    ecc: f64,
    peri_dist: f64,
    beta: f64,
    mu: f64,
) -> f64 {
    let r = (x * x + y * y).sqrt();
    // r - x cancels near perihelion, where it equals y^2 / (r + x).
    let r_minus_x = if x > 0.0 { y * y / (r + x) } else { r - x };
    let g1 = y / (mu.sqrt() * h_scaled);
    let g2 = r_minus_x / (mu * (1.0 + ecc));
    let b_sqrt = beta.abs().sqrt();
    let anomaly = if beta > 0.0 {
        (b_sqrt * g1).atan2(1.0 - beta * g2) / b_sqrt
    } else if beta < 0.0 {
        (b_sqrt * g1).asinh() / b_sqrt
    } else {
        g1
    };
    let [_, g1, _, g3] = universal_g(anomaly, beta);
    peri_dist * g1 + mu * g3
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::GMS_SQRT;

    #[test]
    fn test_specific_conversion() {
        {
            // A case that exercises the branch selection in the conversion.
            let elem = CometElements {
                desig: Desig::Empty,
                epoch: 2461722.5.into(),
                eccentricity: 0.7495474422690582,
                inclination: 0.1582845445910239,
                lon_of_ascending: 1.247985615390004,
                peri_time: 2459273.227910867.into(),
                peri_arg: 4.229481513899533,
                peri_dist: 0.5613867506855604,
                center_id: 10,
                gm_sqrt: GMS_SQRT,
            };
            assert!(elem.to_pos_vel().is_ok());
        }
        {
            // A case that exercises the branch selection in the conversion.
            let elem = CometElements {
                desig: Desig::Empty,
                epoch: 2455341.243793971.into(),
                eccentricity: 1.001148327267,
                inclination: 2.433767,
                lon_of_ascending: -1.24321,
                peri_time: 2454482.5825015577.into(),
                peri_arg: 0.823935226897,
                peri_dist: 5.594792535298549,
                center_id: 10,
                gm_sqrt: GMS_SQRT,
            };
            assert!((elem.true_anomaly().unwrap() - 1.198554792).abs() < 1e-6);
            assert!(elem.to_pos_vel().is_ok());
        }
        {
            // A parabolic orbit, checked against Barker's equation. With
            // `n = sqrt(GM / (2 q^3))` and `D = tan(nu / 2)`,
            // `n (t - T) = D + D^3 / 3`.
            let elem = CometElements {
                desig: Desig::Empty,
                epoch: 2455562.5.into(),
                eccentricity: 1.0,
                inclination: 2.792526803,
                lon_of_ascending: 0.349065850,
                peri_time: 2455369.7.into(),
                peri_arg: -0.8726646259,
                peri_dist: 0.5,
                center_id: 10,
                gm_sqrt: GMS_SQRT,
            };
            let barker = {
                let dt = (elem.epoch - elem.peri_time).elapsed;
                let rate = GMS_SQRT / (2.0 * elem.peri_dist.powi(3)).sqrt();
                let target = rate * dt;
                let mut lo = 0.0_f64;
                let mut hi = 1e3_f64;
                for _ in 0..200 {
                    let mid = f64::midpoint(lo, hi);
                    if mid + mid.powi(3) / 3.0 < target {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                2.0 * lo.atan()
            };
            assert!(
                (elem.true_anomaly().unwrap() - barker).abs() < 1e-9,
                "{} vs Barker {barker}",
                elem.true_anomaly().unwrap()
            );
        }
    }

    /// Check that the true anomaly and the position lie on the same conic.
    ///
    /// `to_pos_vel` and `true_anomaly` both use the perifocal state. The test
    /// checks the distance against the conic equation at the true anomaly.
    #[test]
    fn test_true_anomaly_agrees_with_position() {
        for ecc in [0.0, 0.5, 0.9999, 1.0, 1.0001, 1.5, 3.0] {
            for peri_dist in [0.3, 1.0, 5.0] {
                let elem = CometElements {
                    desig: Desig::Empty,
                    epoch: 2455562.5.into(),
                    eccentricity: ecc,
                    inclination: 0.4,
                    lon_of_ascending: 1.1,
                    peri_time: 2455369.7.into(),
                    peri_arg: 2.3,
                    peri_dist,
                    center_id: 10,
                    gm_sqrt: GMS_SQRT,
                };
                let [pos, _] = elem.to_pos_vel().unwrap();
                let radius = Vector3::new(pos[0], pos[1], pos[2]).norm();

                // The conic equation, evaluated at the reported true anomaly.
                let nu = elem.true_anomaly().unwrap();
                let semi_latus = peri_dist * (1.0 + ecc);
                let expected = semi_latus / (1.0 + ecc * nu.cos());

                assert!(
                    (radius - expected).abs() / radius < 1e-8,
                    "e={ecc} q={peri_dist}: position gives r={radius}, \
                     true anomaly {nu} gives r={expected}",
                );
            }
        }
    }

    /// Convert elements to a state and back near the parabolic limit.
    ///
    /// The eccentricities are below, at and above one. The epochs are before
    /// and after perihelion. The time of perihelion and the state must match.
    #[test]
    fn test_elements_roundtrip_near_parabolic() {
        for de in [
            -1e-2, -1e-4, -1e-6, -1e-8, -1e-10, 0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2,
        ] {
            for peri_dist in [0.01, 1.0, 30.0] {
                for dt in [-3650.0, -30.0, -1.0, 0.0, 1.0, 30.0, 3650.0] {
                    let elem = CometElements {
                        desig: Desig::Empty,
                        epoch: 2460000.5.into(),
                        eccentricity: 1.0 + de,
                        inclination: 0.7,
                        lon_of_ascending: 2.1,
                        peri_time: (2460000.5 - dt).into(),
                        peri_arg: -0.4,
                        peri_dist,
                        center_id: 10,
                        gm_sqrt: GMS_SQRT,
                    };
                    let [pos, vel] = elem
                        .to_pos_vel()
                        .unwrap_or_else(|e| panic!("{de:e} {peri_dist} {dt}: {e}"));
                    let new_elem = CometElements::from_pos_vel(
                        Desig::Empty,
                        elem.epoch,
                        &pos.into(),
                        &vel.into(),
                        10,
                        GMS_SQRT,
                    );
                    // A bound orbit passes perihelion once per period, so the
                    // test compares the time of perihelion modulo the period.
                    let mut peri_time_err = (new_elem.peri_time - elem.peri_time).elapsed;
                    let period = elem.orbital_period();
                    if period.is_finite() {
                        peri_time_err -= period * (peri_time_err / period).round();
                    }
                    let peri_time_err = peri_time_err.abs();
                    assert!(
                        peri_time_err < 1e-9 * dt.abs().max(1.0),
                        "e - 1 = {de:e}, q = {peri_dist}, dt = {dt}: peri time off by {peri_time_err:e} days"
                    );
                    let [new_pos, _] = new_elem.to_pos_vel().unwrap();
                    let (pos, new_pos) = (Vector3::from(pos), Vector3::from(new_pos));
                    let rel = (new_pos - pos).norm() / pos.norm();
                    assert!(
                        rel < 1e-12,
                        "e - 1 = {de:e}, q = {peri_dist}, dt = {dt}: position off by {rel:e}"
                    );
                }
            }
        }
    }

    /// Check that the state is continuous in the eccentricity.
    ///
    /// Each pair of eccentricities is 2e-13 apart. The pairs are at several
    /// distances below and above one. Ten years after perihelion the two
    /// positions must agree to within 1e-10 AU.
    #[test]
    fn test_state_continuous_in_eccentricity() {
        let at = |ecc: f64| {
            let elem = CometElements {
                desig: Desig::Empty,
                epoch: 2460000.5.into(),
                eccentricity: ecc,
                inclination: 0.7,
                lon_of_ascending: 2.1,
                peri_time: (2460000.5 - 3650.0).into(),
                peri_arg: -0.4,
                peri_dist: 1.0,
                center_id: 10,
                gm_sqrt: GMS_SQRT,
            };
            Vector3::from(elem.to_pos_vel().unwrap()[0])
        };
        for edge in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 0.0] {
            for sign in [-1.0, 1.0] {
                let below = at(1.0 + sign * edge - 1e-13);
                let above = at(1.0 + sign * edge + 1e-13);
                let jump = (above - below).norm();
                assert!(
                    jump < 1e-10,
                    "e - 1 = {:e}: position jumped by {jump:e} AU",
                    sign * edge
                );
            }
        }
    }
    /// Convert elements to a state and back for nearly circular orbits.
    ///
    /// The eccentricities run from zero to 1e-3. The planes include exactly and
    /// nearly equatorial ones, prograde and retrograde. The direction of a tiny
    /// eccentricity vector or node vector is mostly rounding, so the test
    /// compares states, not elements.
    #[test]
    fn test_state_roundtrip_near_circular() {
        for ecc in [0.0, 1e-12, 1e-10, 5e-9, 1e-8, 2e-8, 1e-7, 1e-6, 1e-5, 1e-3] {
            for peri_arg in [0.0, 1.0, 2.5, 3.0, 5.0] {
                for incl in [
                    0.0,
                    1e-9,
                    0.3,
                    std::f64::consts::PI - 1e-9,
                    std::f64::consts::PI,
                ] {
                    for days in [0.0, 70.0, 200.0, 333.0] {
                        let elem = CometElements {
                            desig: Desig::Empty,
                            epoch: 2460000.5.into(),
                            eccentricity: ecc,
                            inclination: incl,
                            lon_of_ascending: 2.1,
                            peri_time: (2460000.5 - days).into(),
                            peri_arg,
                            peri_dist: 1.0,
                            center_id: 10,
                            gm_sqrt: GMS_SQRT,
                        };
                        let [pos, vel] = elem.to_pos_vel().unwrap();
                        let new_elem = CometElements::from_pos_vel(
                            Desig::Empty,
                            elem.epoch,
                            &pos.into(),
                            &vel.into(),
                            10,
                            GMS_SQRT,
                        );
                        let [new_pos, new_vel] = new_elem.to_pos_vel().unwrap();
                        let pos_err = (Vector3::from(new_pos) - Vector3::from(pos)).norm();
                        let vel_err = (Vector3::from(new_vel) - Vector3::from(vel)).norm()
                            / Vector3::from(vel).norm();
                        assert!(
                            pos_err < 1e-11 && vel_err < 1e-11,
                            "e = {ecc}, w = {peri_arg}, i = {incl}, {days} days: \
                             position off by {pos_err:e} AU, velocity by {vel_err:e}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_elements_perihelion() {
        for ecc in [0.0, 0.1, 0.5, 1.0, 2.0] {
            for incl in [-2.0, 0.0, 2.0, 3.0] {
                for lon_of_asc in [-0.5, 0.0, 4.0] {
                    for peri_arg in [-2.0, 0.0, 0.1, 0.5, 10.0] {
                        for peri_dist in [0.1, 0.5, 10.0] {
                            let elem = CometElements {
                                desig: Desig::Empty,
                                epoch: 10.0.into(),
                                eccentricity: ecc,
                                inclination: incl,
                                lon_of_ascending: lon_of_asc,
                                peri_time: 10.0.into(),
                                peri_arg,
                                peri_dist,
                                center_id: 10,
                                gm_sqrt: GMS_SQRT,
                            };
                            let [pos, vel] = elem.to_pos_vel().unwrap();
                            assert!(
                                (Vector3::new(pos[0], pos[1], pos[2]).norm() - peri_dist).abs()
                                    < 1e-6
                            );
                            let new_elem = CometElements::from_pos_vel(
                                Desig::Empty,
                                10.0.into(),
                                &pos.into(),
                                &vel.into(),
                                10,
                                GMS_SQRT,
                            );
                            assert!((peri_dist - new_elem.peri_dist).abs() < 1e-8);
                            assert!((ecc - new_elem.eccentricity).abs() < 1e-8);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_elements_roundtrip() {
        for ecc in [0.001, 0.1, 0.5, 1.0, 2.0] {
            for epoch in [-10.0, 0.0, 10.0] {
                for incl in [-2.0, 0.1, 0.0, 2.0] {
                    for lon_of_asc in [-0.5, 0.0, 4.0] {
                        for peri_time in [-100., 0.0, 100.0] {
                            for peri_arg in [-1.0, 0.0, 1.0] {
                                for peri_dist in [0.3, 0.5] {
                                    let elem = CometElements {
                                        desig: Desig::Empty,
                                        epoch: epoch.into(),
                                        eccentricity: ecc,
                                        inclination: incl,
                                        lon_of_ascending: lon_of_asc,
                                        peri_time: peri_time.into(),
                                        peri_arg,
                                        peri_dist,
                                        center_id: 10,
                                        gm_sqrt: GMS_SQRT,
                                    };
                                    let [pos, vel] =
                                        elem.to_pos_vel().expect("Failed to convert to state.");
                                    let new_elem = CometElements::from_pos_vel(
                                        Desig::Empty,
                                        epoch.into(),
                                        &pos.into(),
                                        &vel.into(),
                                        10,
                                        GMS_SQRT,
                                    );
                                    let [new_pos, new_vel] =
                                        new_elem.to_pos_vel().expect("Failed to convert to state.");

                                    for idx in 0..3 {
                                        assert!(
                                            (new_pos[idx] - pos[idx]).abs() < 1e-7,
                                            "\n{elem:?}\n{new_elem:?}\n {pos:?}\n {new_pos:?}\n {vel:?}\n {new_vel:?}",
                                        );
                                        assert!((new_vel[idx] - vel[idx]).abs() < 1e-7);
                                    }

                                    let t_anom = ((elem.true_anomaly().unwrap()
                                        - new_elem.true_anomaly().unwrap())
                                        * 2.0)
                                        .sin()
                                        .abs();

                                    let t_ecc = ((elem.eccentric_anomaly().unwrap()
                                        - new_elem.eccentric_anomaly().unwrap())
                                        * 2.0)
                                        .sin()
                                        .abs();

                                    assert!(t_anom < 1e-6);
                                    assert!(t_ecc < 1e-6);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_elements_roundtrip_circular() {
        let ecc = 0.0;
        for incl in [-1.0, 0.1, 0.0, 1.0] {
            for epoch in [-10.0, 0.0, 10.0] {
                for lon_of_asc in [0.0, 1.5] {
                    for peri_time in [-100., 0.0, 100.0] {
                        for peri_arg in [0.0, 1.0] {
                            for peri_dist in [0.3, 0.5] {
                                let elem = CometElements {
                                    desig: Desig::Empty,
                                    epoch: epoch.into(),
                                    eccentricity: ecc,
                                    inclination: incl,
                                    lon_of_ascending: lon_of_asc,
                                    peri_time: peri_time.into(),
                                    peri_arg,
                                    peri_dist,
                                    center_id: 10,
                                    gm_sqrt: GMS_SQRT,
                                };
                                let [pos, vel] = elem.to_pos_vel().unwrap();
                                let new_elem = CometElements::from_pos_vel(
                                    Desig::Empty,
                                    epoch.into(),
                                    &pos.into(),
                                    &vel.into(),
                                    10,
                                    GMS_SQRT,
                                );

                                let [new_pos, new_vel] = new_elem.to_pos_vel().unwrap();
                                for idx in 0..3 {
                                    assert!(
                                        (new_pos[idx] - pos[idx]).abs() < 1e-7,
                                        "\n{elem:?}\n{new_elem:?}\n{pos:?}\n {new_pos:?}\n {vel:?}\n {new_vel:?}",
                                    );
                                    assert!(
                                        (new_vel[idx] - vel[idx]).abs() < 1e-7,
                                        "\n{elem:?}\n{new_elem:?}\n{pos:?}\n {new_pos:?}\n {vel:?}\n {new_vel:?}",
                                    );
                                    assert!(
                                        (elem.true_anomaly().unwrap() - elem.mean_anomaly()).abs()
                                            < 1e-7,
                                    );
                                    assert!(
                                        (elem.eccentric_anomaly().unwrap() - elem.mean_anomaly())
                                            .abs()
                                            < 1e-7
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
