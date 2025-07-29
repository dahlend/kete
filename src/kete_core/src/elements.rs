//! # Orbital Elements
//! This allows conversion to and from cometary orbital elements to [`State`].
//
// BSD 3-Clause License
//
// Copyright (c) 2025, Dar Dahlen
// Copyright (c) 2025, California Institute of Technology
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

use crate::constants::GMS_SQRT;
use crate::frames::Ecliptic;
use crate::prelude::{Desig, KeteResult, State};
use crate::propagation::{PARABOLIC_ECC_LIMIT, compute_eccentric_anomaly, compute_true_anomaly};

use nalgebra::Vector3;
use std::f64::consts::TAU;

/// Cometary Orbital Elements.
///
/// Central mass is assumed to be the Sun, not the solar system barycenter.
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
    pub epoch: f64,

    /// Eccentricity
    pub eccentricity: f64,

    /// Inclination away from the frame of reference in radians
    pub inclination: f64,

    /// Longitude of ascending node in radians
    pub lon_of_ascending: f64,

    /// Time of perihelion passage in JD TDB scaled time
    pub peri_time: f64,

    /// Argument of perihelion in radians
    pub peri_arg: f64,

    /// Perihelion distance in AU
    pub peri_dist: f64,
}

impl CometElements {
    /// Create cometary elements from a state.
    pub fn from_state(state: &State<Ecliptic>) -> Self {
        Self::from_pos_vel(
            state.desig.clone(),
            state.jd,
            &state.pos.into(),
            &state.vel.into(),
        )
    }

    /// Construct Cometary Orbital elements from a position and velocity vector.
    ///
    /// The units of the vectors are AU and AU/Day, with the central mass assumed
    /// to be the Sun.
    ///
    fn from_pos_vel(desig: Desig, epoch: f64, pos: &Vector3<f64>, vel: &Vector3<f64>) -> Self {
        let vel_scaled = vel / GMS_SQRT;
        let v_mag2 = vel_scaled.norm_squared();
        let p_mag = pos.norm();
        let vp_mag = pos.dot(&vel_scaled);

        // Compute the 3 orthogonal vectors which define the orbit.
        let ecc_vec = (v_mag2 - 1.0 / p_mag) * pos - vp_mag * vel_scaled;
        let ang_vec = pos.cross(&vel_scaled);
        let mut lon_asc_vec = Vector3::new(-ang_vec.y, ang_vec.x, 0.0);

        let ecc = ecc_vec.norm();
        let ang_vec_mag = ang_vec.norm();
        let lon_asc_mag = lon_asc_vec.norm();

        let peri_dist = ang_vec_mag.powi(2) / (1.0 + ecc);
        let incl = (ang_vec.z / ang_vec_mag).acos();

        let lon_of_asc: f64 = {
            // if mag near zero, set the longitude to 0
            if lon_asc_mag < 1e-8 {
                lon_asc_vec = Vector3::new(1.0, 0.0, 0.0);
                0.0
            } else {
                (lon_asc_vec.y / lon_asc_mag).atan2(lon_asc_vec.x / lon_asc_mag)
            }
        };

        let peri_arg: f64 = {
            if ecc < 1e-8 {
                0.0
            } else if lon_asc_mag < 1e-8 {
                let mut tmp = f64::atan2(ecc_vec.y, ecc_vec.x);
                if ang_vec.z < 0.0 {
                    tmp = TAU - tmp;
                }
                tmp
            } else {
                let mut tmp = (lon_asc_vec.dot(&ecc_vec) / (ecc * lon_asc_mag))
                    .clamp(-1.0, 1.0)
                    .acos();
                if ecc_vec.z < 0.0 {
                    tmp = TAU - tmp;
                }
                tmp
            }
        };

        let peri_time: f64 = {
            if (ecc - 1.0).abs() < PARABOLIC_ECC_LIMIT {
                // Parabolic
                let mut true_anomaly = ecc_vec.angle(pos);
                if vp_mag.is_sign_negative() {
                    true_anomaly = -true_anomaly;
                }
                let d = (true_anomaly / 2.0).tan();
                let dt = (2_f64.sqrt() * peri_dist.powf(1.5) / GMS_SQRT) * (d + d.powi(3) / 3.0);
                epoch - dt
            } else if ecc < 1e-6 {
                let semi_major = (2.0 / p_mag - v_mag2).recip();
                let mean_motion = semi_major.abs().powf(-1.5) * GMS_SQRT;
                // for circular cases mean_anomaly == true_anomaly
                // for circular orbits, the eccentric vector is 0, so we use the
                // ascending node as the reference for the true anomaly.
                let mut true_anomaly = lon_asc_vec.angle(pos);
                if ang_vec.cross(&lon_asc_vec).dot(pos).is_sign_negative() {
                    true_anomaly = -true_anomaly;
                }
                epoch - true_anomaly / mean_motion
            } else {
                // Hyperbolic or elliptical
                let semi_major = (2.0 / p_mag - v_mag2).recip();
                let mean_motion = semi_major.abs().powf(-1.5) * GMS_SQRT;
                let mean_anomaly: f64 = {
                    let x_bar = (ang_vec_mag.powi(2) - p_mag) / ecc;
                    let y_bar = vp_mag / ecc * ang_vec_mag;
                    let b = semi_major * (1.0 - ecc.powi(2)).abs().sqrt();
                    let s_e = y_bar / b;
                    if ecc < 1.0 {
                        let c_e = x_bar / semi_major + ecc;
                        f64::atan2(s_e, c_e) - ecc * s_e
                    } else {
                        -ecc * s_e - f64::asinh(-s_e)
                    }
                };
                epoch - mean_anomaly / mean_motion
            }
        };

        Self {
            desig,
            epoch,
            eccentricity: ecc,
            inclination: incl,
            lon_of_ascending: lon_of_asc,
            peri_time,
            peri_arg,
            peri_dist,
        }
    }

    /// Convert cometary elements to an [`State`] if possible.
    ///
    /// Center ID is set to 10.
    pub fn try_to_state(&self) -> KeteResult<State<Ecliptic>> {
        let [pos, vel] = self.to_pos_vel()?;
        Ok(State::new(
            self.desig.clone(),
            self.epoch,
            pos.into(),
            vel.into(),
            10,
        ))
    }

    /// Convert orbital elements into a cartesian coordinate position and velocity.
    /// Units are in AU and AU/Day.
    fn to_pos_vel(&self) -> KeteResult<[[f64; 3]; 2]> {
        let elliptical = self.eccentricity < 1.0 - PARABOLIC_ECC_LIMIT;
        let hyperbolic = self.eccentricity > 1.0 + PARABOLIC_ECC_LIMIT;
        let parabolic = !elliptical && !hyperbolic;

        // these handle parabolic in a non-standard way which allows for the
        // eccentric anomaly calculation to be useful later.
        let semi_major = if parabolic {
            0.0
        } else {
            self.peri_dist / (1.0 - self.eccentricity)
        };

        let mean_motion = if parabolic {
            GMS_SQRT
        } else {
            semi_major.abs().powf(-1.5) * GMS_SQRT
        };

        let mean_anom = mean_motion * (self.epoch - self.peri_time);
        let ecc_anom = compute_eccentric_anomaly(self.eccentricity, mean_anom, self.peri_dist)?;

        let x: f64;
        let y: f64;
        let x_dot: f64;
        let y_dot: f64;

        if elliptical {
            let (sin_e, cos_e) = ecc_anom.sin_cos();
            let e_dot = semi_major.powf(1.5) * (1.0 - self.eccentricity * cos_e);
            let b = semi_major * (1.0 - self.eccentricity.powi(2)).sqrt();

            x = semi_major * (cos_e - self.eccentricity);
            y = b * sin_e;
            x_dot = -semi_major / e_dot * sin_e * GMS_SQRT;
            y_dot = b / e_dot * cos_e * GMS_SQRT;
        } else if hyperbolic {
            let sinh_h = ecc_anom.sinh();
            let cosh_h = ecc_anom.cosh();
            let b = -semi_major * (self.eccentricity.powi(2) - 1.0).sqrt();

            let h_dot = semi_major.abs().powf(1.5) * (1.0 - self.eccentricity * cosh_h);

            x = semi_major * (cosh_h - self.eccentricity);
            y = b * sinh_h;
            x_dot = -semi_major / h_dot * sinh_h * GMS_SQRT;
            y_dot = -b / h_dot * cosh_h * GMS_SQRT;
        } else {
            // Parabolic
            let d_dot = self.peri_dist + ecc_anom.powi(2) / 2.0;

            x = self.peri_dist - ecc_anom.powi(2) / 2.0;
            y = (2.0 * self.peri_dist).sqrt() * ecc_anom;
            x_dot = -ecc_anom / d_dot * GMS_SQRT;
            y_dot = (2.0 * self.peri_dist).sqrt() / d_dot * GMS_SQRT;
        }

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

    /// Compute the eccentric anomaly for the cometary elements.
    pub fn eccentric_anomaly(&self) -> KeteResult<f64> {
        compute_eccentric_anomaly(self.eccentricity, self.mean_anomaly(), self.peri_dist).map(|x| {
            match self.eccentricity {
                ecc if ecc > 1.0 - PARABOLIC_ECC_LIMIT => x,
                _ => x.rem_euclid(TAU),
            }
        })
    }

    /// Compute the semi major axis in AU.
    /// NAN is returned if the orbit is parabolic.
    pub fn semi_major(&self) -> f64 {
        match self.eccentricity {
            ecc if ((ecc - 1.0).abs() <= PARABOLIC_ECC_LIMIT) => f64::NAN,
            ecc => self.peri_dist / (1.0 - ecc),
        }
    }

    /// Compute the orbital period in days.
    /// Infinity is returned if the orbit is parabolic or hyperbolic.
    pub fn orbital_period(&self) -> f64 {
        let semi_major = self.semi_major();
        match semi_major {
            a if a <= 1e-8 => f64::INFINITY,
            a => TAU * a.powf(1.5) / GMS_SQRT,
        }
    }

    /// Compute the Aphelion distance in AU.
    pub fn aphelion(&self) -> f64 {
        match self.eccentricity {
            ecc if ((ecc - 1.0).abs() <= PARABOLIC_ECC_LIMIT) => f64::NAN,
            ecc => self.peri_dist * (1.0 + ecc) / (1.0 - ecc),
        }
    }

    /// Compute the mean motion in radians per day.
    pub fn mean_motion(&self) -> f64 {
        match self.eccentricity {
            ecc if ((ecc - 1.0).abs() <= PARABOLIC_ECC_LIMIT) => {
                GMS_SQRT * 1.5 / 2_f64.sqrt() / self.peri_dist.powf(1.5)
            }
            _ => GMS_SQRT / self.semi_major().abs().powf(1.5),
        }
    }

    /// Compute the mean anomaly in radians.
    pub fn mean_anomaly(&self) -> f64 {
        let mm = self.mean_motion();
        let mean_anomaly = (self.epoch - self.peri_time) * mm;
        match self.eccentricity {
            ecc if ecc < 1.0 - PARABOLIC_ECC_LIMIT => mean_anomaly.rem_euclid(TAU),
            _ => mean_anomaly,
        }
    }

    /// Compute the True Anomaly
    /// The angular distance between perihelion and the current position as seen
    /// from the origin.
    pub fn true_anomaly(&self) -> KeteResult<f64> {
        compute_true_anomaly(self.eccentricity, self.mean_anomaly(), self.peri_dist)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_specific_conversion() {
        {
            // This was previously a failed instance.
            let elem = CometElements {
                desig: Desig::Empty,
                epoch: 2461722.5,
                eccentricity: 0.7495474422690582,
                inclination: 0.1582845445910239,
                lon_of_ascending: 1.247985615390004,
                peri_time: 2459273.227910867,
                peri_arg: 4.229481513899533,
                peri_dist: 0.5613867506855604,
            };
            assert!(elem.to_pos_vel().is_ok());
        }
        {
            // This was previously a failed instance.
            let elem = CometElements {
                desig: Desig::Empty,
                epoch: 2455341.243793971,
                eccentricity: 1.001148327267,
                inclination: 2.433767,
                lon_of_ascending: -1.24321,
                peri_time: 2454482.5825015577,
                peri_arg: 0.823935226897,
                peri_dist: 5.594792535298549,
            };
            assert!((elem.true_anomaly().unwrap() - 1.198554792).abs() < 1e-6);
            assert!(elem.to_pos_vel().is_ok());
        }
        {
            let elem = CometElements {
                desig: Desig::Empty,
                epoch: 2455562.5,
                eccentricity: 0.99999,
                inclination: 2.792526803,
                lon_of_ascending: 0.349065850,
                peri_time: 2455369.7,
                peri_arg: -0.8726646259,
                peri_dist: 0.5,
            };
            assert!((elem.true_anomaly().unwrap() - 2.6071638616282553).abs() < 1e-6);
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
                                epoch: 10.0,
                                eccentricity: ecc,
                                inclination: incl,
                                lon_of_ascending: lon_of_asc,
                                peri_time: 10.0,
                                peri_arg,
                                peri_dist,
                            };
                            let [pos, vel] = elem.to_pos_vel().unwrap();
                            assert!(
                                (Vector3::new(pos[0], pos[1], pos[2]).norm() - peri_dist).abs()
                                    < 1e-6
                            );
                            let new_elem = CometElements::from_pos_vel(
                                Desig::Empty,
                                10.0,
                                &pos.into(),
                                &vel.into(),
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
                                        epoch,
                                        eccentricity: ecc,
                                        inclination: incl,
                                        lon_of_ascending: lon_of_asc,
                                        peri_time,
                                        peri_arg,
                                        peri_dist,
                                    };
                                    let [pos, vel] =
                                        elem.to_pos_vel().expect("Failed to convert to state.");
                                    let new_elem = CometElements::from_pos_vel(
                                        Desig::Empty,
                                        epoch,
                                        &pos.into(),
                                        &vel.into(),
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
                                    epoch,
                                    eccentricity: ecc,
                                    inclination: incl,
                                    lon_of_ascending: lon_of_asc,
                                    peri_time,
                                    peri_arg,
                                    peri_dist,
                                };
                                let [pos, vel] = elem.to_pos_vel().unwrap();
                                let new_elem = CometElements::from_pos_vel(
                                    Desig::Empty,
                                    epoch,
                                    &pos.into(),
                                    &vel.into(),
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
