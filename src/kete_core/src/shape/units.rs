//! Body-natural unit system for extended-gravity propagation.
//!
//! When propagating a particle in close proximity to a small body, working
//! in heliocentric AU/day units with `GM_sun` baked in destroys numerical
//! conditioning: the body's `GM` is ~10^-18 in solar units, particle
//! positions are ~10^-7 AU, and orbital periods are tiny fractions of a
//! day.  Switching to a unit system in which `GM_body = 1` and the chosen
//! length scale (typically the body's bounding radius) equals 1 keeps all
//! quantities O(1) and orbital periods O(2*pi).
//!
//! [`BodyUnits`] stores the two scale factors needed to convert between
//! the heliocentric AU/day system and the body-natural system, and
//! provides scaled conversion helpers for position, velocity, time,
//! and acceleration.
//!
// BSD 3-Clause License
//
// Copyright (c) 2026, Dar Dahlen
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

use nalgebra::Vector3;

use crate::errors::{Error, KeteResult};

/// Body-natural unit system.
///
/// In the body-natural system the length scale `L` (typically the body's
/// bounding radius) and the body's `GM` are both unity.  The derived
/// time scale is `T = sqrt(L^3 / GM)`, chosen so that the equation of
/// motion `r_ddot = -GM r / |r|^3` becomes simply `r_ddot = -r / |r|^3`.
///
/// Conversions are between heliocentric AU/day units and these body-
/// natural units.  All scale factors are stored explicitly so the
/// conversions are reversible to round-off.
#[derive(Debug, Clone, Copy)]
pub struct BodyUnits {
    /// Length scale in AU (1 body length = `length_au` AU).
    pub length_au: f64,
    /// Body gravitational parameter in solar units (AU^3 / day^2).
    pub gm_solar: f64,
    /// Time scale in days (1 body time unit = `time_day` days).
    /// Equals `sqrt(length_au^3 / gm_solar)`.
    pub time_day: f64,
}

impl BodyUnits {
    /// Construct a body-natural unit system from a length scale and
    /// gravitational parameter.
    ///
    /// `length_au` is the chosen body length scale in AU (typically the
    /// body's bounding radius).  `gm_solar` is the body's `GM` in solar
    /// units (AU^3 / day^2); for a Polyhedron in solar units this is just
    /// its `gm` field.
    ///
    /// # Errors
    /// Returns [`Error::ValueError`] if either input is non-positive.
    pub fn try_new(length_au: f64, gm_solar: f64) -> KeteResult<Self> {
        if !(length_au > 0.0 && length_au.is_finite()) {
            return Err(Error::ValueError(
                "BodyUnits length_au must be positive and finite".into(),
            ));
        }
        if !(gm_solar > 0.0 && gm_solar.is_finite()) {
            return Err(Error::ValueError(
                "BodyUnits gm_solar must be positive and finite".into(),
            ));
        }
        let time_day = (length_au.powi(3) / gm_solar).sqrt();
        Ok(Self {
            length_au,
            gm_solar,
            time_day,
        })
    }

    /// Convert a position from AU to body-natural length units.
    #[must_use]
    pub fn pos_to_body(&self, r_au: Vector3<f64>) -> Vector3<f64> {
        r_au / self.length_au
    }

    /// Convert a position from body-natural length units back to AU.
    #[must_use]
    pub fn pos_from_body(&self, r_body: Vector3<f64>) -> Vector3<f64> {
        r_body * self.length_au
    }

    /// Convert a velocity from AU/day to body-natural units (`L/T`).
    #[must_use]
    pub fn vel_to_body(&self, v_au_per_day: Vector3<f64>) -> Vector3<f64> {
        v_au_per_day * (self.time_day / self.length_au)
    }

    /// Convert a velocity from body-natural units back to AU/day.
    #[must_use]
    pub fn vel_from_body(&self, v_body: Vector3<f64>) -> Vector3<f64> {
        v_body * (self.length_au / self.time_day)
    }

    /// Convert an acceleration from AU/day^2 to body-natural units
    /// (`L/T^2`).
    #[must_use]
    pub fn accel_to_body(&self, a_au_per_day2: Vector3<f64>) -> Vector3<f64> {
        a_au_per_day2 * (self.time_day * self.time_day / self.length_au)
    }

    /// Convert an acceleration from body-natural units back to AU/day^2.
    #[must_use]
    pub fn accel_from_body(&self, a_body: Vector3<f64>) -> Vector3<f64> {
        a_body * (self.length_au / (self.time_day * self.time_day))
    }

    /// Convert a duration in days to body-natural time units.
    #[must_use]
    pub fn dt_to_body(&self, dt_day: f64) -> f64 {
        dt_day / self.time_day
    }

    /// Convert a duration in body-natural time units to days.
    #[must_use]
    pub fn dt_from_body(&self, dt_body: f64) -> f64 {
        dt_body * self.time_day
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Crude Bennu-like body (~250 m radius, ~5e10 kg).
    fn bennu_units() -> BodyUnits {
        // 250 m -> AU.
        let length_au = 250.0 / 1.495_978_707e11;
        // GM ~ 5.0 m^3/s^2.  Convert to AU^3 / day^2:
        // 1 m^3/s^2 = (1/AU_M)^3 m^3 * (DAY_S^2 / 1) = DAY_S^2 / AU_M^3 in AU^3/day^2.
        let gm_solar = 5.0 * 86_400.0_f64.powi(2) / 1.495_978_707e11_f64.powi(3);
        BodyUnits::try_new(length_au, gm_solar).unwrap()
    }

    #[test]
    fn rejects_non_positive_inputs() {
        assert!(BodyUnits::try_new(0.0, 1.0).is_err());
        assert!(BodyUnits::try_new(-1.0, 1.0).is_err());
        assert!(BodyUnits::try_new(1.0, 0.0).is_err());
        assert!(BodyUnits::try_new(1.0, -1.0).is_err());
        assert!(BodyUnits::try_new(f64::NAN, 1.0).is_err());
        assert!(BodyUnits::try_new(1.0, f64::INFINITY).is_err());
    }

    #[test]
    fn time_scale_matches_definition() {
        let bu = BodyUnits::try_new(2.0, 8.0).unwrap();
        // T = sqrt(L^3 / GM) = sqrt(8/8) = 1.
        assert!((bu.time_day - 1.0).abs() < 1e-15);
    }

    #[test]
    fn position_round_trip() {
        let bu = bennu_units();
        let r_au = Vector3::new(1e-9, -3e-9, 7e-10);
        let back = bu.pos_from_body(bu.pos_to_body(r_au));
        let rel = (back - r_au).norm() / r_au.norm();
        assert!(rel < 1e-15, "position round-trip rel error {rel:e}");
    }

    #[test]
    fn velocity_round_trip() {
        let bu = bennu_units();
        let v = Vector3::new(1.2e-7, 5e-8, -2e-8);
        let back = bu.vel_from_body(bu.vel_to_body(v));
        let rel = (back - v).norm() / v.norm();
        assert!(rel < 1e-15);
    }

    #[test]
    fn accel_round_trip() {
        let bu = bennu_units();
        let a = Vector3::new(3e-12, -1e-12, 4e-13);
        let back = bu.accel_from_body(bu.accel_to_body(a));
        let rel = (back - a).norm() / a.norm();
        assert!(rel < 1e-15);
    }

    #[test]
    fn duration_round_trip() {
        let bu = bennu_units();
        for dt in [1e-6_f64, 1.0, 365.25, 1e6] {
            let back = bu.dt_from_body(bu.dt_to_body(dt));
            let rel = (back - dt).abs() / dt;
            assert!(rel < 1e-15, "dt={dt} rel={rel:e}");
        }
    }

    #[test]
    fn body_units_make_gm_unity() {
        // The defining property: in body-natural units, GM_body = 1.
        // gm has units L^3 / T^2.  Convert gm_solar (AU^3/day^2) to body units:
        //   gm_body = gm_solar * (T_day^2 / L_au^3)
        // and by construction T_day = sqrt(L_au^3 / gm_solar), so this is 1.
        let bu = bennu_units();
        let gm_body = bu.gm_solar * bu.time_day.powi(2) / bu.length_au.powi(3);
        assert!((gm_body - 1.0).abs() < 1e-12, "gm_body = {gm_body}");
    }

    #[test]
    fn keplerian_circular_orbit_period_is_two_pi() {
        // In body-natural units, a circular orbit at r=1 should have period 2*pi.
        // v_circ = sqrt(GM/r) = 1, period T = 2*pi*r/v = 2*pi.
        // Sanity check: convert that period back to days and check it
        // matches the Keplerian formula in solar units T = 2*pi*sqrt(L^3/gm).
        let bu = bennu_units();
        let period_body = 2.0 * std::f64::consts::PI;
        let period_day = bu.dt_from_body(period_body);
        let expected = 2.0 * std::f64::consts::PI * bu.time_day;
        let rel = (period_day - expected).abs() / expected;
        assert!(rel < 1e-15);
    }
}
