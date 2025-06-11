use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

/// Reflected light properties of a comet using the MK magnitude system.
///
/// <https://en.wikipedia.org/wiki/Absolute_magnitude#Cometary_magnitudes>
///
/// The model for apparent magnitudes are:
///
/// `m1 + k1 * log10(sun2obj.r) + 5.0 * log10(obj2obs.r) + phase_mag_slope_1 * phase`
/// `m2 + k2 * log10(sun2obj.r) + 5.0 * log10(obj2obs.r) + phase_mag_slope_2 * phase`
///
/// Where m1/k1 are related to total magnitudes and m2/k2 are nucleus magnitudes.
///
/// This model is based off of these:
/// <https://ssd.jpl.nasa.gov/horizons/manual.html#obsquan>  (see section 9)
/// <https://en.wikipedia.org/wiki/Absolute_magnitude#Cometary_magnitudes>
///
/// Note that the above model does not include a 2.5x term attached to the K1/2 terms
/// which are present in the wikipedia definition, this matches the definitions used by
/// JPL Horizons.
///
/// This model additionally includes a correction for phase effects.
///
#[derive(Debug, Deserialize, Serialize)]
pub struct CometMKParams {
    /// Designation (name) of the object.
    pub desig: String,

    /// M1 and K1 if defined.
    pub mk_1: Option<[f64; 2]>,

    /// M2 and K2 if defined.
    pub mk_2: Option<[f64; 2]>,

    /// Phase correction coefficients in units of Mag/Deg
    pub phase_corr_coef: [f64; 2],
}

impl CometMKParams {
    /// Create a new [`CometMKParams`] object.
    pub fn new(
        desig: String,
        mk_1: Option<[f64; 2]>,
        mk_2: Option<[f64; 2]>,
        phase_corr_coef: [f64; 2],
    ) -> Self {
        Self {
            desig,
            mk_1,
            mk_2,
            phase_corr_coef,
        }
    }

    /// Compute the apparent total flux including both coma and nucleus of the comet.
    /// This includes an additional 0.035 Mag/Deg phase correction.
    pub fn apparent_total_mag(
        &self,
        sun2obs: &Vector3<f64>,
        sun2obj: &Vector3<f64>,
    ) -> Option<f64> {
        let [m1, k1] = self.mk_1?;
        let obj2obs = -sun2obj + sun2obs;
        let obs_dist = obj2obs.norm();
        let helio_dist = sun2obj.norm();
        let phase_corr = self.phase_corr_coef[0] * obj2obs.angle(&-sun2obj).to_degrees();
        Some(m1 + k1 * helio_dist.log10() + 5.0 * obs_dist.log10() + phase_corr)
    }

    /// Compute the apparent nuclear flux of the comet, not including the coma.
    /// This includes an additional 0.035 Mag/Deg phase correction.
    pub fn apparent_nuclear_mag(
        &self,
        sun2obs: &Vector3<f64>,
        sun2obj: &Vector3<f64>,
    ) -> Option<f64> {
        let [m2, k2] = self.mk_2?;
        let obj2obs = -sun2obj + sun2obs;
        let obs_dist = obj2obs.norm();
        let helio_dist = sun2obj.norm();
        let phase_corr = self.phase_corr_coef[1] * obj2obs.angle(&-sun2obj).to_degrees();
        Some(m2 + k2 * helio_dist.log10() + 5.0 * obs_dist.log10() + phase_corr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comet_mags() {
        // Testing 12P against JPL horizons values.

        let mk_1 = [5.0, 15.0];
        let mk_2 = [11.0, 10.0];

        let pos = Vector3::from([-1.154216937776, -2.385684461737, -1.893975509337]);
        let obs = Vector3::from([0.469511752038, 0.868580775506, -5.2896978e-5]);

        let comet_mag = CometMKParams::new("12P".into(), Some(mk_1), Some(mk_2), [0.0, 0.0]);

        let nucl_mags = comet_mag.apparent_nuclear_mag(&obs, &pos).unwrap();
        let total_mags = comet_mag.apparent_total_mag(&obs, &pos).unwrap();

        // Horizons values: 15.757  19.192
        assert!((total_mags - 15.757).abs() < 1e-3);
        assert!((nucl_mags - 19.192).abs() < 1e-3);
    }
}
