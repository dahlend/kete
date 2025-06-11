//! Observatory codes used by the MPC
use lazy_static::lazy_static;
use nalgebra::{Rotation3, UnitVector3, Vector3};
use serde::Deserialize;

use crate::desigs::Desig;
use crate::frames::{EARTH_A, ecef_to_geodetic_lat_lon};
use crate::prelude::{Error, KeteResult};
use core::f64;
use std::str;
use std::str::FromStr;

/// Observatory information
#[derive(Debug, Deserialize)]
pub struct ObsCode {
    /// observatory code
    pub code: Desig,

    /// longitude in degrees
    pub lon: f64,

    /// latitude in degrees
    pub lat: f64,

    /// altitude in meters
    pub altitude: f64,

    /// name of the observatory
    pub name: String,
}

impl FromStr for ObsCode {
    type Err = Error;

    /// Load an [`ObsCode`] from a single string.
    fn from_str(row: &str) -> KeteResult<Self> {
        let code = row[0..3].to_string();
        let rec_lon = f64::from_str(row[3..13].trim()).unwrap_or(f64::NAN);
        let cos = f64::from_str(row[13..21].trim()).unwrap_or(f64::NAN);
        let sin = f64::from_str(row[21..30].trim()).unwrap_or(f64::NAN);
        let vec = Vector3::new(cos, 0.0, sin) * EARTH_A;

        let rot = Rotation3::from_axis_angle(
            &UnitVector3::new_normalize([0.0, 0.0, 1.0].into()),
            rec_lon.to_radians(),
        );
        let vec = rot.transform_vector(&vec);

        let (lat, lon, altitude) = ecef_to_geodetic_lat_lon(vec.x, vec.y, vec.z);

        let name = row[30..].trim().to_string();
        Ok(Self {
            code: Desig::ObservatoryCode(code),
            lon: lon.to_degrees(),
            lat: lat.to_degrees(),
            altitude,
            name,
        })
    }
}

const PRELOAD_OBS: &[u8] = include_bytes!("../../data/mpc_obs.tsv");

lazy_static! {
    /// Observatory Codes
    pub static ref OBS_CODES: Vec<ObsCode> = {
        let mut codes = Vec::new();
        let text = str::from_utf8(PRELOAD_OBS).unwrap().split('\n');
        for row in text.skip(1) {
            // entries with gaps are skipped
            if let Ok(code) = ObsCode::from_str(row) { codes.push(code) };
        }
        codes
    };
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn obs_codes() {
        let codes = &OBS_CODES;
        assert!(!codes.is_empty());
    }
}
