//! Coordinate frames and related conversions.
//!
//! Distances measured in AU, time is in units of days with TDB scaling.
//!

mod definitions;
mod rotation;
mod vector;
mod wgs_84;

pub use definitions::{
    Ecliptic, Equatorial, FK4, Galactic, InertialFrame, NonInertialFrame, calc_obliquity,
    earth_precession_rotation,
};
pub use rotation::{euler_rotation, quaternion_to_euler};
pub use vector::Vector;
pub use wgs_84::{
    EARTH_A, ecef_to_geodetic_lat_lon, geodetic_lat_lon_to_ecef, geodetic_lat_to_geocentric,
    prime_vert_radius,
};
