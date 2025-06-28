//! Coordinate frames and related conversions.
//!
//! Distances measured in AU, time is in units of days with TDB scaling.
//!

mod definitions;
mod earth;
mod rotation;
mod vector;

pub use definitions::{Ecliptic, Equatorial, FK4, Galactic, InertialFrame, NonInertialFrame};
pub use earth::{
    EARTH_A, approx_earth_pos_to_ecliptic, approx_solar_noon, approx_sun_dec, earth_obliquity,
    earth_precession_rotation, earth_rotation_angle, ecef_to_geodetic_lat_lon, equation_of_time,
    geocentric_radius, geodetic_lat_lon_to_ecef, geodetic_lat_to_geocentric, next_sunset_sunrise,
    prime_vert_radius,
};
pub use rotation::{euler_rotation, quaternion_to_euler};
pub use vector::Vector;
