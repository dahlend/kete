//! Conversion tools to and from WGS84 coordinate system

/// Earth semi major axis in km as defined by WGS84
pub const EARTH_A: f64 = 6378.1370;

/// Earth semi minor axis in km as defined by WGS84
const _EARTH_B: f64 = 6356.7523142;

// /// Earth inverse flattening as defined by WGS84
const _EARTH_INV_FLAT: f64 = 298.2572235629972;

/// Earth surface eccentricity squared, calculated from above.
/// e^2 = (2 - flattening) * flattening
const EARTH_E2: f64 = 0.0066943799901413165;

/// Prime vertical radius of curvature.
/// This is the radius of curvature of the earth surface at the specific geodetic
/// latitude.
pub fn prime_vert_radius(geodetic_lat: f64) -> f64 {
    EARTH_A / (1.0 - EARTH_E2 * geodetic_lat.sin().powi(2)).sqrt()
}

/// Compute geodetic lat/lon/height in radians/km from ECEF position in km.
pub fn ecef_to_geodetic_lat_lon(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let longitude = f64::atan2(y, x);
    let p = (x * x + y * y).sqrt();
    let geocen_lat = f64::atan2(p, z);

    // initial guess, and iterate.
    let mut geodetic_lat = geocen_lat;
    let mut h = 0.0;
    // this usually converges in only 1-2 iterations, but to reduce CPU branching
    // don't bother with a convergence check.
    for _ in 0..5 {
        let n = prime_vert_radius(geodetic_lat);
        h = p / geodetic_lat.cos() - n;
        geodetic_lat = f64::atan(z / p / (1.0 - EARTH_E2 * n / (n + h)));
    }

    (geodetic_lat, longitude, h)
}

/// Compute geocentric latitude from geodetic lat/height in radians/km .
pub fn geodetic_lat_to_geocentric(geodetic_lat: f64, h: f64) -> f64 {
    let n = prime_vert_radius(geodetic_lat);
    ((1.0 - EARTH_E2 * n / (n + h)) * geodetic_lat.tan()).atan()
}

/// Compute the ECEF X/Y/Z position in km from geodetic lat/lon/height in radians/km
pub fn geodetic_lat_lon_to_ecef(geodetic_lat: f64, geodetic_lon: f64, h: f64) -> (f64, f64, f64) {
    let n = prime_vert_radius(geodetic_lat);
    let (sin_gd_lat, cos_gd_lat) = geodetic_lat.sin_cos();
    let (sin_gd_lon, cos_gd_lon) = geodetic_lon.sin_cos();
    let x = (n + h) * cos_gd_lat * cos_gd_lon;
    let y = (n + h) * cos_gd_lat * sin_gd_lon;
    let z = ((1.0 - EARTH_E2) * n + h) * sin_gd_lat;
    (x, y, z)
}
