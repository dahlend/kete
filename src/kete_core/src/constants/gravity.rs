use nalgebra::Vector3;

use crate::frames::{Ecliptic, InertialFrame};

use super::{AU_KM, C_AU_PER_DAY_INV_SQUARED};

/// Standard Gravitational Constants of the Sun
/// AU^3 / (Day^2 * Solar Mass)
pub const GMS: f64 = 0.00029591220828411956;

/// Gaussian gravitational constant, equivalent to sqrt of GMS.
/// AU^(3/2) per (Day sqrt(Solar Mass))
pub const GMS_SQRT: f64 = 0.01720209894996;

/// Sun J2 Parameter
///
/// This paper below a source, however there are several papers which all put
/// the Sun's J2 at 2.2e-7.
///
/// "Prospects of Dynamical Determination of General Relativity Parameter β and Solar
/// Quadrupole Moment J2 with Asteroid Radar Astronomy"
/// The Astrophysical Journal, 845:166 (5pp), 2017 August 20
pub const SUN_J2: f64 = 2.2e-7;

/// Earth J2 Parameter
/// See "Revisiting Spacetrack Report #3" - Final page of appendix.
pub const EARTH_J2: f64 = 0.00108262998905;

/// Earth J3 Parameter
pub const EARTH_J3: f64 = -0.00000253215306;

/// Earth J4 Parameter
pub const EARTH_J4: f64 = -0.00000161098761;

/// Jupiter J2 Parameter
///
/// "Measurement of Jupiter’s asymmetric gravity field"
/// <https://www.nature.com/articles/nature25776>
/// Nature 555, 220-220, 2018 March 8
pub const JUPITER_J2: f64 = 0.014696572;

/// Known massive objects
pub const MASSES: &[GravParams] = &[
    // Sun
    GravParams {
        naif_id: 10,
        mass: GMS,
        radius: 0.004654758765894654,
    },
    // Mercury Barycenter
    GravParams {
        naif_id: 1,
        mass: 1.66012082548908e-07 * GMS,
        radius: 1.63139354095098e-05,
    },
    // Venus Barycenter
    GravParams {
        naif_id: 2,
        mass: 2.44783828779694e-06 * GMS,
        radius: 4.04537843465442e-05,
    },
    // Earth
    GravParams {
        naif_id: 399,
        mass: 3.00348961546514e-06 * GMS,
        radius: 4.33036684926559e-05,
    },
    // Moon
    GravParams {
        naif_id: 301,
        mass: 3.69430335010988e-08 * GMS,
        radius: 1.16138016662292e-05,
    },
    // Mars Barycenter
    GravParams {
        naif_id: 4,
        mass: 3.22715608291416e-07 * GMS,
        radius: 2.27021279387769e-05,
    },
    // Jupiter
    GravParams {
        naif_id: 5,
        mass: 0.0009547919099414246 * GMS,
        radius: 0.00047789450254521576,
    },
    // Saturn Barycenter
    GravParams {
        naif_id: 6,
        mass: 0.00028588567002459455 * GMS,
        radius: 0.0004028666966848747,
    },
    // Uranus Barycenter
    GravParams {
        naif_id: 7,
        mass: 4.36624961322212e-05 * GMS,
        radius: 0.0001708513622580592,
    },
    // Neptune Barycenter
    GravParams {
        naif_id: 8,
        mass: 5.15138377265457e-05 * GMS,
        radius: 0.0001655371154958558,
    },
    // Ceres
    GravParams {
        naif_id: 20000001,
        mass: 4.7191659919436e-10 * GMS,
        radius: 6.276827e-6,
    },
    // Pallas
    GravParams {
        naif_id: 20000002,
        mass: 1.0259143964958e-10 * GMS,
        radius: 3.415824e-6,
    },
    // Vesta
    GravParams {
        naif_id: 20000004,
        mass: 1.3026452498655e-10 * GMS,
        radius: 3.512082e-6,
    },
    // Hygiea
    GravParams {
        naif_id: 20000010,
        mass: 4.3953391300849e-11 * GMS,
        radius: 2.894426e-6,
    },
    // Interamnia
    GravParams {
        naif_id: 20000704,
        mass: 1.911e-11 * GMS,
        radius: 2.219283e-6,
    },
];

/// Earth-Moon Barycenter
pub const EM_BARY: GravParams = GravParams {
    naif_id: 3,
    mass: (3.00348961546514e-06 + 3.69430335010988e-08) * GMS,
    // earth - moon distance
    radius: 385_000.0 / AU_KM,
};

/// Known planet masses, Sun through Neptune, including the Moon
pub const PLANETS: &[GravParams] = &[
    MASSES[0], MASSES[1], MASSES[2], MASSES[3], MASSES[4], MASSES[5], MASSES[6], MASSES[7],
    MASSES[8], MASSES[9],
];

/// Known planet masses, Sun through Neptune, Excluding the moon
pub const SIMPLE_PLANETS: &[GravParams] = &[
    MASSES[0], MASSES[1], MASSES[2], EM_BARY, MASSES[5], MASSES[6], MASSES[7], MASSES[8], MASSES[9],
];

/// Gravitational model for a basic object
#[derive(Debug, Clone, Copy)]
pub struct GravParams {
    /// Associated NAIF id
    pub naif_id: i32,

    /// Mass of the object in GMS
    pub mass: f64,

    /// Radius of the object in AU.
    pub radius: f64,
}

impl GravParams {
    /// Add acceleration to the provided accel vector.
    #[inline(always)]
    pub fn add_acceleration(
        self,
        accel: &mut Vector3<f64>,
        rel_pos: &Vector3<f64>,
        rel_vel: &Vector3<f64>,
    ) {
        // Basic newtonian gravity
        let mass = self.mass;
        *accel -= &(rel_pos * (mass * rel_pos.norm().powi(-3)));

        // Special cases for different objects
        match self.naif_id {
            5 => {
                let radius = self.radius;

                // GR correction
                apply_gr_correction(accel, rel_pos, rel_vel, &mass);

                // J2 correction
                let rel_pos_eclip = Ecliptic::from_equatorial(*rel_pos);
                *accel += Ecliptic::to_equatorial(j2_correction(
                    &rel_pos_eclip,
                    &radius,
                    &JUPITER_J2,
                    &mass,
                ));
            }
            10 => {
                let radius = self.radius;

                // GR correction
                apply_gr_correction(accel, rel_pos, rel_vel, &mass);

                // J2 correction
                let rel_pos_eclip = Ecliptic::from_equatorial(*rel_pos);
                *accel +=
                    Ecliptic::to_equatorial(j2_correction(&rel_pos_eclip, &radius, &SUN_J2, &mass));
            }
            399 => *accel += j2_correction(rel_pos, &self.radius, &EARTH_J2, &mass),
            _ => (),
        }
    }
}

/// Calculate the effects of the J2 term
///
/// Z is the z component of the unit vector.
#[inline(always)]
fn j2_correction(rel_pos: &Vector3<f64>, radius: &f64, j2: &f64, mass: &f64) -> Vector3<f64> {
    let r = rel_pos.norm();
    let z_squared = 5.0 * (rel_pos.z / r).powi(2);

    // this is formatted a little funny in an attempt to reduce numerical noise
    // 3/2 * j2 * mass * earth_r^2 / distance^5
    let coef = 1.5 * j2 * mass * (radius / r).powi(2) * r.powi(-3);
    Vector3::<f64>::new(
        rel_pos.x * coef * (z_squared - 1.0),
        rel_pos.y * coef * (z_squared - 1.0),
        rel_pos.z * coef * (z_squared - 3.0),
    )
}

/// Add the effects of general relativistic motion to an acceleration vector
#[inline(always)]
fn apply_gr_correction(
    accel: &mut Vector3<f64>,
    rel_pos: &Vector3<f64>,
    rel_vel: &Vector3<f64>,
    mass: &f64,
) {
    let r_v = 4.0 * rel_pos.dot(rel_vel);

    let rel_v2: f64 = rel_vel.norm_squared();
    let r = rel_pos.norm();

    let gr_const: f64 = mass * C_AU_PER_DAY_INV_SQUARED * r.powi(-3);
    let c: f64 = 4. * mass / r - rel_v2;
    *accel += gr_const * (c * rel_pos + r_v * rel_vel);
}
