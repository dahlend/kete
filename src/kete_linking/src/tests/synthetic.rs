// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Synthetic observers, objects and sights.
//!
//! Everything here is analytic, so no SPICE kernels are needed. Objects move
//! on exact two-body orbits about the Sun, which is the motion the search
//! assumes, so any error a test measures belongs to the linking code.

use std::f64::consts::TAU;

use kete_core::constants::{C_AU_PER_DAY_INV, GMS, GMS_SQRT};
use kete_core::desigs::Desig;
use kete_core::elements::CometElements;
use kete_core::frames::{Ecliptic, Equatorial, SunCenter, Vector};
use kete_core::kepler::{light_time_correct, propagate_two_body};
use kete_core::prelude::State;
use kete_core::time::{TDB, Time};

use crate::hypothesis::axes;
use crate::{Hypothesis, Sight};

/// One second, in days.
pub(crate) const SECOND: f64 = 1.0 / 86400.0;

/// One hour, in days.
pub(crate) const HOUR: f64 = 1.0 / 24.0;

/// One arcsecond, in radians.
pub(crate) const ARCSEC: f64 = TAU / 1_296_000.0;

/// Half-lengths of the search window in days.
pub(crate) const WINDOWS: [f64; 4] = [0.25, 1.0, 7.0, 15.0];

/// Time gaps between the two sights of a pair, in days.
pub(crate) const GAPS: [f64; 5] = [SECOND, 60.0 * SECOND, HOUR, 1.0, 10.0];

/// A state about the Sun.
pub(crate) type SunState = State<Equatorial, SunCenter>;

/// State at `epoch` on a circular orbit of `radius` AU about the Sun,
/// inclined to the ecliptic by `incl` degrees, at `phase` degrees from the
/// ascending node.
pub(crate) fn circular(epoch: Time<TDB>, radius: f64, incl: f64, phase: f64) -> SunState {
    let (sin_i, cos_i) = incl.to_radians().sin_cos();
    let (sin_p, cos_p) = phase.to_radians().sin_cos();
    let speed = (GMS / radius).sqrt();
    let pos = Vector::<Ecliptic>::new([cos_p, sin_p * cos_i, sin_p * sin_i]) * radius;
    let vel = Vector::<Ecliptic>::new([-sin_p, cos_p * cos_i, cos_p * sin_i]) * speed;
    State::new(
        Desig::Empty,
        epoch,
        pos.into_frame::<Equatorial>(),
        vel.into_frame::<Equatorial>(),
        SunCenter,
    )
}

/// An observer whose position is an analytic function of time.
///
/// The position is a circular orbit about the Sun, scaled by a constant, plus
/// a circle about that point. The scale gives points that keep station on the
/// line from the Sun to the orbiting point, such as near a Lagrange point. The
/// circle gives the daily motion of a ground site or the orbit of a satellite.
#[derive(Debug, Clone)]
pub(crate) struct Observer {
    /// Short name for tables.
    pub(crate) name: &'static str,

    /// Orbit about the Sun. Its epoch is where the circle starts.
    orbit: SunState,

    /// Factor applied to the position and velocity on the orbit.
    scale: f64,

    /// Radius in AU of the circle about the scaled point.
    radius: f64,

    /// Period in days of that circle.
    period: f64,
}

impl Observer {
    /// The center of a planet on a circular orbit 1 AU from the Sun.
    pub(crate) fn planet(epoch: Time<TDB>) -> Self {
        Self::near_planet("planet 1 AU", epoch, 1.0, 0.0, 1.0)
    }

    /// A site on the surface of that planet, which turns once a day.
    pub(crate) fn ground(epoch: Time<TDB>) -> Self {
        Self::near_planet("ground site", epoch, 1.0, 4.26e-5, 1.0)
    }

    /// A satellite in low orbit about that planet.
    pub(crate) fn low_orbit(epoch: Time<TDB>) -> Self {
        Self::near_planet("low orbit", epoch, 1.0, 4.6e-5, 0.066)
    }

    /// A point 0.01 AU sunward of that planet, as near its inner Lagrange
    /// point.
    pub(crate) fn sunward(epoch: Time<TDB>) -> Self {
        Self::near_planet("L1-like", epoch, 0.99, 0.0, 1.0)
    }

    /// An observer on a Venus-like orbit.
    pub(crate) fn inner(epoch: Time<TDB>) -> Self {
        Self::on_orbit("0.72 AU", circular(epoch, 0.723, 3.4, 200.0))
    }

    /// An observer at the distance of Jupiter.
    pub(crate) fn outer(epoch: Time<TDB>) -> Self {
        Self::on_orbit("5.2 AU", circular(epoch, 5.2, 1.3, 300.0))
    }

    /// An observer on an orbit inclined 70 degrees, 1.4 AU out of the
    /// ecliptic at `epoch`.
    pub(crate) fn polar(epoch: Time<TDB>) -> Self {
        Self::on_orbit("polar 1.5 AU", circular(epoch, 1.5, 70.0, 90.0))
    }

    /// A second site on the planet, `turn` degrees around it from
    /// [`Self::ground`].
    pub(crate) fn ground_at(name: &'static str, epoch: Time<TDB>, turn: f64) -> Self {
        let mut site = Self::ground(epoch);
        site.name = name;
        site.orbit = propagate_two_body(&site.orbit, epoch - turn / 360.0).unwrap();
        site
    }

    /// A spacecraft on the orbit of the planet, trailing it by 0.3 AU.
    pub(crate) fn trailing(epoch: Time<TDB>) -> Self {
        Self::on_orbit("trailing 0.3 AU", circular(epoch, 1.0, 0.0, 23.0))
    }

    /// Every kind of observer.
    pub(crate) fn all(epoch: Time<TDB>) -> Vec<Self> {
        vec![
            Self::planet(epoch),
            Self::ground(epoch),
            Self::low_orbit(epoch),
            Self::sunward(epoch),
            Self::inner(epoch),
            Self::outer(epoch),
            Self::polar(epoch),
        ]
    }

    /// An observer tied to the planet at 1 AU.
    fn near_planet(
        name: &'static str,
        epoch: Time<TDB>,
        scale: f64,
        radius: f64,
        period: f64,
    ) -> Self {
        Self {
            name,
            orbit: circular(epoch, 1.0, 0.0, 40.0),
            scale,
            radius,
            period,
        }
    }

    /// An observer following an orbit about the Sun.
    fn on_orbit(name: &'static str, orbit: SunState) -> Self {
        Self {
            name,
            orbit,
            scale: 1.0,
            radius: 0.0,
            period: 1.0,
        }
    }

    /// Position and velocity of the observer with respect to the Sun.
    pub(crate) fn state(&self, time: Time<TDB>) -> SunState {
        let mut state = propagate_two_body(&self.orbit, time).unwrap();
        let turn = TAU * (time - self.orbit.epoch).elapsed / self.period;
        let (sin, cos) = turn.sin_cos();
        let rate = TAU / self.period;
        state.pos = state.pos * self.scale + Vector::new([cos, sin, 0.0]) * self.radius;
        state.vel = state.vel * self.scale + Vector::new([-sin, cos, 0.0]) * (self.radius * rate);
        state
    }
}

/// A kind of object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Population {
    /// Main belt asteroids.
    MainBelt,

    /// Objects within 0.1 AU of the planet at 1 AU.
    Close,

    /// Near-Earth objects, at any distance from the planet.
    NearEarth,

    /// Objects which stay inside 1 AU from the Sun.
    Interior,

    /// Objects at the distance of Jupiter.
    Trojan,

    /// Objects beyond Neptune.
    Tno,

    /// Highly inclined and retrograde orbits.
    Tilted,

    /// Unbound orbits.
    Hyperbolic,

    /// Objects between Jupiter and Neptune. Not in [`Self::ALL`].
    Centaur,
}

impl Population {
    /// Every population.
    pub(crate) const ALL: [Self; 8] = [
        Self::MainBelt,
        Self::Close,
        Self::NearEarth,
        Self::Interior,
        Self::Trojan,
        Self::Tno,
        Self::Tilted,
        Self::Hyperbolic,
    ];

    /// Every population whose orbits are bound, which are the ones in scope.
    pub(crate) const BOUND: [Self; 7] = [
        Self::MainBelt,
        Self::Close,
        Self::NearEarth,
        Self::Interior,
        Self::Trojan,
        Self::Tno,
        Self::Tilted,
    ];

    /// State at `epoch` of a random member of the population.
    pub(crate) fn draw(self, rng: &mut Rng, epoch: Time<TDB>) -> SunState {
        let Some((peri, ecc, incl)) = self.shape(rng) else {
            // A near-Earth orbit through a point close to the planet.
            let planet = Observer::planet(epoch).state(epoch);
            let dist = 10.0_f64.powf(rng.between(-3.0, -1.0));
            let pos = planet.pos + rng.direction() * dist;
            let vel = planet.vel * rng.between(0.8, 1.2) + rng.direction() * 0.004;
            return State::new(Desig::Empty, epoch, pos, vel, SunCenter);
        };
        // Bound orbits start anywhere on the orbit. Unbound ones start
        // within a few months of perihelion.
        let semi = peri / (1.0 - ecc);
        let from_peri = if ecc < 1.0 {
            rng.uniform() * TAU * (semi.powi(3) / GMS).sqrt()
        } else {
            rng.between(-150.0, 150.0)
        };
        let elements = CometElements {
            desig: Desig::Empty,
            epoch,
            eccentricity: ecc,
            inclination: incl.to_radians(),
            lon_of_ascending: rng.between(0.0, TAU),
            peri_time: epoch - from_peri,
            peri_arg: rng.between(0.0, TAU),
            peri_dist: peri,
            center_id: 10,
            gm_sqrt: GMS_SQRT,
        };
        let state = elements.try_to_state().unwrap().into_frame::<Equatorial>();
        State::new(Desig::Empty, epoch, state.pos, state.vel, SunCenter)
    }

    /// Perihelion distance in AU, eccentricity, and inclination to the
    /// ecliptic in degrees, of a random member of the population. `None` for
    /// [`Self::Close`], which is described by where it is instead.
    pub(crate) fn shape(self, rng: &mut Rng) -> Option<(f64, f64, f64)> {
        Some(match self {
            Self::MainBelt => {
                let ecc = rng.between(0.0, 0.25);
                (
                    rng.between(2.1, 3.3) * (1.0 - ecc),
                    ecc,
                    rng.between(0.0, 20.0),
                )
            }
            Self::Close => return None,
            Self::NearEarth => {
                let peri = rng.between(0.4, 1.25);
                let semi = rng.between(peri.max(0.9) + 0.05, 2.6);
                (peri, 1.0 - peri / semi, rng.between(0.0, 30.0))
            }
            Self::Interior => {
                let semi = rng.between(0.5, 0.85);
                let ecc = rng.between(0.0, 0.95 / semi - 1.0);
                (semi * (1.0 - ecc), ecc, rng.between(0.0, 30.0))
            }
            Self::Trojan => {
                let ecc = rng.between(0.0, 0.12);
                (
                    rng.between(5.1, 5.3) * (1.0 - ecc),
                    ecc,
                    rng.between(0.0, 30.0),
                )
            }
            Self::Tno => {
                let ecc = rng.between(0.0, 0.2);
                (
                    rng.between(38.0, 48.0) * (1.0 - ecc),
                    ecc,
                    rng.between(0.0, 30.0),
                )
            }
            Self::Tilted => {
                let ecc = rng.between(0.0, 0.5);
                (
                    rng.between(1.5, 4.0) * (1.0 - ecc),
                    ecc,
                    rng.between(60.0, 170.0),
                )
            }
            Self::Hyperbolic => (
                rng.between(0.5, 3.0),
                rng.between(1.1, 3.0),
                rng.between(0.0, 180.0),
            ),
            Self::Centaur => {
                let ecc = rng.between(0.0, 0.5);
                (
                    rng.between(8.0, 25.0) * (1.0 - ecc),
                    ecc,
                    rng.between(0.0, 35.0),
                )
            }
        })
    }

    /// State at the epoch of `observer` of a random bound member of the
    /// population which the observer sees in a random direction of `field`.
    ///
    /// Without `distances`, the distance from the Sun follows the time the
    /// orbit spends there, and the object is where the line of sight meets
    /// it. With `distances`, the distance from the observer is uniform in its
    /// logarithm between the two values in AU, and the orbit is one of the
    /// population which passes through that point, using the shapes of
    /// [`Self::NearEarth`] for [`Self::Close`]. Either way the orbit plane
    /// is turned about the ecliptic pole to hold the point, so only orbits
    /// inclined at least as much as the point's ecliptic latitude about the
    /// Sun can be used.
    pub(crate) fn in_field(
        self,
        rng: &mut Rng,
        observer: &SunState,
        field: &Field,
        distances: Option<(f64, f64)>,
    ) -> SunState {
        let shapes = if self == Self::Close {
            Self::NearEarth
        } else {
            self
        };
        for _ in 0..1_000_000 {
            let (peri, ecc, incl) = shapes.shape(rng).unwrap();
            assert!(ecc < 1.0, "only bound orbits can be placed in a field");
            let semi = peri / (1.0 - ecc);
            let toward = field.direction(rng);
            // Eccentric anomaly at the point, from its distance to the Sun.
            let (pos, ecc_anomaly) = if let Some((near, far)) = distances {
                let range = near * (far / near).powf(rng.uniform());
                let pos = observer.pos + toward * range;
                let cos_e = (1.0 - pos.norm() / semi) / ecc;
                if cos_e.is_nan() || cos_e.abs() > 1.0 {
                    continue;
                }
                let sign = if rng.uniform() < 0.5 { -1.0 } else { 1.0 };
                (pos, sign * cos_e.acos())
            } else {
                let mean = rng.between(-TAU / 2.0, TAU / 2.0);
                let mut ecc_anomaly = mean;
                for _ in 0..50 {
                    ecc_anomaly -= (ecc_anomaly - ecc * ecc_anomaly.sin() - mean)
                        / (1.0 - ecc * ecc_anomaly.cos());
                }
                let dist = semi * (1.0 - ecc * ecc_anomaly.cos());
                // Where the line of sight is `dist` from the Sun.
                let along = toward.dot(&observer.pos);
                let square = along * along - observer.pos.norm_squared() + dist * dist;
                if square < 0.0 {
                    continue;
                }
                let root = square.sqrt();
                let range = if rng.uniform() < 0.5 && -along - root > 0.0 {
                    -along - root
                } else {
                    -along + root
                };
                if range <= 0.0 {
                    continue;
                }
                (observer.pos + toward * range, ecc_anomaly)
            };

            // The orbit plane holds the point and has the inclination.
            let radial = pos.into_frame::<Ecliptic>().normalize();
            let (lat, lon) = (radial[2].asin(), radial[1].atan2(radial[0]));
            let tilt = incl.to_radians();
            let sine = -lat.tan() / tilt.tan();
            if sine.is_nan() || sine.abs() > 1.0 {
                continue;
            }
            let node = if rng.uniform() < 0.5 {
                lon + sine.asin()
            } else {
                lon + TAU / 2.0 - sine.asin()
            };
            let pole = Vector::<Ecliptic>::new([
                tilt.sin() * node.sin(),
                -tilt.sin() * node.cos(),
                tilt.cos(),
            ]);
            let dist = pos.norm();
            let out = (GMS * semi).sqrt() * ecc * ecc_anomaly.sin() / dist;
            let across = (GMS * semi * (1.0 - ecc * ecc)).sqrt() / dist;
            let vel = (radial * out + pole.cross(&radial) * across).into_frame::<Equatorial>();
            return State::new(Desig::Empty, observer.epoch, pos, vel, SunCenter);
        }
        panic!("no member of {self:?} can be seen in this field");
    }
}

/// A round patch of the sky.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Field {
    /// Unit vector to the center.
    pub(crate) center: Vector<Equatorial>,

    /// Angular radius in radians.
    pub(crate) radius: f64,
}

impl Field {
    /// The field of `radius` degrees seen from an observer at `observer`,
    /// centered `elong` degrees from the Sun at `lat` degrees of ecliptic
    /// latitude, on the eastern side of the Sun.
    pub(crate) fn new(observer: &Vector<Equatorial>, elong: f64, lat: f64, radius: f64) -> Self {
        let sun = (-*observer).into_frame::<Ecliptic>().normalize();
        let pole = Vector::<Ecliptic>::new([0.0, 0.0, 1.0]);
        let level = Vector::<Ecliptic>::new([sun[0], sun[1], 0.0]).normalize();
        let east = pole.cross(&level);
        let (sin_b, cos_b) = lat.to_radians().sin_cos();
        let cos_l = (elong.to_radians().cos() - sin_b * sun[2]) / (cos_b * sun.dot(&level));
        assert!(
            cos_l.abs() <= 1.0 + 1e-6,
            "no direction at elongation {elong} and latitude {lat}"
        );
        let lon = cos_l.clamp(-1.0, 1.0).acos();
        let center = (level * (cos_b * lon.cos()) + east * (cos_b * lon.sin()) + pole * sin_b)
            .into_frame::<Equatorial>();
        Self {
            center,
            radius: radius.to_radians(),
        }
    }

    /// Solid angle in square degrees.
    pub(crate) fn area(&self) -> f64 {
        TAU * (1.0 - self.radius.cos()) * (180.0 / (TAU / 2.0)).powi(2)
    }

    /// A direction uniform over the field.
    pub(crate) fn direction(&self, rng: &mut Rng) -> Vector<Equatorial> {
        // The coordinate axis farthest from the center, so that the cross
        // product is well defined.
        let [x, y, z] = <[f64; 3]>::from(self.center).map(f64::abs);
        let axis = if x <= y && x <= z {
            [1.0, 0.0, 0.0]
        } else if y <= z {
            [0.0, 1.0, 0.0]
        } else {
            [0.0, 0.0, 1.0]
        };
        let first = Vector::new(axis).cross(&self.center).normalize();
        let second = self.center.cross(&first);
        let cos_t = 1.0 - rng.uniform() * (1.0 - self.radius.cos());
        let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
        let (sin_p, cos_p) = rng.between(0.0, TAU).sin_cos();
        (self.center * cos_t + first * (sin_t * cos_p) + second * (sin_t * sin_p)).normalize()
    }
}

/// An object described by how an observer sees it at one time.
#[derive(Debug)]
pub(crate) struct Case {
    /// Short name for tables.
    pub(crate) name: &'static str,

    /// Kind of object this stands for.
    pub(crate) population: Population,

    /// Distance from the observer in AU.
    pub(crate) dist: f64,

    /// Angle between the object and the Sun as seen by the observer, degrees.
    elong: f64,

    /// Speed in units of the circular speed at the object's distance from the
    /// Sun. Above the square root of two the orbit is hyperbolic.
    speed: f64,

    /// Tilt of the velocity out of the ecliptic in degrees. Above 90 the
    /// orbit is retrograde.
    incl: f64,

    /// Angle of the velocity above the plane normal to the Sun direction, in
    /// degrees. Positive is moving away from the Sun.
    climb: f64,
}

/// Shorthand to build a [`Case`].
const fn case(
    name: &'static str,
    population: Population,
    [dist, elong, speed, incl, climb]: [f64; 5],
) -> Case {
    Case {
        name,
        population,
        dist,
        elong,
        speed,
        incl,
        climb,
    }
}

/// Objects covering the populations and geometries of interest.
///
/// The names describe the object as seen from an observer 1 AU from the Sun.
/// From other observers the same rows are simply objects at that distance and
/// elongation.
pub(crate) const CASES: [Case; 22] = [
    case(
        "main belt e180",
        Population::MainBelt,
        [1.7, 180.0, 1.0, 8.0, 5.0],
    ),
    case(
        "main belt e120",
        Population::MainBelt,
        [2.2, 120.0, 0.95, 12.0, -8.0],
    ),
    case(
        "main belt e90",
        Population::MainBelt,
        [2.5, 90.0, 1.05, 5.0, 3.0],
    ),
    case(
        "main belt e45",
        Population::MainBelt,
        [3.3, 45.0, 1.0, 15.0, -4.0],
    ),
    case(
        "close 0.001 e120",
        Population::Close,
        [0.001, 120.0, 1.2, 10.0, 20.0],
    ),
    case(
        "close 0.003 e170",
        Population::Close,
        [0.003, 170.0, 0.85, 25.0, -15.0],
    ),
    case(
        "close 0.01 e90",
        Population::Close,
        [0.01, 90.0, 1.25, 5.0, -25.0],
    ),
    case(
        "close 0.05 e60",
        Population::Close,
        [0.05, 60.0, 1.15, 30.0, 15.0],
    ),
    case(
        "close 0.1 e45",
        Population::Close,
        [0.1, 45.0, 1.1, 8.0, -20.0],
    ),
    case(
        "near 0.3 e150",
        Population::NearEarth,
        [0.3, 150.0, 1.2, 20.0, 30.0],
    ),
    case(
        "near 0.2 e25",
        Population::NearEarth,
        [0.2, 25.0, 1.1, 12.0, 10.0],
    ),
    case(
        "interior near e30",
        Population::Interior,
        [0.5, 30.0, 0.9, 15.0, 10.0],
    ),
    case(
        "interior far e30",
        Population::Interior,
        [1.2, 30.0, 0.9, 15.0, -10.0],
    ),
    case(
        "interior e20",
        Population::Interior,
        [0.35, 20.0, 1.0, 25.0, 5.0],
    ),
    case(
        "trojan e180",
        Population::Trojan,
        [4.2, 180.0, 1.0, 12.0, 2.0],
    ),
    case(
        "trojan e60",
        Population::Trojan,
        [5.6, 60.0, 1.0, 20.0, -3.0],
    ),
    case("tno e180", Population::Tno, [39.0, 180.0, 1.0, 5.0, 2.0]),
    case("tno e90", Population::Tno, [40.0, 90.0, 0.95, 20.0, -5.0]),
    case(
        "retrograde e135",
        Population::Tilted,
        [2.0, 135.0, 1.1, 160.0, 15.0],
    ),
    case(
        "high incl e100",
        Population::Tilted,
        [1.5, 100.0, 1.0, 80.0, -10.0],
    ),
    case(
        "hyperbolic e110",
        Population::Hyperbolic,
        [1.5, 110.0, 1.6, 40.0, -35.0],
    ),
    case(
        "hyperbolic close",
        Population::Hyperbolic,
        [0.02, 140.0, 2.0, 120.0, 40.0],
    ),
];

impl Case {
    /// The case with this name.
    pub(crate) fn named(name: &str) -> &'static Self {
        CASES.iter().find(|case| case.name == name).unwrap()
    }

    /// State of the object at the epoch of `observer`, which is the state of
    /// the observer who sees it as this case describes.
    pub(crate) fn object(&self, observer: &SunState) -> SunState {
        let pole = Vector::<Ecliptic>::new([0.0, 0.0, 1.0]).into_frame::<Equatorial>();

        // The direction to the object is rotated from the Sun direction by
        // the elongation, in a plane tilted from the ecliptic so that the
        // geometry is not coplanar with the orbit of the observer.
        let sun = -observer.pos.normalize();
        let east = pole.cross(&sun).normalize();
        let north = sun.cross(&east);
        let (sin_e, cos_e) = self.elong.to_radians().sin_cos();
        let direction = sun * cos_e + (east * 0.95 + north * 0.3).normalize() * sin_e;
        let pos = observer.pos + direction * self.dist;

        let radial = pos.normalize();
        let along = pole.cross(&radial).normalize();
        let normal = radial.cross(&along);
        let (sin_i, cos_i) = self.incl.to_radians().sin_cos();
        let (sin_c, cos_c) = self.climb.to_radians().sin_cos();
        let speed = self.speed * (GMS / pos.norm()).sqrt();
        let vel = (radial * sin_c + (along * cos_i + normal * sin_i) * cos_c) * speed;
        State::new(Desig::Empty, observer.epoch, pos, vel, SunCenter)
    }
}

/// The state of `object` when it sent the light which reaches the point
/// `observer` at `time`.
pub(crate) fn emitted(
    object: &SunState,
    observer: &Vector<Equatorial>,
    time: Time<TDB>,
) -> SunState {
    let mut state = object.clone();
    // Each pass shrinks the light time error by the speed of the object over
    // the speed of light.
    for _ in 0..5 {
        let delay = (state.pos - observer).norm() * C_AU_PER_DAY_INV;
        state = propagate_two_body(object, time - delay).unwrap();
    }
    state
}

/// Uncertainty recorded in an exact sight, in radians. Every sight needs a
/// positive one; a picoradian is far below any astrometry and far above the
/// rounding of a unit vector.
pub(crate) const EXACT: f64 = 1e-12;

/// The exact line of sight from `observer` to `object` for light received at
/// `time`, and the distance to the object along it in AU.
pub(crate) fn sight(observer: &Observer, object: &SunState, time: Time<TDB>) -> (Sight, f64) {
    let observer = observer.state(time).pos;
    let offset = emitted(object, &observer, time).pos - observer;
    let range = offset.norm();
    let sight = Sight {
        time,
        observer,
        direction: offset / range,
        sigma: EXACT,
    };
    (sight, range)
}

/// Exact sights of every object by every observer at every time, and for
/// each sight the index of the object it shows.
pub(crate) fn observe(
    observers: &[Observer],
    objects: &[SunState],
    times: &[Time<TDB>],
) -> (Vec<Sight>, Vec<usize>) {
    let mut sights = Vec::new();
    let mut labels = Vec::new();
    for &time in times {
        for observer in observers {
            for (label, object) in objects.iter().enumerate() {
                sights.push(sight(observer, object, time).0);
                labels.push(label);
            }
        }
    }
    (sights, labels)
}

/// Turn the direction of `sight` by a Gaussian error of `sigma` radians on
/// each of the two axes of the sky, and record `sigma` in it, or [`EXACT`]
/// for none.
pub(crate) fn add_noise(sight: &mut Sight, sigma: f64, rng: &mut Rng) {
    let dir = sight.direction;
    // The coordinate axis farthest from the direction, so that the cross
    // product is well defined everywhere on the sky.
    let [x, y, z] = <[f64; 3]>::from(dir).map(f64::abs);
    let axis = if x <= y && x <= z {
        [1.0, 0.0, 0.0]
    } else if y <= z {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let east = Vector::new(axis).cross(&dir).normalize();
    let north = dir.cross(&east);
    let shift = east * (sigma * rng.normal()) + north * (sigma * rng.normal());
    sight.direction = (dir + shift).normalize();
    sight.sigma = sigma.max(EXACT);
}

/// The Sun as the origin of a hypothesis with reference time `epoch`.
pub(crate) fn sun(epoch: Time<TDB>) -> SunState {
    let zero = Vector::new([0.0; 3]);
    State::new(Desig::Empty, epoch, zero, zero, SunCenter)
}

/// The coordinate axis closest to `direction`.
pub(crate) fn nearest_axis(direction: &Vector<Equatorial>) -> Vector<Equatorial> {
    let aligned = |axis: &Vector<Equatorial>| axis.dot(direction);
    axes()
        .into_iter()
        .max_by(|a, b| aligned(a).total_cmp(&aligned(b)))
        .unwrap()
}

/// The hypothesis which is true of `object`, a state at the epoch of
/// `origin`, for a given origin and direction.
pub(crate) fn true_hypothesis(
    object: &SunState,
    origin: &SunState,
    direction: Vector<Equatorial>,
) -> Hypothesis {
    Hypothesis {
        origin: origin.clone(),
        direction,
        depth: direction.dot(&(object.pos - origin.pos)),
        depth_rate: direction.dot(&(object.vel - origin.vel)),
    }
}

/// A small deterministic random number generator, so that random scenarios
/// repeat exactly and need no dependency.
#[derive(Debug)]
pub(crate) struct Rng(pub(crate) u64);

impl Rng {
    /// Uniform sample in `(0, 1]`, from the `SplitMix64` sequence.
    pub(crate) fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 11) + 1) as f64 / (1_u64 << 53) as f64
    }

    /// Integer sample below `count`.
    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        reason = "the product is in [0, count]"
    )]
    pub(crate) fn below(&mut self, count: usize) -> usize {
        ((self.uniform() * count as f64) as usize).min(count - 1)
    }

    /// Uniform sample between `low` and `high`.
    pub(crate) fn between(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.uniform()
    }

    /// Sample of the unit normal distribution, by the Box-Muller transform.
    pub(crate) fn normal(&mut self) -> f64 {
        (-2.0 * self.uniform().ln()).sqrt() * (TAU * self.uniform()).cos()
    }

    /// Unit vector in a direction uniform over the sphere.
    pub(crate) fn direction(&mut self) -> Vector<Equatorial> {
        Vector::new([self.normal(), self.normal(), self.normal()]).normalize()
    }
}

/// Lines of sight of the two-body orbit of `object` from the two-body orbit
/// of `observer`, at these days after `epoch`, with light time, and with
/// uncertainty `sigma`. Each direction is moved by `noise()` radians across
/// the sky, along the equator, and then by `noise()` up it.
pub(crate) fn sights(
    object: &SunState,
    observer: &SunState,
    epoch: f64,
    days: &[f64],
    sigma: f64,
    mut noise: impl FnMut() -> f64,
) -> Vec<Sight> {
    days.iter()
        .map(|&day| {
            let time: Time<TDB> = (epoch + day).into();
            let seen_from = propagate_two_body(observer, time).unwrap().pos;
            let at = propagate_two_body(object, time).unwrap();
            let at = light_time_correct(&at, &seen_from).unwrap();
            let direction = (at.pos - seen_from).normalize();
            let across = direction.cross(&Vector::new([0.0, 0.0, 1.0])).normalize();
            let up = direction.cross(&across);
            Sight {
                time,
                observer: seen_from,
                direction: (direction + across * noise() + up * noise()).normalize(),
                sigma,
            }
        })
        .collect()
}
