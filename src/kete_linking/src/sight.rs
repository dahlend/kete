// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! A line of sight, the input unit of linking.

use kete_core::constants::C_AU_PER_DAY_INV;
use kete_core::frames::{Equatorial, Vector};
use kete_core::time::{TDB, Time};

/// One detection reduced to a line of sight.
///
/// The observer position is with respect to the Sun, so that nothing after
/// this point needs an ephemeris.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sight {
    /// Time the light reached the observer.
    pub time: Time<TDB>,

    /// Position of the observer with respect to the Sun at [`Sight::time`],
    /// in AU.
    pub observer: Vector<Equatorial>,

    /// Unit vector from the observer toward the detection: the astrometric
    /// direction, corrected for light travel time but not for aberration.
    /// See [`Sight::is_usable`] for how close to unit length it must be.
    pub direction: Vector<Equatorial>,

    /// One-sigma uncertainty of the direction on each axis of the sky, in
    /// radians. Positive, see [`Sight::is_usable`].
    pub sigma: f64,
}

/// Largest count of days this crate does arithmetic on, as a Julian date or
/// as a light travel time. Past it a number of days no longer resolves one
/// day, and sums of them overflow the day count of a [`Time`].
pub(crate) const MAX_DAYS: f64 = 9e15;

/// What makes a sight not [`Sight::is_usable`], for error messages.
pub(crate) const UNUSABLE: &str = "a time 9e15 days or more from JD 0, an observer which is not \
                                   finite, a direction which is not a unit vector, or an \
                                   uncertainty which is not finite and positive";

/// Largest error of the squared length of a vector taken as a unit vector.
///
/// A length error only rescales the ranges along a sight, or the depth of a
/// hypothesis, by the same fraction. This keeps it far below a step of the
/// hypothesis grid, and far above the rounding of a normalized vector.
const UNIT_TOLERANCE: f64 = 1e-9;

/// Angle between two unit vectors, accurate at every angle.
pub(crate) fn angle(a: &Vector<Equatorial>, b: &Vector<Equatorial>) -> f64 {
    a.cross(b).norm().atan2(a.dot(b))
}

/// Whether a vector has unit length to within [`UNIT_TOLERANCE`].
pub(crate) fn is_unit(vector: &Vector<Equatorial>) -> bool {
    (vector.norm_squared() - 1.0).abs() < UNIT_TOLERANCE
}

impl Sight {
    /// Whether every function of this crate accepts the sight: a Julian date
    /// within 9e15 days of zero, a finite observer, a direction whose squared
    /// length is within 1e-9 of one, and a finite positive uncertainty.
    ///
    /// Residuals and tolerances are in units of the uncertainty, so a sight
    /// without one could not be weighed against any other.
    #[must_use]
    pub fn is_usable(&self) -> bool {
        self.time.jd().abs() < MAX_DAYS
            && self.observer.is_finite()
            && is_unit(&self.direction)
            && self.sigma > 0.0
            && self.sigma.is_finite()
    }

    /// The point `range` AU along the line of sight, with respect to the Sun.
    pub fn position(&self, range: f64) -> Vector<Equatorial> {
        self.observer + self.direction * range
    }

    /// The time light left a point `range` AU along the line of sight to
    /// reach the observer at [`Sight::time`].
    pub fn emission_time(&self, range: f64) -> Time<TDB> {
        self.time - range * C_AU_PER_DAY_INV
    }
}

/// The earliest and the latest of some Julian dates, infinite and negative
/// infinite if there are none.
pub(crate) fn first_and_last(jds: impl IntoIterator<Item = f64>) -> (f64, f64) {
    jds.into_iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), t| {
            (lo.min(t), hi.max(t))
        })
}
