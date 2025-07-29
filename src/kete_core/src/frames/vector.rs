//! Cartesian Vectors with frame information.
//!
// BSD 3-Clause License
//
// Copyright (c) 2025, Dar Dahlen
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

use super::{Ecliptic, Equatorial, InertialFrame};
use nalgebra::{Rotation3, UnitVector3, Vector3};
use serde::{Deserialize, Serialize};
use std::f64::consts::{PI, TAU};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, Neg, Sub, SubAssign};

/// Vector with frame information.
/// All vectors are 3D vectors in an inertial frame.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub struct Vector<T: InertialFrame> {
    /// Underlying vector data.
    raw: [f64; 3],

    /// [`PhantomData`] is used here to keep track of the frame type.
    frame: PhantomData<T>,
}

impl<T: InertialFrame> Vector<T> {
    /// New Vector
    #[inline(always)]
    pub fn new(vec: [f64; 3]) -> Self {
        Self {
            raw: vec,
            frame: PhantomData,
        }
    }

    /// New Vector of NANs
    #[inline(always)]
    pub fn new_nan() -> Self {
        Self {
            raw: [f64::NAN, f64::NAN, f64::NAN],
            frame: PhantomData,
        }
    }

    /// Are all element of the vector finite.
    #[inline(always)]
    pub fn is_finite(&self) -> bool {
        self.raw.iter().all(|x| x.is_finite())
    }

    /// Convert Vector from one frame to another.
    #[inline(always)]
    pub fn into_frame<Target: InertialFrame>(self) -> Vector<Target> {
        let vec = T::convert::<Target>(self.into());
        Vector::<Target>::new(vec.into())
    }

    /// Rotate a vector around the specified rotation vector.
    ///
    /// # Arguments
    ///
    /// * `rotation_vec` - The single vector around which to rotate the vectors.
    /// * `angle` - The angle in radians to rotate the vectors.
    ///
    #[inline(always)]
    pub fn rotate_around(self, rotation_vec: Self, angle: f64) -> Self {
        let rot =
            Rotation3::from_axis_angle(&UnitVector3::new_normalize(rotation_vec.into()), angle);
        rot.transform_vector(&self.into()).into()
    }

    /// Dot product between two vectors
    #[inline(always)]
    pub fn dot(&self, other: &Self) -> f64 {
        self.raw
            .iter()
            .zip(other.raw.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Cross product between two vectors
    #[inline(always)]
    pub fn cross(&self, other: &Self) -> Self {
        Vector3::from(self.raw)
            .cross(&Vector3::from(other.raw))
            .into()
    }

    /// Squared euclidean length.
    #[inline(always)]
    pub fn norm_squared(&self) -> f64 {
        self.raw.iter().map(|a| a.powi(2)).sum()
    }

    /// The euclidean length of the vector.
    #[inline(always)]
    pub fn norm(&self) -> f64 {
        self.raw.iter().map(|a| a.powi(2)).sum::<f64>().sqrt()
    }

    /// The angle betweeen two vectors in radians.
    #[inline(always)]
    pub fn angle(&self, other: &Self) -> f64 {
        Vector3::from(self.raw).angle(&Vector3::from(other.raw))
    }

    /// Create a new vector of unit length in the same direction as this vector.
    #[inline(always)]
    pub fn normalize(&self) -> Self {
        self / self.norm()
    }

    /// Create a unit vector from polar spherical theta and phi angles in radians.
    ///
    /// <https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates>
    #[inline(always)]
    pub fn from_polar_spherical(theta: f64, phi: f64) -> Self {
        let (theta_sin, theta_cos) = theta.sin_cos();
        let (phi_sin, phi_cos) = phi.sin_cos();
        [theta_sin * phi_cos, theta_sin * phi_sin, theta_cos].into()
    }

    /// Convert a vector to polar spherical coordinates.
    ///
    /// <https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates>
    pub fn to_polar_spherical(&self) -> (f64, f64) {
        let vec = self.normalize();
        let theta = vec.raw[2].acos();
        let phi = vec.raw[1].atan2(vec.raw[0]) % TAU;
        (theta, phi)
    }
}

impl Vector<Ecliptic> {
    /// Create a unit vector from latitude and longitude in units of radians.
    #[inline(always)]
    pub fn from_lat_lon(lat: f64, lon: f64) -> Self {
        Self::from_polar_spherical(PI / 2.0 - lat, lon)
    }

    /// Convert a unit vector to latitude and longitude.
    #[inline(always)]
    pub fn to_lat_lon(self) -> (f64, f64) {
        let (mut lat, mut lon) = self.to_polar_spherical();
        if lat > PI {
            lat = TAU - lat;
            lon += PI;
        }
        (PI / 2.0 - lat, lon)
    }
}

impl Vector<Equatorial> {
    /// Create a unit vector from ra and dec in units of radians.
    #[inline(always)]
    pub fn from_ra_dec(ra: f64, dec: f64) -> Self {
        Self::from_polar_spherical(PI / 2.0 - dec, ra)
    }

    /// Convert a unit vector to ra and dec.
    #[inline(always)]
    pub fn to_ra_dec(self) -> (f64, f64) {
        let (mut dec, mut ra) = self.to_polar_spherical();
        if dec > PI {
            dec = TAU - dec;
            ra += PI;
        }
        (ra.rem_euclid(TAU), PI / 2.0 - dec)
    }
}

impl<T: InertialFrame> Index<usize> for Vector<T> {
    type Output = f64;
    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.raw[index]
    }
}

impl<T: InertialFrame> IndexMut<usize> for Vector<T> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.raw[index]
    }
}

impl<T: InertialFrame> IntoIterator for Vector<T> {
    type Item = f64;
    type IntoIter = std::array::IntoIter<Self::Item, 3>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.raw.into_iter()
    }
}

impl<T: InertialFrame> From<[f64; 3]> for Vector<T> {
    #[inline(always)]
    fn from(value: [f64; 3]) -> Self {
        Self::new(value)
    }
}

impl<T: InertialFrame> From<Vector<T>> for [f64; 3] {
    #[inline(always)]
    fn from(value: Vector<T>) -> Self {
        value.raw
    }
}

impl<T: InertialFrame> From<Vector3<f64>> for Vector<T> {
    #[inline(always)]
    fn from(value: Vector3<f64>) -> Self {
        Self::new(value.into())
    }
}

impl<T: InertialFrame> From<Vector<T>> for Vector3<f64> {
    #[inline(always)]
    fn from(value: Vector<T>) -> Self {
        value.raw.into()
    }
}

impl<T: InertialFrame> From<Vector<T>> for Vec<f64> {
    #[inline(always)]
    fn from(value: Vector<T>) -> Self {
        value.raw.into()
    }
}

impl<T: InertialFrame> Sub<&Self> for Vector<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(mut self, rhs: &Self) -> Self::Output {
        (0..3).for_each(|i| self.raw[i] -= rhs.raw[i]);
        self
    }
}

impl<T: InertialFrame> Sub<&Vector<T>> for &Vector<T> {
    type Output = Vector<T>;
    #[inline(always)]
    fn sub(self, rhs: &Vector<T>) -> Self::Output {
        let mut raw = self.raw;
        (0..3).for_each(|i| raw[i] -= rhs.raw[i]);
        raw.into()
    }
}

impl<T: InertialFrame> Sub<Self> for Vector<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(mut self, rhs: Self) -> Self::Output {
        (0..3).for_each(|i| self.raw[i] -= rhs.raw[i]);
        self
    }
}

impl<T: InertialFrame> Sub<Vector<T>> for &Vector<T> {
    type Output = Vector<T>;
    #[inline(always)]
    fn sub(self, rhs: Vector<T>) -> Self::Output {
        let mut raw = self.raw;
        (0..3).for_each(|i| raw[i] -= rhs.raw[i]);
        raw.into()
    }
}

impl<T: InertialFrame> AddAssign<&Self> for Vector<T> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: &Self) {
        (0..3).for_each(|i| self.raw[i] += rhs.raw[i]);
    }
}

impl<T: InertialFrame> SubAssign<&Self> for Vector<T> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &Self) {
        (0..3).for_each(|i| self.raw[i] -= rhs.raw[i]);
    }
}

impl<T: InertialFrame> Add<Self> for Vector<T> {
    type Output = Self;
    #[inline(always)]
    fn add(mut self, rhs: Self) -> Self::Output {
        (0..3).for_each(|i| self.raw[i] += rhs.raw[i]);
        self
    }
}

impl<T: InertialFrame> Add<&Self> for Vector<T> {
    type Output = Self;
    #[inline(always)]
    fn add(mut self, rhs: &Self) -> Self::Output {
        (0..3).for_each(|i| self.raw[i] += rhs.raw[i]);
        self
    }
}

impl<T: InertialFrame> Add<&Vector<T>> for &Vector<T> {
    type Output = Vector<T>;
    #[inline(always)]
    fn add(self, rhs: &Vector<T>) -> Self::Output {
        let mut vec = self.raw;
        (0..3).for_each(|i| vec[i] += rhs.raw[i]);
        vec.into()
    }
}

impl<T: InertialFrame> Add<Vector<T>> for &Vector<T> {
    type Output = Vector<T>;
    #[inline(always)]
    fn add(self, rhs: Vector<T>) -> Self::Output {
        let mut vec = self.raw;
        (0..3).for_each(|i| vec[i] += rhs.raw[i]);
        vec.into()
    }
}

impl<T: InertialFrame> Div<f64> for Vector<T> {
    type Output = Self;
    #[inline(always)]
    fn div(mut self, rhs: f64) -> Self::Output {
        (0..3).for_each(|i| self.raw[i] /= rhs);
        self
    }
}

impl<T: InertialFrame> Div<f64> for &Vector<T> {
    type Output = Vector<T>;
    #[inline(always)]
    fn div(self, rhs: f64) -> Self::Output {
        let mut vec = self.raw;
        (0..3).for_each(|i| vec[i] /= rhs);
        vec.into()
    }
}

impl<T: InertialFrame> Mul<f64> for Vector<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(mut self, rhs: f64) -> Self::Output {
        (0..3).for_each(|i| self.raw[i] *= rhs);
        self
    }
}

impl<T: InertialFrame> Mul<f64> for &Vector<T> {
    type Output = Vector<T>;
    #[inline(always)]
    fn mul(self, rhs: f64) -> Self::Output {
        let mut vec = self.raw;
        (0..3).for_each(|i| vec[i] *= rhs);
        vec.into()
    }
}

impl<T: InertialFrame> Mul<Vector<T>> for f64 {
    type Output = Vector<T>;
    #[inline(always)]
    fn mul(self, mut rhs: Vector<T>) -> Self::Output {
        (0..3).for_each(|i| rhs.raw[i] *= self);
        rhs
    }
}

impl<T: InertialFrame> Mul<&Vector<T>> for f64 {
    type Output = Vector<T>;
    #[inline(always)]
    fn mul(self, rhs: &Vector<T>) -> Self::Output {
        let mut vec = rhs.raw;
        (0..3).for_each(|i| vec[i] *= self);
        vec.into()
    }
}

impl<T: InertialFrame> Neg for &Vector<T> {
    type Output = Vector<T>;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        let mut vec = self.raw;
        (0..3).for_each(|i| vec[i] = -vec[i]);
        vec.into()
    }
}

impl<T: InertialFrame> Neg for Vector<T> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        let mut vec = self.raw;
        (0..3).for_each(|i| vec[i] = -vec[i]);
        vec.into()
    }
}

#[cfg(test)]
mod tests {
    use crate::frames::{FK4, Galactic};

    use super::*;

    #[test]
    fn test_frame_change() {
        let a = Vector::<Equatorial>::new([1.0, 2.0, 7.0]);
        let b = a.into_frame::<Ecliptic>();
        let c = b.into_frame::<Galactic>();
        let d = c.into_frame::<FK4>();
        let a_new = d.into_frame::<Equatorial>();
        assert!((a - a_new).norm() < f64::EPSILON * 20.0);
        assert!((b - a_new.into_frame()).norm() < f64::EPSILON * 20.0);
        assert!((c - a_new.into_frame()).norm() < f64::EPSILON * 20.0);
        assert!((d - a_new.into_frame()).norm() < f64::EPSILON * 20.0);
    }

    #[test]
    fn test_normalize() {
        let a = Vector::<Equatorial>::new([1.0, -2.0, 2.0]);
        let a_expected: Vector<_> = [1.0 / 3.0, -2.0 / 3.0, 2.0 / 3.0].into();
        let a_norm = a.normalize();

        assert!((a_norm - a_expected).norm() < f64::EPSILON * 10.0);
        assert!((a.norm() - 3.0).abs() < f64::EPSILON * 10.0);
        assert!((a.norm_squared() - 9.0).abs() < f64::EPSILON * 10.0);
    }
}
