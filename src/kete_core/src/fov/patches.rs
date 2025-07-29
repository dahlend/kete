//! Basic Geometric shapes on the surface of a sphere
// BSD 3-Clause License
//
// Copyright (c) 2025, Dar Dahlen
// Copyright (c) 2025, California Institute of Technology
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

use crate::{
    frames::{Equatorial, Vector},
    io::serde_const_arr,
};
use serde::{Deserialize, Serialize};
use std::{f64::consts::FRAC_PI_2, ops::Neg};

/// Bounded areas can either contains a vector or not.
/// This enum specifies if the vector is within the area, or
/// the minimum distance the vector must move to be within the area.
#[derive(Debug, Clone)]
pub enum Contains {
    /// Vector is contained within the area.
    Inside,

    /// Vector is outside of the area
    /// The f64 defines the minimum distance required to move into the area.
    Outside(f64),
}

impl Contains {
    /// Returns true if the vector is inside the area.
    pub fn is_inside(&self) -> bool {
        matches!(self, Self::Inside)
    }
}

/// Given an iterable of [`Contains`], find the closest one to being Inside.
pub(super) fn closest_inside(contains: &[Contains]) -> (usize, Contains) {
    let mut best = (usize::MAX, f64::INFINITY);
    for (idx, con) in contains.iter().enumerate() {
        match con {
            Contains::Inside => return (idx, Contains::Inside),
            Contains::Outside(d) => {
                if d < &best.1 {
                    best = (idx, *d);
                }
            }
        }
    }
    (best.0, Contains::Outside(best.1))
}

/// Trait which defines an area on the surface area of a sphere.
pub trait SkyPatch: Sized {
    /// Checks to see if a unit vector is within the bounded area.
    fn contains(&self, obs_to_obj: &Vector<Equatorial>) -> Contains;

    /// Center of the field of view
    fn pointing(&self) -> Vector<Equatorial>;
}

/// A Spherical Polygon as represented by a series of planes through the central axis.
///
/// Conceptually the spherical polygon as defined here can be thought of as a unit
/// sphere with a polygon drawn on it. This polygon has sides defined by great circle
/// lines. If these great circle lines were continued all the way around the sphere,
/// they would define planes which cut the circle. Defining surface normals to each of
/// these planes enables us to define an "inside" and "outside" of the polygon, where
/// inside is defined by having all of the unit vectors pointing toward the inside of
/// the polygon.
///
/// To test if a point on the unit circle is inside of the polygon, all that is
/// required is it multiply each surface normal against the vector which defines the
/// point's position. If all of the dot products are positive, then the vector must lie
/// within the polygon. If any are negative, then the point must be outside of the
/// polygon.
///
/// Note that the surface normals are defined by planes through the center of the
/// sphere meaning that constructed polygons can only have great circle edges.
///
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SphericalPolygon<const N_SIDES: usize> {
    /// Normal vectors which define the boundary of a polygon.
    #[serde(with = "serde_const_arr")]
    edge_normals: [Vector<Equatorial>; N_SIDES],
}

/// A rectangular patch of sky.
pub type OnSkyRectangle = SphericalPolygon<4>;

impl OnSkyRectangle {
    /// Construct a rectangular spherical polygon.
    ///
    /// # Arguments
    ///
    /// * `edge_normals` - Normal vectors which define the boundary of a polygon.
    pub fn from_normals(edge_normals: [Vector<Equatorial>; 4]) -> Self {
        // construct the 4 normal vectors
        Self { edge_normals }
    }

    /// Construct a rectangular spherical polygon.
    ///
    /// This constructs a new [`SphericalPolygon`] made up of a rectangular shape on
    /// the unit sphere. Where the edges of the rectangle are great circle arcs.
    ///
    /// # Arguments
    ///
    /// * `pointing` - A vector pointing to the center of the rectangle.
    /// * `rotation` - Rotation of the center of the rectangle in radians.
    /// * `lon_width` - If the rotation is 0, this defines the width of the rectangle
    ///   longitudinally in radians.
    /// * `lat_width` - If the rotation is 0, this defines the width of the rectangle
    ///   latitudinally in radians.
    pub fn new(
        pointing: Vector<Equatorial>,
        rotation: f64,
        lon_width: f64,
        lat_width: f64,
    ) -> Self {
        // Rotate the Z axis to match the defined rotation angle, this vector is not
        // orthogonal to the pointing vector, but is in the correct plane of the final
        // up vector.
        let up_vec = &Vector::new([0.0, 0.0, 1.0]).rotate_around(pointing, -rotation);

        // construct the vector orthogonal to the pointing and rotate z axis vectors.
        // left = cross(up, pointing)
        let left_vec = pointing.cross(up_vec);

        // Given the new left vector, and the existing orthogonal pointing vector,
        // construct a new up vector which is in the same plane as it was before, but
        // now orthogonal to the two existing vectors.
        // up = cross(pointing, left)
        let up_vec = pointing.cross(&left_vec);

        // These have to be enumerated in clockwise order for the pointing calculation
        // to be correct.
        let n1 = left_vec.rotate_around(up_vec, -lon_width / 2.0);
        let n2 = up_vec.rotate_around(left_vec, lat_width / 2.0);
        let n3 = (-left_vec).rotate_around(up_vec, lon_width / 2.0);
        let n4 = (-up_vec).rotate_around(left_vec, -lat_width / 2.0);

        // construct the 4 normal vectors
        Self {
            edge_normals: [n1, n2, n3, n4],
        }
    }

    /// Construct the patch from the 4 corners of the field of view.
    /// The corners have to be provided in order, either clockwise or
    /// counter-clockwise.
    ///
    /// This only works for fields of view where the largest angle is less than 180
    /// degrees, if the field is wider than that, this will flip the field in the other
    /// direction.
    ///
    /// # Arguments
    ///
    /// * `corners` - 4 vectors which define the corners of the fov, must be provided
    ///   in order.
    /// * `expand_angle` - Expand the fov by the specified angle away from the center,
    ///   units of radians.
    ///
    pub fn from_corners(corners: [Vector<Equatorial>; 4], expand_angle: f64) -> Self {
        // compute the pointing vector from the corners
        let pointing = {
            let mut point: Vector<Equatorial> = [0.0; 3].into();
            corners.iter().for_each(|c| point += c);
            point.normalize()
        };

        let n1 = corners[0].cross(&corners[1]).normalize();
        let n2 = corners[1].cross(&corners[2]).normalize();
        let n3 = corners[2].cross(&corners[3]).normalize();
        let n4 = corners[3].cross(&corners[0]).normalize();

        let mut edge_normals = [n1, n2, n3, n4];

        // check the direction of the normals, if they are too far away from the
        // pointing vector, then we need to flip the signs.
        if n1.dot(&pointing).is_sign_negative() {
            edge_normals.iter_mut().for_each(|x| *x = x.neg());
            edge_normals.reverse();
        }

        // move the normals away from the pointing vector by the specified angle.
        for v in edge_normals.iter_mut() {
            let rot_vec = v.cross(&pointing);
            *v = v.rotate_around(rot_vec, expand_angle);
        }

        Self { edge_normals }
    }

    /// Return the 4 corners of the patch.
    pub fn corners(&self) -> [Vector<Equatorial>; 4] {
        (0..4)
            .map(|idx| {
                let idy = (idx + 1) % 4;
                let a = self.edge_normals[idx];
                let b = self.edge_normals[idy];
                a.cross(&b)
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    /// Latitudinal width of the patch, the assumes the patch is rectangular.
    pub fn lat_width(&self) -> f64 {
        let pointing = self.pointing();
        2.0 * (FRAC_PI_2 - pointing.angle(&self.edge_normals[1]))
    }

    /// Longitudinal width of the patch, the assumes the patch is rectangular.
    pub fn lon_width(&self) -> f64 {
        let pointing = self.pointing();
        2.0 * (FRAC_PI_2 - pointing.angle(&self.edge_normals[0]))
    }
}

impl<const D: usize> SkyPatch for SphericalPolygon<D> {
    /// Is the vector inside of the polygon.
    fn contains(&self, obs_to_obj: &Vector<Equatorial>) -> Contains {
        let mut closest_edge = f64::NEG_INFINITY;
        for normal in self.edge_normals.iter() {
            let d = obs_to_obj.dot(normal);
            if d.is_nan() {
                return Contains::Outside(d);
            }
            // of all the edges where d is outside, we need to find the maximum
            // distance one. This will be the *minimum* distance which the
            // object must move to be inside of the patch.
            if d.is_sign_negative() && d.abs() > closest_edge {
                closest_edge = d.abs();
            }
        }

        match closest_edge {
            x if x.is_finite() => Contains::Outside(x.min(obs_to_obj.norm())),
            _ => Contains::Inside,
        }
    }

    fn pointing(&self) -> Vector<Equatorial> {
        let mut point: Vector<Equatorial> = [0.0; 3].into();

        for (idx, idy) in (0..D).zip(1..D) {
            let v = self.edge_normals[idx]
                .cross(&self.edge_normals[idy])
                .normalize();
            point += &v;
        }

        let v = self.edge_normals[D - 1]
            .cross(&self.edge_normals[0])
            .normalize();
        point += &v;

        point.normalize()
    }
}

/// Represent a cone on a sphere.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SphericalCone {
    /// Unit vector which defines the direction of the cone.
    pointing: Vector<Equatorial>,

    /// Size of the cone in degrees.
    pub angle: f64,
}

impl SphericalCone {
    /// Construct a new `SphericalCone` given the central vector and the angle of the
    /// cone.
    pub fn new(pointing: &Vector<Equatorial>, angle: f64) -> Self {
        let pointing = pointing.normalize();
        Self { pointing, angle }
    }
}

impl SkyPatch for SphericalCone {
    /// Is the vector inside of the cone.
    fn contains(&self, obs_to_obj: &Vector<Equatorial>) -> Contains {
        let dist = self.pointing.dot(&obs_to_obj.normalize()).acos().abs();
        match dist {
            // if d is less than the angle, it is inside cone
            d if d <= self.angle => Contains::Inside,

            // outside of the cone, but how badly?
            d => {
                let r = obs_to_obj.norm();
                let min_angle = (d - self.angle).abs();

                // The minimum distance required for the object to be within the cone
                // of shame.
                match min_angle {
                    // if the min angle is more than 90 degrees, then the object
                    // can be visible by moving on top of the observer, so going r dist
                    theta if theta > FRAC_PI_2 => Contains::Outside(r),

                    // if the min angle is less than 90 degrees, then the object can
                    // move directly toward the edge of the cone, which is a right
                    // angle triangle, where the hypotenuse is r
                    theta => Contains::Outside(theta.sin() * r),
                }
            }
        }
    }

    fn pointing(&self) -> Vector<Equatorial> {
        self.pointing
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rectangular_patch() {
        let rot = (45_f64).to_radians();
        let inside = [1.0, 0.01, 0.01].into();
        let outside = [1.0, 0.1, 0.0].into();
        let just_inside = [1.0, (0.05_f64).sin() * 0.99, (0.05_f64).sin() * 0.99].into();
        let just_outside = [1.0, (0.05_f64).sin() * 1.01, (0.05_f64).sin() * 1.01].into();
        let fov = OnSkyRectangle::new([1.0, 0.0, 0.0].into(), 0.0, 0.1, 0.1);
        let fov_rot = OnSkyRectangle::new([1.0, 0.0, 0.0].into(), rot, 0.1, 0.1);

        assert!(fov.contains(&inside).is_inside());
        assert!(fov.contains(&just_inside).is_inside());
        assert!(!fov.contains(&outside).is_inside());
        assert!(!fov.contains(&just_outside).is_inside());

        assert!(fov_rot.contains(&inside).is_inside());
        assert!(!fov_rot.contains(&just_inside).is_inside());
        assert!((fov_rot.pointing() - Vector::new([1.0, 0.0, 0.0])).norm() < 1e-10);
    }

    #[test]
    fn test_rectangular_patch_latlon() {
        let rot = (45_f64).to_radians();
        let fov = OnSkyRectangle::new([1.0, 0.0, 0.0].into(), 0.0, 0.1, 0.2);
        let fov_rot = OnSkyRectangle::new([1.0, 0.0, 0.0].into(), rot, 0.1, 0.2);

        assert!((fov.lat_width() - 0.2).abs() < 1e-10);
        assert!((fov.lon_width() - 0.1).abs() < 1e-10);
        assert!((fov_rot.lat_width() - 0.2).abs() < 1e-10);
        assert!((fov_rot.lon_width() - 0.1).abs() < 1e-10);
    }
}
