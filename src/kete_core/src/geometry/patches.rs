// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-FileCopyrightText: 2025 California Institute of Technology
// SPDX-License-Identifier: BSD-3-Clause

//! Basic Geometric shapes on the surface of a sphere

use crate::errors::{Error, KeteResult};
use crate::frames::{Equatorial, Vector};
use smallvec::SmallVec;
use std::{
    f64::consts::{FRAC_PI_2, PI},
    ops::Neg,
};

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
    #[must_use]
    pub fn is_inside(&self) -> bool {
        matches!(self, Self::Inside)
    }
}

/// Given an iterable of [`Contains`], find the closest one to being Inside.
///
/// Returns the index of the first [`Contains::Inside`], or else the index and
/// distance of the first smallest [`Contains::Outside`].
pub(crate) fn closest_inside(contains: impl IntoIterator<Item = Contains>) -> (usize, Contains) {
    let mut best = (usize::MAX, f64::INFINITY);
    for (idx, con) in contains.into_iter().enumerate() {
        match con {
            Contains::Inside => return (idx, Contains::Inside),
            Contains::Outside(d) => {
                if d < best.1 {
                    best = (idx, d);
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

/// A polygon on the sky with great circle edges, convex or not.
///
/// Each edge lies on a plane through the center of the sphere. The polygon
/// keeps the unit normal of each plane, pointing toward the inside.
///
/// # Convex polygons
///
/// A direction is inside a convex polygon if its dot product with every normal
/// is positive or zero. For an object outside, [`SkyPatch::contains`] gives the
/// largest distance from the object to the plane of an edge it is outside of,
/// at most the distance `r` of the object. Every rectangle is convex.
///
/// # Non-convex polygons
///
/// The corners of a non-convex polygon lie in the open hemisphere around their
/// center, the normalized sum of the corners. A direction is inside if it is in
/// that hemisphere and the polygon winds around it: the signed angles that the
/// edges subtend, seen from the direction, sum to a full turn. For an object
/// outside, [`SkyPatch::contains`] gives the distance from the object to the
/// nearest point of the cone of directions inside. This is the distance to the
/// plane of the nearest edge, or to the ray through the nearest corner, at most
/// `r`.
///
/// Both distances are at most the distance the object must move to be inside.
#[derive(Debug, Clone)]
pub struct SphericalPolygon {
    /// Unit normals of the planes of the edges, pointing toward the inside.
    /// Up to four, the case of every survey FOV, are stored without a heap
    /// allocation.
    pub(crate) edge_normals: SmallVec<[Vector<Equatorial>; 4]>,

    /// Unit vector of the center of the corners, for a non-convex polygon;
    /// `None` if it is convex.
    center: Option<Vector<Equatorial>>,
}

impl SphericalPolygon {
    /// The convex polygon whose edges lie on the planes with the normals
    /// `edge_normals`, given in order and pointing toward the inside. The normals
    /// are kept as unit vectors.
    #[must_use]
    pub fn from_normals(edge_normals: &[Vector<Equatorial>]) -> Self {
        Self {
            edge_normals: edge_normals.iter().map(Vector::normalize).collect(),
            center: None,
        }
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
    ///
    /// The rotation is measured from the direction to the celestial north pole. That
    /// direction is undefined when `pointing` is at either celestial pole, so such a
    /// pointing is an error. Construct the rectangle from its corners instead.
    ///
    /// # Errors
    /// Returns [`Error::ValueError`] if `pointing` is not finite, is zero, or points
    /// at a celestial pole of the equatorial frame.
    pub fn new(
        pointing: Vector<Equatorial>,
        rotation: f64,
        lon_width: f64,
        lat_width: f64,
    ) -> KeteResult<Self> {
        // Rotate the Z axis to match the defined rotation angle, this vector is not
        // orthogonal to the pointing vector, but is in the correct plane of the final
        // up vector.
        let up_vec = &Vector::new([0.0, 0.0, 1.0]).rotate_around(pointing, -rotation);

        // construct the vector orthogonal to the pointing and rotate z axis vectors.
        // left = cross(up, pointing)
        let left_vec = pointing.cross(up_vec);
        if !left_vec.is_finite() || left_vec.norm() <= f64::EPSILON * pointing.norm() {
            return Err(Error::ValueError(
                "A rectangle FOV cannot point at a celestial pole, where its rotation \
                 from celestial north is undefined; construct it from its corners instead."
                    .into(),
            ));
        }

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

        Ok(Self::from_normals(&[n1, n2, n3, n4]))
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
    #[must_use]
    pub fn from_corners(corners: [Vector<Equatorial>; 4], expand_angle: f64) -> Self {
        // compute the pointing vector from the corners
        let pointing = {
            let mut point: Vector<Equatorial> = [0.0; 3].into();
            for c in corners {
                point += &c;
            }
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
            for x in &mut edge_normals {
                *x = x.neg();
            }
            edge_normals.reverse();
        }

        // move the normals away from the pointing vector by the specified angle.
        for v in &mut edge_normals {
            let rot_vec = v.cross(&pointing);
            *v = v.rotate_around(rot_vec, expand_angle);
        }

        Self::from_normals(&edge_normals)
    }

    /// The polygon with `corners`, given in order around it, clockwise or
    /// counterclockwise. The polygon can be convex or not.
    ///
    /// # Errors
    /// [`Error::ValueError`] if there are fewer than 3 corners, if a corner is
    /// zero or not finite, if two consecutive corners are parallel, if a corner
    /// is 89.9 degrees or more from the center of the corners, if three
    /// consecutive corners lie on one great circle, or if two edges
    /// cross.
    pub fn try_from_corners(corners: &[Vector<Equatorial>]) -> KeteResult<Self> {
        if corners.len() < 3 {
            return Err(Error::ValueError(format!(
                "A polygon needs at least 3 corners, found {}.",
                corners.len()
            )));
        }
        let units = corners
            .iter()
            .map(|c| {
                let norm = c.norm();
                if norm > 0.0 && norm.is_finite() {
                    Ok(c.normalize())
                } else {
                    Err(Error::ValueError(
                        "A polygon corner is zero or not finite.".into(),
                    ))
                }
            })
            .collect::<KeteResult<Vec<_>>>()?;
        let mut sum: Vector<Equatorial> = [0.0; 3].into();
        for c in &units {
            sum += c;
        }
        // The cosine of 89.9 degrees: a corner closer to 90 degrees from the
        // center projects too far for a reliable test.
        let min_cos = 1.745e-3;
        if sum.norm() == 0.0 || units.iter().any(|c| c.dot(&sum.normalize()) <= min_cos) {
            return Err(Error::ValueError(
                "The polygon corners must lie within 89.9 degrees of their center.".into(),
            ));
        }
        let center = sum.normalize();
        let seed: Vector<Equatorial> = if center[0].abs() < 0.9 {
            [1.0, 0.0, 0.0].into()
        } else {
            [0.0, 1.0, 0.0].into()
        };
        let east = center.cross(&seed).normalize();
        let north = center.cross(&east);
        let projected: Vec<[f64; 2]> = units
            .iter()
            .map(|corner| {
                let height = corner.dot(&center);
                [corner.dot(&east) / height, corner.dot(&north) / height]
            })
            .collect();

        let n = projected.len();
        for i in 0..n {
            for j in i + 1..n {
                // Adjacent edges share a corner and do not count as a crossing.
                if j == i + 1 || (i == 0 && j == n - 1) {
                    continue;
                }
                if segments_cross(
                    projected[i],
                    projected[(i + 1) % n],
                    projected[j],
                    projected[(j + 1) % n],
                ) {
                    return Err(Error::ValueError(format!(
                        "Polygon edges {i} and {j} cross; the corners must go around \
                         the polygon in order."
                    )));
                }
            }
        }

        // With the corners counterclockwise in the tangent plane, the cross
        // product of consecutive corners points toward the inside.
        let area: f64 = (0..n)
            .map(|i| {
                let (start, end) = (projected[i], projected[(i + 1) % n]);
                start[0] * end[1] - end[0] * start[1]
            })
            .sum();
        let sign = if area < 0.0 { -1.0 } else { 1.0 };
        let edge_normals = (0..n)
            .map(|i| {
                let normal = units[i].cross(&units[(i + 1) % n]) * sign;
                if normal.norm() > 0.0 {
                    Ok(normal.normalize())
                } else {
                    Err(Error::ValueError(format!(
                        "Polygon corners {i} and {} are parallel.",
                        (i + 1) % n
                    )))
                }
            })
            .collect::<KeteResult<Vec<_>>>()?;
        let convex = edge_normals
            .iter()
            .all(|normal| units.iter().all(|c| normal.dot(c) >= -1e-12));

        // A corner between two edges on almost the same great circle adds
        // nothing to the shape, and cannot be rebuilt from the edges.
        for i in 0..n {
            if edge_normals[(i + n - 1) % n].cross(&edge_normals[i]).norm() < MIN_CORNER_SIN {
                return Err(Error::ValueError(format!(
                    "Polygon corner {i} lies on the great circle of its neighbors."
                )));
            }
        }
        Ok(Self {
            edge_normals: edge_normals.into(),
            center: (!convex).then_some(center),
        })
    }

    /// The corners of the polygon as unit vectors, in order.
    ///
    /// Corner `i` is where the planes of edges `i - 1` and `i` meet. For a
    /// polygon from [`Self::try_from_corners`], corner `i` is the direction of
    /// the given corner `i`.
    #[must_use]
    pub fn corners(&self) -> Vec<Vector<Equatorial>> {
        (0..self.edge_normals.len())
            .map(|i| self.corner(i))
            .collect()
    }

    /// Corner `i` as a unit vector; see [`Self::corners`].
    fn corner(&self, i: usize) -> Vector<Equatorial> {
        let n = self.edge_normals.len();
        let corner = self.edge_normals[(i + n - 1) % n]
            .cross(&self.edge_normals[i])
            .normalize();
        // The cross product points to the opposite side of the sphere at a reflex
        // corner of a non-convex polygon, and at every corner of a convex polygon
        // whose edges go clockwise. A corner of a convex polygon is strictly
        // inside the next edge, which settles the sign there.
        let reference = match self.center {
            Some(center) => center,
            None => self.edge_normals[(i + 1) % n],
        };
        if corner.dot(&reference) < 0.0 {
            -corner
        } else {
            corner
        }
    }

    /// The edge normals and the center of a non-convex polygon, for storage.
    pub(crate) fn parts(&self) -> (&[Vector<Equatorial>], Option<Vector<Equatorial>>) {
        (&self.edge_normals[..], self.center)
    }

    /// The polygon from the parts that [`Self::parts`] gives.
    pub(crate) fn from_parts(
        edge_normals: Vec<Vector<Equatorial>>,
        center: Option<Vector<Equatorial>>,
    ) -> Self {
        Self {
            edge_normals: edge_normals.into(),
            center,
        }
    }

    /// Whether the polygon is convex.
    #[must_use]
    pub fn is_convex(&self) -> bool {
        self.center.is_none()
    }

    /// Latitudinal width of the patch, the assumes the patch is rectangular.
    #[must_use]
    pub fn lat_width(&self) -> f64 {
        let pointing = self.pointing();
        2.0 * (FRAC_PI_2 - pointing.angle(&self.edge_normals[1]))
    }

    /// Longitudinal width of the patch, the assumes the patch is rectangular.
    #[must_use]
    pub fn lon_width(&self) -> f64 {
        let pointing = self.pointing();
        2.0 * (FRAC_PI_2 - pointing.angle(&self.edge_normals[0]))
    }
}

impl SkyPatch for SphericalPolygon {
    /// Whether the vector is inside the polygon, and if not, a lower bound on
    /// the distance it must move to be inside; see [`SphericalPolygon`].
    fn contains(&self, obs_to_obj: &Vector<Equatorial>) -> Contains {
        if let Some(center) = &self.center {
            return self.non_convex_contains(obs_to_obj, center);
        }
        // A fixed size array lets the compiler unroll the loop for the four
        // edges of a rectangle, the case of every survey FOV.
        if let Ok(normals) = <&[Vector<Equatorial>; 4]>::try_from(self.edge_normals.as_slice()) {
            return convex_contains(normals, obs_to_obj);
        }
        convex_contains(&self.edge_normals, obs_to_obj)
    }

    fn pointing(&self) -> Vector<Equatorial> {
        if let Some(center) = self.center {
            return center;
        }
        let mut point: Vector<Equatorial> = [0.0; 3].into();
        for corner in self.corners() {
            point += &corner;
        }
        point.normalize()
    }
}

impl SphericalPolygon {
    /// [`SkyPatch::contains`] for a non-convex polygon with center `center`.
    ///
    /// Kept out of line: inlined, its register use would make every call of
    /// [`SkyPatch::contains`] save and restore registers, convex or not.
    #[inline(never)]
    fn non_convex_contains(
        &self,
        obs_to_obj: &Vector<Equatorial>,
        center: &Vector<Equatorial>,
    ) -> Contains {
        let r = obs_to_obj.norm();
        if r.is_nan() {
            return Contains::Outside(r);
        }
        if r == 0.0 {
            return Contains::Inside;
        }
        let dir = obs_to_obj.normalize();
        let units = self.corners();
        let n = units.len();
        if dir.dot(center) > 0.0 {
            // The signed angle of each edge seen from `dir`: between the great
            // circles from `dir` through the two ends of the edge.
            let turn: f64 = (0..n)
                .map(|i| {
                    let (a, b) = (&units[i], &units[(i + 1) % n]);
                    let sin = dir.dot(&a.cross(b));
                    let cos = a.dot(b) - dir.dot(a) * dir.dot(b);
                    sin.atan2(cos)
                })
                .sum();
            // A full turn is inside, and no turn is outside.
            if turn.abs() > PI {
                return Contains::Inside;
            }
        }
        let theta = (0..n)
            .map(|i| arc_angle(&dir, &units[i], &units[(i + 1) % n]))
            .fold(f64::INFINITY, f64::min);
        // The distance from the object to the plane of the nearest edge, or to
        // the ray through the nearest corner.
        if theta >= FRAC_PI_2 {
            Contains::Outside(r)
        } else {
            Contains::Outside(r * theta.sin())
        }
    }
}

/// [`SkyPatch::contains`] for a convex polygon with the edge normals
/// `normals`.
#[inline(always)]
fn convex_contains<'a>(
    normals: impl IntoIterator<Item = &'a Vector<Equatorial>>,
    obs_to_obj: &Vector<Equatorial>,
) -> Contains {
    let mut closest_edge = f64::NEG_INFINITY;
    for normal in normals {
        let d = obs_to_obj.dot(normal);
        if d.is_nan() {
            return Contains::Outside(d);
        }
        // Of all the edges the object is outside of, the farthest plane bounds
        // the distance the object must move to be inside.
        if d.is_sign_negative() && d.abs() > closest_edge {
            closest_edge = d.abs();
        }
    }
    match closest_edge {
        x if x.is_finite() => Contains::Outside(x.min(obs_to_obj.norm())),
        _ => Contains::Inside,
    }
}

/// The sine of the smallest angle between the planes of two consecutive edges
/// of a polygon from [`SphericalPolygon::try_from_corners`].
const MIN_CORNER_SIN: f64 = 1e-6;

/// Whether the segments `a`-`b` and `c`-`d` of a plane cross.
///
/// Segments that only touch, or that are collinear, do not count as a
/// crossing.
fn segments_cross(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> bool {
    let orient = |p: [f64; 2], q: [f64; 2], r: [f64; 2]| {
        (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    };
    let (o1, o2) = (orient(a, b, c), orient(a, b, d));
    let (o3, o4) = (orient(c, d, a), orient(c, d, b));
    o1 * o2 < 0.0 && o3 * o4 < 0.0
}

/// The angle from the unit vector `p` to the nearest point of the minor great
/// circle arc from the unit vector `a` to the unit vector `b`.
fn arc_angle(p: &Vector<Equatorial>, a: &Vector<Equatorial>, b: &Vector<Equatorial>) -> f64 {
    let ends = p.angle(a).min(p.angle(b));
    let normal = a.cross(b);
    if normal.norm() == 0.0 {
        return ends;
    }
    let normal = normal.normalize();
    let off_plane = p.dot(&normal);
    let in_plane = *p - normal * off_plane;
    if in_plane.norm() == 0.0 {
        return FRAC_PI_2;
    }
    // The nearest point of the full great circle is on the arc when it lies
    // between the two ends.
    let foot = in_plane.normalize();
    if a.cross(&foot).dot(&normal) >= 0.0 && foot.cross(b).dot(&normal) >= 0.0 {
        off_plane.abs().atan2(in_plane.norm())
    } else {
        ends
    }
}

/// Represent a cone on a sphere.
#[derive(Debug, Clone)]
pub struct SphericalCone {
    /// Unit vector which defines the direction of the cone.
    pub(crate) pointing: Vector<Equatorial>,

    /// Angle from the central pointing vector to the edge of the cone, in radians.
    angle: f64,

    /// Cosine of `angle`.
    cos_angle: f64,

    /// Sine of `angle`.
    sin_angle: f64,
}

impl SphericalCone {
    /// Construct a new `SphericalCone` given the central vector and the angle from
    /// the central vector to the edge of the cone in radians.
    #[must_use]
    pub fn new(pointing: &Vector<Equatorial>, angle: f64) -> Self {
        Self::from_parts(pointing.normalize(), angle)
    }

    /// The cone with the unit vector `pointing`, used as given.
    pub(crate) fn from_parts(pointing: Vector<Equatorial>, angle: f64) -> Self {
        let (sin_angle, cos_angle) = angle.sin_cos();
        Self {
            pointing,
            angle,
            cos_angle,
            sin_angle,
        }
    }

    /// Angle from the central pointing vector to the edge of the cone, in radians.
    #[must_use]
    pub fn angle(&self) -> f64 {
        self.angle
    }
}

impl SkyPatch for SphericalCone {
    /// Is the vector inside of the cone.
    ///
    /// The angle `theta` from the pointing is handled through its sine and cosine.
    /// The sine comes from the cross product, which keeps it exact for small
    /// angles, and no inverse trigonometric function is needed.
    fn contains(&self, obs_to_obj: &Vector<Equatorial>) -> Contains {
        let r = obs_to_obj.norm();
        if r == 0.0 || self.angle >= PI {
            return Contains::Inside;
        }
        let cos_theta = obs_to_obj.dot(&self.pointing) / r;
        let sin_theta = obs_to_obj.cross(&self.pointing).norm() / r;

        // sin(angle - theta), which is not negative when theta is at most angle.
        let sin_inside = self.sin_angle * cos_theta - self.cos_angle * sin_theta;
        if sin_inside >= 0.0 {
            return Contains::Inside;
        }

        // Outside by theta - angle. Past a right angle the object is nearest the
        // cone by moving on top of the observer, a distance r. Otherwise it can
        // move directly toward the edge of the cone, which is a right angle
        // triangle with hypotenuse r.
        if cos_theta * self.cos_angle + sin_theta * self.sin_angle < 0.0 {
            Contains::Outside(r)
        } else {
            Contains::Outside(-sin_inside * r)
        }
    }

    fn pointing(&self) -> Vector<Equatorial> {
        self.pointing
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The cone check as it was written with `Vector::angle`, for comparison.
    fn cone_contains_by_angle(cone: &SphericalCone, obs_to_obj: &Vector<Equatorial>) -> Contains {
        let theta = cone.pointing.angle(obs_to_obj);
        if theta <= cone.angle() {
            return Contains::Inside;
        }
        let r = obs_to_obj.norm();
        match theta - cone.angle() {
            outside if outside > FRAC_PI_2 => Contains::Outside(r),
            outside => Contains::Outside(outside.sin() * r),
        }
    }

    #[test]
    fn cone_matches_angle_form() {
        let pointing = Vector::<Equatorial>::from_ra_dec(0.3, 0.2);
        for angle in [1e-7, 1e-3, 0.05, 1.0, 2.0] {
            let cone = SphericalCone::new(&pointing, angle);
            for i in 0..72 {
                for j in 0..37 {
                    let dir = Vector::<Equatorial>::from_ra_dec(
                        f64::from(i) * 0.0875,
                        -1.55 + f64::from(j) * 0.0861,
                    );
                    let obs_to_obj = dir * 1.7;
                    // The angle form is not exact within about 1e-8 radians of
                    // the edge, so the decision there is not compared.
                    if (pointing.angle(&dir) - angle).abs() < 1e-6 {
                        continue;
                    }
                    match (
                        cone.contains(&obs_to_obj),
                        cone_contains_by_angle(&cone, &obs_to_obj),
                    ) {
                        (Contains::Inside, Contains::Inside) => {}
                        (Contains::Outside(a), Contains::Outside(b)) => {
                            assert!((a - b).abs() < 1e-9, "angle {angle}: {a} vs {b}");
                        }
                        (a, b) => panic!("angle {angle}, ({i}, {j}): {a:?} vs {b:?}"),
                    }
                }
            }
        }
    }

    #[test]
    fn cone_small_angles() {
        // An object 1e-10 radians outside a cone of 1e-9 radians, where the cosine
        // of either angle rounds to one.
        let pointing = Vector::<Equatorial>::new([1.0, 0.0, 0.0]);
        let cone = SphericalCone::new(&pointing, 1e-9);
        let outside = Vector::<Equatorial>::new([2.0, 2.0 * 1.1e-9, 0.0]);
        let inside = Vector::<Equatorial>::new([2.0, 2.0 * 0.9e-9, 0.0]);
        assert!(cone.contains(&inside).is_inside());
        match cone.contains(&outside) {
            Contains::Outside(d) => assert!((d - 2.0e-10).abs() < 1e-18, "{d}"),
            Contains::Inside => panic!("expected outside"),
        }
        assert!(
            cone.contains(&Vector::<Equatorial>::new([0.0, 0.0, 0.0]))
                .is_inside()
        );
    }

    /// A rectangle pointed at a celestial pole has no defined rotation and is an
    /// error. Just off the pole it builds and contains its center.
    #[test]
    fn rectangle_at_pole_is_error() {
        for z in [1.0, -1.0] {
            assert!(SphericalPolygon::new([0.0, 0.0, z].into(), 0.3, 0.1, 0.1).is_err());
        }
        assert!(SphericalPolygon::new([0.0, 0.0, 0.0].into(), 0.0, 0.1, 0.1).is_err());
        let near: Vector<Equatorial> = [1e-6, 0.0, 1.0].into();
        let fov = SphericalPolygon::new(near, 0.3, 0.1, 0.1).unwrap();
        assert!(fov.contains(&near).is_inside());
    }

    #[test]
    fn test_rectangular_patch() {
        let rot = (45_f64).to_radians();
        let inside = [1.0, 0.01, 0.01].into();
        let outside = [1.0, 0.1, 0.0].into();
        let just_inside = [1.0, (0.05_f64).sin() * 0.99, (0.05_f64).sin() * 0.99].into();
        let just_outside = [1.0, (0.05_f64).sin() * 1.01, (0.05_f64).sin() * 1.01].into();
        let fov = SphericalPolygon::new([1.0, 0.0, 0.0].into(), 0.0, 0.1, 0.1).unwrap();
        let fov_rot = SphericalPolygon::new([1.0, 0.0, 0.0].into(), rot, 0.1, 0.1).unwrap();

        assert!(fov.contains(&inside).is_inside());
        assert!(fov.contains(&just_inside).is_inside());
        assert!(!fov.contains(&outside).is_inside());
        assert!(!fov.contains(&just_outside).is_inside());

        assert!(fov_rot.contains(&inside).is_inside());
        assert!(!fov_rot.contains(&just_inside).is_inside());
        assert!((fov_rot.pointing() - Vector::new([1.0, 0.0, 0.0])).norm() < 1e-10);
    }

    /// The distance outside a rectangle is the distance to the plane of its edge,
    /// for any length of the pointing vector and any declination.
    #[test]
    fn rectangle_distance_outside() {
        let (half, offset) = (0.05_f64, 0.2_f64);
        for (length, dec) in [(1.0, 0.0_f64), (3.0, 0.0), (0.01, 0.0), (1.0, 1.0)] {
            let pointing = Vector::new([dec.cos(), 0.0, dec.sin()]) * length;
            let fov = SphericalPolygon::new(pointing, 0.0, 2.0 * half, 2.0 * half).unwrap();
            // Two au away, `offset` radians toward the north of the pointing.
            let point = Vector::new([(dec + offset).cos(), 0.0, (dec + offset).sin()]) * 2.0;
            let Contains::Outside(dist) = fov.contains(&point) else {
                panic!("the point is outside");
            };
            assert!((dist - 2.0 * (offset - half).sin()).abs() < 1e-12);
        }
    }

    #[test]
    fn test_rectangular_patch_latlon() {
        let rot = (45_f64).to_radians();
        let fov = SphericalPolygon::new([1.0, 0.0, 0.0].into(), 0.0, 0.1, 0.2).unwrap();
        let fov_rot = SphericalPolygon::new([1.0, 0.0, 0.0].into(), rot, 0.1, 0.2).unwrap();

        assert!((fov.lat_width() - 0.2).abs() < 1e-10);
        assert!((fov.lon_width() - 0.1).abs() < 1e-10);
        assert!((fov_rot.lat_width() - 0.2).abs() < 1e-10);
        assert!((fov_rot.lon_width() - 0.1).abs() < 1e-10);
    }

    /// A direction from the sky angles `x` and `y`, in radians, near +x.
    fn dir(x: f64, y: f64) -> Vector<Equatorial> {
        Vector::new([1.0, x.tan(), y.tan()]).normalize()
    }

    /// A U shape, open toward +y: two arms joined at the bottom.
    fn u_shape() -> SphericalPolygon {
        let deg = 1_f64.to_radians();
        SphericalPolygon::try_from_corners(&[
            dir(-3.0 * deg, -2.0 * deg),
            dir(3.0 * deg, -2.0 * deg),
            dir(3.0 * deg, 2.0 * deg),
            dir(deg, 2.0 * deg),
            dir(deg, 0.0),
            dir(-deg, 0.0),
            dir(-deg, 2.0 * deg),
            dir(-3.0 * deg, 2.0 * deg),
        ])
        .unwrap()
    }

    /// The notch of a non-convex polygon is outside; its arms and base are
    /// inside.
    #[test]
    fn non_convex_polygon_contains() {
        let poly = u_shape();
        let deg = 1_f64.to_radians();
        for (x, y) in [(-2.0, 1.0), (2.0, 1.0), (0.0, -1.0), (2.9, -1.9)] {
            assert!(poly.contains(&dir(x * deg, y * deg)).is_inside(), "{x} {y}");
        }
        for (x, y) in [(0.0, 1.0), (0.0, 1.9), (4.0, 0.0), (0.0, -3.0)] {
            assert!(
                !poly.contains(&dir(x * deg, y * deg)).is_inside(),
                "{x} {y}"
            );
        }
        assert!(!poly.contains(&Vector::new([-1.0, 0.0, 0.0])).is_inside());
    }

    /// A convex polygon built from corners agrees with the rectangle patch,
    /// in the inside test and in the plane distance outside.
    #[test]
    fn convex_polygon_matches_rectangle() {
        let corners = [
            dir(-0.05, -0.03),
            dir(0.05, -0.03),
            dir(0.05, 0.03),
            dir(-0.05, 0.03),
        ];
        let poly = SphericalPolygon::try_from_corners(&corners).unwrap();
        assert!(poly.is_convex());
        let rect = SphericalPolygon::from_corners(corners, 0.0);
        for i in -20..=20 {
            for j in -20..=20 {
                // The offsets keep the grid off the edges, where either answer
                // is acceptable.
                let v = dir(
                    f64::from(i) * 0.004 + 0.0011,
                    f64::from(j) * 0.0025 + 0.0007,
                );
                match (poly.contains(&(v * 2.0)), rect.contains(&(v * 2.0))) {
                    (Contains::Inside, Contains::Inside) => {}
                    (Contains::Outside(a), Contains::Outside(b)) => {
                        assert!((a - b).abs() < 1e-14, "{i} {j}: {a} {b}");
                    }
                    other => panic!("{i} {j}: {other:?}"),
                }
            }
        }
    }

    /// The outside distance is `r sin(theta)` for the angle `theta` to the
    /// nearest edge, which a dense sampling of the edges bounds from above.
    #[test]
    fn polygon_outside_distance() {
        let poly = u_shape();
        let deg = 1_f64.to_radians();
        let corners = poly.corners();
        assert!(!poly.is_convex());
        for (x, y, r) in [
            (0.0, 1.0, 2.0),
            (6.0, 0.5, 1.0),
            (0.0, -5.0, 3.0),
            (-2.0, 9.0, 1.0),
        ] {
            let target = dir(x * deg, y * deg) * r;
            let Contains::Outside(dist) = poly.contains(&target) else {
                panic!("{x} {y} inside");
            };
            let mut sampled = f64::INFINITY;
            for i in 0..corners.len() {
                let (a, b) = (corners[i], corners[(i + 1) % corners.len()]);
                for k in 0..=2000 {
                    let t = f64::from(k) / 2000.0;
                    let point = (a * (1.0 - t) + b * t).normalize();
                    sampled = sampled.min(target.angle(&point));
                }
            }
            let expected = r * sampled.min(FRAC_PI_2).sin();
            assert!(dist <= expected + 1e-12, "{x} {y}: {dist} > {expected}");
            assert!(dist > expected - 1e-6, "{x} {y}: {dist} << {expected}");
        }
    }

    /// The corners rebuilt from the edges are the directions of the corners as
    /// given, in the same order, for a convex and a non-convex polygon.
    #[test]
    fn corners_are_rebuilt() {
        let deg = 1_f64.to_radians();
        let given = [
            dir(-3.0 * deg, -2.0 * deg),
            dir(3.0 * deg, -2.0 * deg),
            dir(3.0 * deg, 2.0 * deg),
            dir(deg, 2.0 * deg),
            dir(deg, 0.0),
            dir(-deg, 0.0),
            dir(-deg, 2.0 * deg),
            dir(-3.0 * deg, 2.0 * deg),
        ];
        // Counterclockwise and clockwise, non-convex and convex.
        let mut reversed = given;
        reversed.reverse();
        for corners in [&given[..], &given[..3], &reversed[..], &reversed[5..]] {
            let poly = SphericalPolygon::try_from_corners(corners).unwrap();
            for (got, want) in poly.corners().iter().zip(corners) {
                assert!((*got - want.normalize()).norm() < 1e-15);
            }
            let mut sum: Vector<Equatorial> = [0.0; 3].into();
            for c in corners {
                sum += &c.normalize();
            }
            assert!(poly.pointing().dot(&sum.normalize()) > 0.99);
        }
        // Three corners on one great circle.
        let line = [
            dir(0.0, 0.0),
            dir(0.01, 0.0),
            dir(0.02, 0.0),
            dir(0.01, 0.01),
        ];
        assert!(SphericalPolygon::try_from_corners(&line).is_err());
    }

    /// Too few corners, corners beyond the hemisphere, and crossing edges are
    /// errors.
    #[test]
    fn invalid_polygons() {
        assert!(SphericalPolygon::try_from_corners(&[dir(0.0, 0.0), dir(0.1, 0.0)]).is_err());
        assert!(
            SphericalPolygon::try_from_corners(&[
                Vector::new([1.0, 0.0, 0.0]),
                Vector::new([0.0, 1.0, 0.0]),
                Vector::new([-1.0, 0.01, 0.0]),
            ])
            .is_err()
        );
        // A bow tie: the edges 0-1 and 2-3 cross.
        assert!(
            SphericalPolygon::try_from_corners(&[
                dir(-0.1, -0.1),
                dir(0.1, 0.1),
                dir(0.1, -0.1),
                dir(-0.1, 0.1)
            ])
            .is_err()
        );
    }
}
