// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Geometry: closed triangle meshes, faceted shape models, and patches on the
//! celestial sphere.

mod mesh;
mod patches;
mod shapes;

pub use self::mesh::{Edge, TriMesh};
pub(crate) use self::patches::closest_inside;
pub use self::patches::{Contains, SkyPatch, SphericalCone, SphericalPolygon};
pub use self::shapes::{ConvexShape, Facet, TriangleFacet, TriangleShape};
