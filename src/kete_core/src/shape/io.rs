//! Shape-model file I/O.
//!
//! Currently supports a minimal subset of the Wavefront OBJ format: lines
//! starting with `v` (vertex `x y z`) and `f` (triangular face with three
//! 1-based vertex indices).  Texture and normal indices in faces (`f
//! v/vt/vn`) are accepted; only the vertex index is used.  All other line
//! types are ignored.
//
// BSD 3-Clause License
//
// Copyright (c) 2026, Dar Dahlen
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

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use nalgebra::Vector3;

use crate::errors::{Error, KeteResult};
use crate::shape::Polyhedron;

/// Load a polyhedron from a Wavefront OBJ file.
///
/// Vertices are taken from `v` lines (in the file's coordinate system and
/// length units) and triangular faces from `f` lines (1-based vertex
/// indices).  Non-triangular faces, negative indices, and other OBJ
/// features (groups, materials, textures, normals) are not supported and
/// will produce an error or be silently ignored as documented.
///
/// `gm` is the body's gravitational parameter `G * M` in caller-chosen
/// units consistent with the vertex length units.
///
/// # Errors
/// Returns an error if the file cannot be opened or read, if vertex or
/// face lines are malformed, if non-triangular faces are present, or if
/// the resulting mesh is not a closed orientable manifold.
pub fn load_obj<P: AsRef<Path>>(path: P, gm: f64) -> KeteResult<Polyhedron> {
    let file =
        File::open(path.as_ref()).map_err(|e| Error::IOError(format!("opening OBJ: {e}")))?;
    let reader = BufReader::new(file);

    let mut vertices: Vec<Vector3<f64>> = Vec::new();
    let mut faces: Vec<[u32; 3]> = Vec::new();

    for (lineno, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| Error::IOError(format!("reading OBJ line: {e}")))?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let mut tokens = trimmed.split_ascii_whitespace();
        let Some(tag) = tokens.next() else { continue };
        match tag {
            "v" => {
                let coords: Vec<&str> = tokens.collect();
                if coords.len() < 3 {
                    return Err(Error::IOError(format!(
                        "OBJ line {}: vertex needs 3 coordinates",
                        lineno + 1
                    )));
                }
                let x: f64 = coords[0].parse().map_err(|e| {
                    Error::IOError(format!("OBJ line {}: bad vertex x: {e}", lineno + 1))
                })?;
                let y: f64 = coords[1].parse().map_err(|e| {
                    Error::IOError(format!("OBJ line {}: bad vertex y: {e}", lineno + 1))
                })?;
                let z: f64 = coords[2].parse().map_err(|e| {
                    Error::IOError(format!("OBJ line {}: bad vertex z: {e}", lineno + 1))
                })?;
                vertices.push(Vector3::new(x, y, z));
            }
            "f" => {
                let raw: Vec<&str> = tokens.collect();
                if raw.len() != 3 {
                    return Err(Error::IOError(format!(
                        "OBJ line {}: only triangular faces are supported \
                         (got {} vertices)",
                        lineno + 1,
                        raw.len()
                    )));
                }
                let mut idx = [0_u32; 3];
                for (k, token) in raw.iter().enumerate() {
                    // Accept "v", "v/vt", "v//vn", or "v/vt/vn"; only "v" is used.
                    let v_part = token.split('/').next().unwrap_or(token);
                    let i: i64 = v_part.parse().map_err(|e| {
                        Error::IOError(format!(
                            "OBJ line {}: bad face vertex index '{token}': {e}",
                            lineno + 1
                        ))
                    })?;
                    if i <= 0 {
                        return Err(Error::IOError(format!(
                            "OBJ line {}: non-positive vertex index {i} not supported",
                            lineno + 1
                        )));
                    }
                    idx[k] = u32::try_from(i - 1).map_err(|_| {
                        Error::IOError(format!(
                            "OBJ line {}: vertex index {i} too large",
                            lineno + 1
                        ))
                    })?;
                }
                faces.push(idx);
            }
            _ => {
                // Ignore everything else (vt, vn, g, o, mtllib, usemtl, s, ...).
            }
        }
    }

    Polyhedron::try_new(vertices, &faces, gm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_tmp(name: &str, body: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(name);
        let mut f = File::create(&p).unwrap();
        f.write_all(body.as_bytes()).unwrap();
        p
    }

    #[test]
    fn parses_minimal_tetrahedron() {
        // Regular tetrahedron centered at origin with outward CCW winding.
        let body = "\
            # tetra\n\
            v  1.0  1.0  1.0\n\
            v -1.0 -1.0  1.0\n\
            v -1.0  1.0 -1.0\n\
            v  1.0 -1.0 -1.0\n\
            f 1 3 2\n\
            f 1 2 4\n\
            f 1 4 3\n\
            f 2 3 4\n\
        ";
        let p = write_tmp("kete_shape_tetra.obj", body);
        let poly = load_obj(&p, 1.0).unwrap();
        assert_eq!(poly.n_vertices(), 4);
        assert_eq!(poly.n_faces(), 4);
        assert_eq!(poly.n_edges(), 6);
        assert!(poly.volume > 0.0);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn handles_face_index_with_slashes() {
        let body = "\
            v  1.0  1.0  1.0\n\
            v -1.0 -1.0  1.0\n\
            v -1.0  1.0 -1.0\n\
            v  1.0 -1.0 -1.0\n\
            f 1/1 3/3 2/2\n\
            f 1/1/1 2/2/1 4/4/1\n\
            f 1//1 4//1 3//1\n\
            f 2 3 4\n\
        ";
        let p = write_tmp("kete_shape_slash.obj", body);
        let poly = load_obj(&p, 1.0).unwrap();
        assert_eq!(poly.n_faces(), 4);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn rejects_non_triangle_face() {
        let body = "\
            v 0 0 0\n\
            v 1 0 0\n\
            v 1 1 0\n\
            v 0 1 0\n\
            f 1 2 3 4\n\
        ";
        let p = write_tmp("kete_shape_quad.obj", body);
        assert!(load_obj(&p, 1.0).is_err());
        let _ = std::fs::remove_file(&p);
    }
}
