//! Python bindings for constant-density polyhedron gravity.
use std::sync::Arc;

use kete_core::constants::{AU_KM, GMS};
use kete_core::forces::{GravParams, Orientation, Polyhedron, Shape};
use kete_core::geometry::TriMesh;
use kete_core::prelude::Error;
use nalgebra::{Matrix3, Vector3};
use pyo3::{PyResult, pyclass, pyfunction, pymethods};

use crate::spherical_harmonics::PySphericalHarmonics;

/// Build a validated mesh from vertex rows and face rows.
fn build_mesh(vertices: Vec<[f64; 3]>, faces: Vec<[u32; 3]>, scale: f64) -> PyResult<TriMesh> {
    let verts = vertices
        .into_iter()
        .map(|v| Vector3::from(v) * scale)
        .collect();
    Ok(TriMesh::new(verts, faces)?)
}

/// Gravity field of a constant-density polyhedron.
///
/// The closed-form field of Werner and Scheeres (1997). The mesh must be closed,
/// consistently wound counter-clockwise seen from outside, and free of degenerate
/// faces. It is used where it is placed: fields are evaluated at positions
/// relative to the mesh origin, and the center of mass is the volume centroid,
/// see :attr:`centroid`. Units are whatever the vertices and ``gm`` are given in: with
/// vertices in km and ``gm`` in km^3/s^2, accelerations are in km/s^2.
///
/// At and beyond :attr:`far_field_radius` (3 times :attr:`bounding_radius`) the field
/// comes from the polyhedron's exact spherical harmonic expansion, cut at the
/// lowest degree that matches the closed form to 1e-11 (acceleration) and 1e-9
/// (gradient) relative on that sphere; it is built on the first evaluation there.
/// The closed form costs a pass over every face and edge and loses relative
/// precision far away, where its sums cancel toward the point-mass value.
/// :meth:`without_far_field` returns the closed form at every distance.
///
/// Parameters
/// ----------
/// vertices :
///     Vertex positions, shape ``(n, 3)``.
/// faces :
///     Faces as 0-based vertex indices, shape ``(m, 3)``, counter-clockwise seen
///     from outside.
/// gm :
///     Gravitational parameter of the body, in the units of the vertices cubed per
///     time squared.
#[pyclass(module = "kete._core", name = "Polyhedron", frozen, from_py_object)]
#[derive(Clone, Debug)]
pub struct PyPolyhedron(pub Arc<Polyhedron>);

#[pymethods]
impl PyPolyhedron {
    /// Build the field of a mesh.
    #[new]
    pub fn new(vertices: Vec<[f64; 3]>, faces: Vec<[u32; 3]>, gm: f64) -> PyResult<Self> {
        let mesh = build_mesh(vertices, faces, 1.0)?;
        Ok(Self(Arc::new(Polyhedron::new(mesh, gm)?)))
    }

    /// Acceleration at a position relative to the mesh origin.
    ///
    /// Returns the acceleration and the solid angle the surface subtends at the
    /// position: ``4 pi`` inside the body and ``0`` outside.
    pub fn field(&self, pos: [f64; 3]) -> ([f64; 3], f64) {
        let (accel, omega) = self.0.field(&pos.into());
        (accel.into(), omega)
    }

    /// Acceleration, its gradient with respect to position (``grad[i][j]`` is
    /// ``d accel_i / d pos_j``), and the solid angle, at a position relative to the
    /// mesh origin.
    pub fn field_and_gradient(&self, pos: [f64; 3]) -> ([f64; 3], [[f64; 3]; 3], f64) {
        let (accel, grad, omega) = self.0.field_and_gradient(&pos.into());
        let rows = [0, 1, 2].map(|i| [grad[(i, 0)], grad[(i, 1)], grad[(i, 2)]]);
        (accel.into(), rows, omega)
    }

    /// Gravitational potential at a position relative to the mesh origin,
    /// positive, so that the acceleration is its gradient.
    pub fn potential(&self, pos: [f64; 3]) -> f64 {
        self.0.potential(&pos.into())
    }

    /// Gravitational parameter.
    #[getter]
    pub fn gm(&self) -> f64 {
        self.0.gm()
    }

    /// Distance from the mesh origin at and beyond which the far field is used, or
    /// None when it is turned off.
    #[getter]
    pub fn far_field_radius(&self) -> Option<f64> {
        self.0.far_field_radius()
    }

    /// The far field as a :class:`SphericalHarmonics` (built if needed), or None
    /// when it is turned off or no degree up to 40 met the tolerances, in which case
    /// the closed form is used everywhere.
    pub fn far_field(&self) -> Option<PySphericalHarmonics> {
        self.0
            .far_field()
            .map(|field| PySphericalHarmonics(Arc::new(field.clone())))
    }

    /// The same polyhedron using the closed form at every distance.
    pub fn without_far_field(&self) -> Self {
        Self(Arc::new((*self.0).clone().without_far_field()))
    }

    /// Enclosed volume.
    #[getter]
    pub fn volume(&self) -> f64 {
        self.0.mesh().volume()
    }

    /// Volume centroid, the center of mass at constant density, relative to the
    /// mesh origin.
    #[getter]
    pub fn centroid(&self) -> [f64; 3] {
        self.0.mesh().centroid().into()
    }

    /// Largest distance of a vertex from the mesh origin.
    #[getter]
    pub fn bounding_radius(&self) -> f64 {
        self.0.mesh().bounding_radius()
    }

    /// Vertices, shape ``(n, 3)``.
    #[getter]
    pub fn vertices(&self) -> Vec<[f64; 3]> {
        self.0
            .mesh()
            .vertices()
            .iter()
            .map(|v| (*v).into())
            .collect()
    }

    /// Faces as 0-based vertex indices, shape ``(m, 3)``.
    #[getter]
    pub fn faces(&self) -> Vec<[u32; 3]> {
        self.0.mesh().faces().to_vec()
    }

    /// String representation.
    pub fn __repr__(&self) -> String {
        format!(
            "Polyhedron({} vertices, {} faces, gm={})",
            self.0.mesh().vertices().len(),
            self.0.mesh().faces().len(),
            self.0.gm()
        )
    }
}

/// The body-frame orientation from exactly one of a CK frame and a fixed rotation.
pub(crate) fn orientation(
    frame_id: Option<i32>,
    rotation: Option<[[f64; 3]; 3]>,
) -> PyResult<Orientation> {
    match (frame_id, rotation) {
        (Some(frame_id), None) => Ok(Orientation::Ck { frame_id }),
        (None, Some(rows)) => {
            let rot = Matrix3::from_fn(|i, j| rows[i][j]);
            if (rot * rot.transpose() - Matrix3::identity()).abs().max() > 1e-9
                || rot.determinant() <= 0.0
            {
                Err(Error::ValueError(
                    "rotation must be a proper rotation matrix.".into(),
                ))?;
            }
            Ok(Orientation::Fixed(rot))
        }
        _ => Err(Error::ValueError(
            "Exactly one of frame_id and rotation must be given.".into(),
        ))?,
    }
}

/// Register a massive body whose gravity near it is a constant-density polyhedron.
///
/// Registered bodies are part of the force model when ``include_asteroids=True`` is
/// passed to :func:`propagate_n_body` or orbit fitting; the default force model is a
/// fixed list of planets and the Moon, which a registration does not change. Within
/// ``switch_radius`` of the body those functions use the polyhedron field of the
/// shape model in place of the point mass; beyond it, the point mass. The body must have an SPK ephemeris loaded; the shape
/// model's origin is placed at the body's SPK position, so vertices are given in
/// the body frame about that point. The point mass beyond ``switch_radius`` is also
/// at the SPK position, while the polyhedron's center of mass is its volume
/// centroid; if the two differ the force changes across the switch by that
/// offset as well. A body
/// already registered with the same NAIF ID is replaced.
///
/// Exactly one orientation is required: ``frame_id``, a CK frame read at each
/// evaluation inside ``switch_radius`` (the CK and its clock must be loaded, and
/// a time without pointing is an error), or ``rotation``, a fixed matrix taking
/// body-frame vectors to equatorial (J2000) axes.
///
/// The polyhedron field and the point mass agree ever more closely with distance,
/// so ``switch_radius`` trades evaluation cost against the step in the force where
/// they meet; that step is largest for elongated bodies and falls as the inverse
/// square of ``switch_radius``.
///
/// Parameters
/// ----------
/// naif_id :
///     NAIF ID of the body.
/// vertices :
///     Shape model vertices in the body frame, in km, shape ``(n, 3)``.
/// faces :
///     Faces as 0-based vertex indices, counter-clockwise seen from outside, shape
///     ``(m, 3)``.
/// switch_radius :
///     Distance from the body in km inside which the polyhedron field is used.
/// mass :
///     Mass of the body as a fraction of the Sun's mass. Defaults to the value in
///     the built-in mass table.
/// frame_id :
///     CK frame ID of the body frame.
/// rotation :
///     Fixed rotation matrix from the body frame to equatorial axes, shape
///     ``(3, 3)``.
///
/// Returns
/// -------
/// Polyhedron
///     The registered field, in AU and AU^3/day^2.
#[pyfunction]
#[pyo3(signature = (naif_id, vertices, faces, switch_radius, mass=None, frame_id=None, rotation=None))]
pub fn register_polyhedron(
    naif_id: i32,
    vertices: Vec<[f64; 3]>,
    faces: Vec<[u32; 3]>,
    switch_radius: f64,
    mass: Option<f64>,
    frame_id: Option<i32>,
    rotation: Option<[[f64; 3]; 3]>,
) -> PyResult<PyPolyhedron> {
    let orientation = orientation(frame_id, rotation)?;
    if !switch_radius.is_finite() || switch_radius <= 0.0 {
        Err(Error::ValueError(
            "switch_radius must be positive and finite.".into(),
        ))?;
    }
    let gm = match mass {
        Some(m) => m * GMS,
        None => GravParams::try_mass_from_naif_id(naif_id)?,
    };
    let mesh = build_mesh(vertices, faces, 1.0 / AU_KM)?;
    let model = Arc::new(Polyhedron::new(mesh, gm)?);
    let switch_radius = switch_radius / AU_KM;
    if switch_radius <= model.mesh().bounding_radius() {
        Err(Error::ValueError(
            "switch_radius must be larger than the shape model.".into(),
        ))?;
    }
    // Build the far field now if propagation can reach it, rather than on its first
    // use in the middle of a propagation.
    if model.far_field_radius().is_some_and(|r| r < switch_radius) {
        let _ = model.far_field();
    }
    let mut params = GravParams::new(naif_id, gm, model.mesh().bounding_radius() as f32);
    params.shape = Shape::Polyhedron {
        model: Arc::clone(&model),
        orientation,
        switch_radius,
    };
    params.register();
    Ok(PyPolyhedron(model))
}
