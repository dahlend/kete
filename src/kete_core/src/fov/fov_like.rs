// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-FileCopyrightText: 2025 California Institute of Technology
// SPDX-License-Identifier: BSD-3-Clause

//! # Field of View like trait
//! This trait defines field of view checks for portions of the sky.

use crate::errors::{Error, KeteResult};
use crate::fov::FOV;
use crate::frames::{Equatorial, Vector};
use crate::geometry::Contains;
use crate::state::State;

/// Field of View like objects.
/// These may contain multiple unique sky patches, so as a result the expected
/// behavior is to return the index as well as the [`Contains`] for the closest
/// sky patch.
pub trait FovLike: Sync + Sized {
    /// The type of the child FOV, which is the FOV of a single patch. For example,
    /// a ZTF field contains 16 CCD quads, so the child FOV of a ZTF field is a ZTF CCD.
    type ChildFov: FovLike;

    /// Return the FOV of the patch at the specified index.
    /// This will panic if the index is out of allowed bounds.
    fn get_child(&self, index: usize) -> Self::ChildFov;

    /// Position of the observer.
    fn observer(&self) -> &State<Equatorial>;

    /// Is the specified vector contained within this [`FovLike`].
    /// A [`Contains`] is returned for each sky patch.
    fn contains(&self, obs_to_obj: &Vector<Equatorial>) -> (usize, Contains);

    /// Number of sky patches contained within this FOV.
    fn n_patches(&self) -> usize;

    /// Get the pointing vector of the FOV.
    ///
    /// # Errors
    /// Some ``FoVs`` may not have a well formed pointing vector.
    fn pointing(&self) -> KeteResult<Vector<Equatorial>>;

    /// Get the corners of the FOV.
    ///
    /// # Errors
    /// Not all ``FoVs`` contain corners, such as a Cone.
    fn corners(&self) -> KeteResult<Vec<Vector<Equatorial>>>;

    /// Convert this into an FOV Enum.
    ///
    /// # Errors
    /// This may fail if the FOV cannot be converted into a known FOV type.
    fn into_fov(self) -> FOV;
}

/// Given a collection of static positions, return the index of the input vector
/// which was visible.
pub fn check_statics<F: FovLike>(
    fov: &F,
    pos: &[Vector<Equatorial>],
) -> Vec<Option<(Vec<usize>, F::ChildFov)>> {
    let mut visible: Vec<Vec<usize>> = vec![Vec::new(); fov.n_patches()];

    pos.iter().enumerate().for_each(|(vec_idx, p)| {
        if let (patch_idx, Contains::Inside) = fov.contains(p) {
            visible[patch_idx].push(vec_idx);
        }
    });

    visible
        .into_iter()
        .enumerate()
        .map(|(idx, vis_patch)| {
            if vis_patch.is_empty() {
                None
            } else {
                Some((vis_patch, fov.get_child(idx)))
            }
        })
        .collect()
}

/// Unit vector along the sum of the pointings of `patches`.
///
/// # Errors
/// Fails if `patches` is empty, or if the pointing of a patch fails.
pub(crate) fn patches_pointing<F: FovLike>(patches: &[F]) -> KeteResult<Vector<Equatorial>> {
    if patches.is_empty() {
        Err(Error::ValueError("FOV has no patches.".into()))?;
    }
    let mut pointing = Vector::new([0.0; 3]);
    for patch in patches {
        pointing += &patch.pointing()?;
    }
    Ok(pointing.normalize())
}

/// All corners of all `patches`.
///
/// # Errors
/// Fails if `patches` is empty, or if the corners of a patch fail.
pub(crate) fn patches_corners<F: FovLike>(patches: &[F]) -> KeteResult<Vec<Vector<Equatorial>>> {
    if patches.is_empty() {
        Err(Error::ValueError("FOV has no patches.".into()))?;
    }
    let mut corners = Vec::with_capacity(4 * patches.len());
    for patch in patches {
        corners.extend(patch.corners()?);
    }
    Ok(corners)
}
