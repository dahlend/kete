//! `NonInertialFrame` SPICE-dependent rotation resolution.
//!
//! The CK-dependent branch of `rotations_to_equatorial` lives here.

use kete_core::errors::{Error, KeteResult};
use kete_core::frames::NonInertialFrame;
use nalgebra::{Matrix3, Rotation3};

use crate::spice::{CkArray, LOADED_CK};

/// Resolve `rotations_to_equatorial` for a [`NonInertialFrame`] including CK data.
///
/// This extends the core implementation by supporting non-inertial reference
/// frames (negative `reference_frame_id`) via loaded CK kernels.
///
/// # Errors
/// Fails when reference frame is not found or supported.
///
/// # Panics
/// Panics can occur if a non-inertial reference frame is used but the lock on CK's
/// cannot be taken.
pub fn rotations_to_equatorial_full(
    frame: &NonInertialFrame,
) -> KeteResult<(Rotation3<f64>, Matrix3<f64>)> {
    // Try the inertial-only resolution first
    match frame.rotations_to_equatorial() {
        ok @ Ok(_) => ok,
        Err(_) if frame.reference_frame_id < 0 => {
            // CK-dependent resolution
            let cks = LOADED_CK.try_read()?;

            for segment in &cks.segments {
                let array: &CkArray = segment.into();
                if array.instrument_id == frame.reference_frame_id {
                    let orientation =
                        segment.try_get_orientation(frame.reference_frame_id, frame.time);
                    if orientation.is_err() {
                        continue;
                    }
                    let (time, ref_frame) = orientation.unwrap();
                    if (time.jd - frame.time.jd).abs() > 1e-8 {
                        continue;
                    }
                    let (rot, vel) = rotations_to_equatorial_full(&ref_frame)?;
                    return Ok((
                        rot * frame.rotation,
                        vel * frame.rotation_rate.unwrap_or_else(Matrix3::identity),
                    ));
                }
            }
            Err(Error::Bounds(format!(
                "Reference frame ID {} not found in CK data.",
                frame.reference_frame_id
            )))
        }
        err => err,
    }
}
