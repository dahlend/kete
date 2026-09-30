//! `NonInertialFrame` SPICE-dependent rotation resolution.
//!
//! The CK-dependent branch of `rotations_to_equatorial` lives here.

use kete_core::errors::{Error, KeteResult};
use kete_core::frames::NonInertialFrame;
use nalgebra::{Matrix3, Rotation3};

use crate::ck::LOADED_CK;

/// Return the rotation and rotation rate of a frame to the Equatorial frame.
///
/// This function first tries [`NonInertialFrame::rotations_to_equatorial`]. If
/// that fails and `reference_frame_id` is negative, the reference frame comes
/// from the loaded CK kernels. The function resolves the reference frame
/// recursively, then chains the rotations and the rotation rates.
///
/// # Errors
/// - [`Error::Bounds`] if `reference_frame_id` is not negative and is not a
///   supported inertial frame.
/// - [`Error::LockFailed`] if the CK or SCLK read lock cannot be taken.
/// - [`Error::ValueError`] if no SCLK clock is loaded for the spacecraft of the
///   reference frame.
/// - [`Error::Bounds`] if no CK segment holds pointing for the reference frame
///   at the frame time.
/// - [`Error::Bounds`] if the pointing time differs from the frame time by more
///   than 1e-8 days.
/// - Any error from the evaluation of the CK segment.
pub fn rotations_to_equatorial_full(
    frame: &NonInertialFrame,
) -> KeteResult<(Rotation3<f64>, Matrix3<f64>)> {
    // Try the inertial-only resolution first
    match frame.rotations_to_equatorial() {
        ok @ Ok(_) => ok,
        Err(_) if frame.reference_frame_id < 0 => {
            // The reference frame is itself a CK frame, for example a camera
            // relative to its spacecraft. The standard CK lookup selects the
            // segment, so segment coverage and load order apply. The result is
            // then chained with this frame.
            let (time, ref_frame) = LOADED_CK
                .try_read()?
                .try_get_frame(frame.time.jd, frame.reference_frame_id)?;
            if (time.jd - frame.time.jd).abs() > 1e-8 {
                return Err(Error::Bounds(format!(
                    "Reference frame ID {} has no CK data at the requested time.",
                    frame.reference_frame_id
                )));
            }
            // d(R_ref R) / dt = dR_ref R + R_ref dR
            let (ref_rot, ref_rate) = rotations_to_equatorial_full(&ref_frame)?;
            let rate = frame.rotation_rate.unwrap_or_else(Matrix3::zeros);
            Ok((
                ref_rot * frame.rotation,
                ref_rate * frame.rotation.matrix() + ref_rot.matrix() * rate,
            ))
        }
        err => err,
    }
}
