//! The loaded SPICE kernels as a [`kete_core::ephemeris::Ephemeris`].

use crossbeam::sync::ShardedLockReadGuard;
use kete_core::ephemeris::Ephemeris;
use kete_core::errors::{Error, KeteResult};
use kete_core::frames::{Equatorial, NonInertialFrame};
use kete_core::state::State;
use kete_core::time::{TDB, Time};

use crate::ck::{LOADED_CK, POINTING_TOLERANCE_DAYS};
use crate::pck::LOADED_PCK;
use crate::spk::{LOADED_SPK, SpkCollection};

/// The loaded SPK, PCK and CK kernels, as an [`Ephemeris`].
///
/// States come from the SPK files. A body frame comes from the PCK files (for example
/// the Earth, 3000) or, when they have no frame with that id, from the CK files and
/// their clocks (spacecraft, instruments, and comets with an attitude kernel). SPICE
/// declares a frame's class in a frames kernel, which kete does not read, so both are
/// searched; the id is the class id of the PCK segment or the CK id.
///
/// It holds a read guard on the SPK files for its life, so SPK files cannot be loaded
/// while it exists; take one for a propagation and drop it afterwards. The PCK and CK
/// files are read only for the duration of each frame lookup, so a propagation that
/// needs no frame does not touch them.
pub struct SpiceEphemeris {
    spk: ShardedLockReadGuard<'static, SpkCollection>,
}

impl SpiceEphemeris {
    /// A read guard on the loaded SPK files.
    ///
    /// # Errors
    /// [`Error::LockFailed`] if the SPK singleton cannot be read.
    pub fn loaded() -> KeteResult<Self> {
        Ok(Self {
            spk: LOADED_SPK.try_read()?,
        })
    }

    /// The loaded SPK files.
    #[must_use]
    pub fn spk(&self) -> &SpkCollection {
        &self.spk
    }
}

impl std::fmt::Debug for SpiceEphemeris {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpiceEphemeris").finish_non_exhaustive()
    }
}

impl Ephemeris for SpiceEphemeris {
    #[inline(always)]
    fn try_get_state_with_center(
        &self,
        id: i32,
        time: Time<TDB>,
        center: i32,
    ) -> KeteResult<State<Equatorial>> {
        self.spk.try_get_state_with_center(id, time, center)
    }

    fn try_change_center(&self, state: &mut State<Equatorial>, center: i32) -> KeteResult<()> {
        self.spk.try_change_center(state, center)
    }

    /// The frame from the PCK files when any of them holds `frame_id`, otherwise from
    /// the CK files, resolved through its chain of reference frames. The CK must hold pointing
    /// at `time`; a gap is an error.
    fn try_frame(&self, frame_id: i32, time: Time<TDB>) -> KeteResult<NonInertialFrame> {
        {
            let pck = LOADED_PCK.try_read()?;
            if pck.has_frame(frame_id) {
                return pck.try_get_orientation(frame_id, time);
            }
        }
        let ck = LOADED_CK.try_read()?;
        // The error keeps its kind; its message says the PCK files were searched first.
        let context = |msg: String| {
            format!(
                "Frame {frame_id} is in no loaded PCK file; CK lookup at JD {}: {msg}",
                time.jd()
            )
        };
        #[allow(
            clippy::wildcard_enum_match_arm,
            reason = "Error is non_exhaustive; kinds without a message pass through unchanged"
        )]
        let (frame_time, frame) = ck.try_get_frame(time, frame_id).map_err(|err| match err {
            Error::Bounds(msg) => Error::Bounds(context(msg)),
            Error::ValueError(msg) => Error::ValueError(context(msg)),
            Error::IOError(msg) => Error::IOError(context(msg)),
            other => other,
        })?;
        if (frame_time - time).elapsed.abs() > POINTING_TOLERANCE_DAYS {
            return Err(Error::Bounds(format!(
                "CK frame {frame_id} has no pointing at JD {}.",
                time.jd()
            )));
        }
        Ok(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A frame id neither the PCK nor the CK files have, of either sign, is an error
    /// that says the PCK files lack it and why the CK lookup failed.
    #[test]
    fn a_missing_frame_names_both_kernels() {
        crate::test_data::ensure_test_spk();
        let eph = SpiceEphemeris::loaded().unwrap();
        let time = Time::<TDB>::new(2_451_545.0);
        for id in [-987_654_000, 987_654] {
            let err = eph.try_frame(id, time).unwrap_err().to_string();
            assert!(
                err.contains("no loaded PCK file") && err.contains("CK lookup"),
                "{err}"
            );
        }
    }

    /// The Earth frame from the provider is the one the PCK files hold. Runs where an
    /// Earth PCK is loaded (the user cache); kernels are not committed.
    #[test]
    fn earth_frame_is_the_pck_frame() {
        crate::test_data::ensure_test_spk();
        let eph = SpiceEphemeris::loaded().unwrap();
        let time = Time::<TDB>::new(2_460_000.5);
        let Ok(direct) = LOADED_PCK
            .try_read()
            .unwrap()
            .try_get_orientation(3000, time)
        else {
            return;
        };
        let frame = eph.try_frame(3000, time).unwrap();
        let v = nalgebra::Vector3::new(0.3, -0.2, 0.9);
        let (a, da) = frame.to_equatorial(v, nalgebra::Vector3::zeros()).unwrap();
        let (b, db) = direct.to_equatorial(v, nalgebra::Vector3::zeros()).unwrap();
        assert_eq!((a, da), (b, db));
    }
}
