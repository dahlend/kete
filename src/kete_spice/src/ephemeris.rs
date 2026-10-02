//! The loaded SPICE kernels as a [`kete_core::ephemeris::Ephemeris`].

use crossbeam::sync::ShardedLockReadGuard;
use kete_core::ephemeris::Ephemeris;
use kete_core::errors::KeteResult;
use kete_core::frames::{Equatorial, FrameId, NonInertialFrame};
use kete_core::state::State;
use kete_core::time::{TDB, Time};

use crate::frames::try_frame_at;
use crate::spk::{LOADED_SPK, SpkCollection};

/// The loaded kernels, as an [`Ephemeris`].
///
/// States come from the SPK files. [`try_frame_at`] gives the frames.
///
/// It holds a read guard on the SPK files for its life. Thus SPK files cannot
/// load while it exists. Take one for a propagation and drop it afterward.
///
/// The PCK, CK and text kernels are read only for the duration of each frame
/// lookup. Thus a propagation that needs no frame does not lock them.
pub struct SpiceEphemeris {
    spk: ShardedLockReadGuard<'static, SpkCollection>,
}

impl SpiceEphemeris {
    /// A read guard on the loaded SPK files.
    ///
    /// # Errors
    /// [`kete_core::errors::Error::LockFailed`] if the SPK singleton cannot be
    /// read.
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

    /// See [`try_frame_at`].
    fn try_frame(&self, frame_id: FrameId, time: Time<TDB>) -> KeteResult<NonInertialFrame> {
        try_frame_at(frame_id, time)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::ITRF93;
    use crate::pck::LOADED_PCK;

    /// ITRF93 from the provider is the Earth frame the PCK files hold, class ID
    /// 3000. Runs where an Earth PCK is loaded (the user cache); kernels are
    /// not committed.
    #[test]
    fn itrf93_is_the_earth_pck_frame() {
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
        let frame = eph.try_frame(ITRF93, time).unwrap();
        let v = nalgebra::Vector3::new(0.3, -0.2, 0.9);
        let (a, da) = frame.to_equatorial(v, nalgebra::Vector3::zeros()).unwrap();
        let (b, db) = direct.to_equatorial(v, nalgebra::Vector3::zeros()).unwrap();
        assert_eq!((a, da), (b, db));
    }
}
