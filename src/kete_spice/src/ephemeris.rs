// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! The loaded SPICE kernels as a [`kete_core::ephemeris::EphemerisSource`].

use crossbeam::sync::ShardedLockReadGuard;
use kete_core::ephemeris::EphemerisSource;
use kete_core::errors::KeteResult;
use kete_core::frames::Equatorial;
use kete_core::state::State;
use kete_core::time::{TDB, Time};

use crate::spk::{LOADED_SPK, SpkCollection};

/// The loaded kernels, as an [`EphemerisSource`].
///
/// States come from the SPK files. [`SpiceFrames`](crate::frames::SpiceFrames)
/// gives the frames, and holds the PCK, CK and text kernels the same way.
///
/// It holds a read guard on the SPK files for its life. Thus SPK files cannot
/// load while it exists. Take one for a propagation and drop it afterward.
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

impl EphemerisSource for SpiceEphemeris {
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
}
