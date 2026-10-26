// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Frames defined in memory, by ID.

use super::{BodyRotation, DynFrameId, FrameSource, InstantFrame, OffsetFrame};
use crate::errors::{Error, KeteResult};
use crate::time::{TDB, Time};
use std::collections::BTreeMap;

/// Frames defined in memory, by ID: the [`FrameSource`] without SPICE.
///
/// Empty, as [`Self::new`] gives it, it resolves the static inertial frames only.
/// Each frame it holds is a [`FrameDefinition`], relative to a frame that the table or
/// the static inertial frames provide.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FrameTable {
    frames: BTreeMap<DynFrameId, FrameDefinition>,
}

impl FrameTable {
    /// A table with no frames.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            frames: BTreeMap::new(),
        }
    }

    /// Define frame `frame_id`, replacing any definition it had.
    ///
    /// # Errors
    /// [`Error::ValueError`] if `frame_id` is [`DynFrameId::UNNAMED`] or a static
    /// inertial frame, which the table cannot define.
    pub fn insert(
        &mut self,
        frame_id: DynFrameId,
        definition: impl Into<FrameDefinition>,
    ) -> KeteResult<()> {
        if frame_id == DynFrameId::UNNAMED || frame_id.static_rotation_to_equatorial().is_some() {
            return Err(Error::ValueError(format!(
                "Frame {frame_id} cannot be defined: it is unnamed or a static inertial \
                 frame."
            )));
        }
        let _ = self.frames.insert(frame_id, definition.into());
        Ok(())
    }

    /// The definition of frame `frame_id`, if the table holds it.
    #[must_use]
    pub fn get(&self, frame_id: DynFrameId) -> Option<&FrameDefinition> {
        self.frames.get(&frame_id)
    }
}

impl FrameSource for FrameTable {
    fn try_link(&self, frame_id: DynFrameId, time: Time<TDB>) -> KeteResult<InstantFrame> {
        match self.frames.get(&frame_id) {
            Some(FrameDefinition::Offset(frame)) => Ok(frame.relative_to_reference(time)),
            Some(FrameDefinition::Spin(model)) => Ok(model.frame(time)),
            None => Err(Error::Bounds(format!(
                "No orientation is available for frame {frame_id} at JD {}: it is not in \
                 the frame table.",
                time.jd()
            ))),
        }
    }
}

/// The definition of a frame in a [`FrameTable`].
#[derive(Debug, Clone, PartialEq)]
pub enum FrameDefinition {
    /// A constant rotation relative to another frame.
    Offset(OffsetFrame),

    /// A spinning body, relative to the reference frame of its model.
    Spin(BodyRotation),
}

impl From<OffsetFrame> for FrameDefinition {
    fn from(frame: OffsetFrame) -> Self {
        Self::Offset(frame)
    }
}

impl From<BodyRotation> for FrameDefinition {
    fn from(model: BodyRotation) -> Self {
        Self::Spin(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::{Ecliptic, StaticFrame};
    use nalgebra::Rotation3;

    const BODY: DynFrameId = DynFrameId(-1_000_012);
    const LANDER: DynFrameId = DynFrameId(-1_000_013);

    /// An offset frame on a spinning body turns with it, and an empty table resolves
    /// only the static frames.
    #[test]
    fn spin_and_fixed_frames() {
        let epoch = Time::<TDB>::new(2_457_000.5);
        let model = BodyRotation::uniform(69.0, 64.0, 114.0, 696.5, epoch, DynFrameId::ECLIPJ2000);
        let mount = Rotation3::from_euler_angles(0.1, 0.2, 0.3);
        let mut table = FrameTable::new();
        table.insert(BODY, model.clone()).unwrap();
        table
            .insert(LANDER, OffsetFrame::new(LANDER, mount, BODY))
            .unwrap();

        let time = epoch + 0.37;
        let lander = table.try_frame(LANDER, time).unwrap();
        let expected = Ecliptic::rotation_to_equatorial() * model.frame(time).rotation * mount;
        assert!((lander.rotation.matrix() - expected.matrix()).norm() < 1e-14);
        assert!(lander.rotation_rate.is_some());

        let empty = FrameTable::new();
        assert!(empty.try_frame(LANDER, time).is_err());
        let ecliptic = empty.try_frame(DynFrameId::ECLIPJ2000, time).unwrap();
        assert_eq!(&ecliptic.rotation, Ecliptic::rotation_to_equatorial());
    }

    /// The table cannot define an unnamed or a static frame.
    #[test]
    fn reserved_ids() {
        let mut table = FrameTable::new();
        let model = BodyRotation::uniform(0.0, 90.0, 0.0, 1.0, Time::j2000(), DynFrameId::J2000);
        assert!(table.insert(DynFrameId::UNNAMED, model.clone()).is_err());
        assert!(table.insert(DynFrameId::GALACTIC, model).is_err());
    }
}
