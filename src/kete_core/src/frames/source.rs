// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Sources of frames: an interface for building time-dependent coordinate frames.
//!
//! Many commonly used frames are static, such as [`Equatorial`]. Other frames vary
//! with time, such as the rotating orientation of a small body like 67P, or the
//! orientation of a spacecraft. Those frames come from some producer of frame
//! information, and the [`FrameSource`] trait is the interface to it. There are two
//! primary implementers: the loaded kernels in `kete_spice`, and
//! [`FrameTable`](super::FrameTable) for frames defined without SPICE.
//!
//! Static frames (i.e., [`StaticFrame`](super::StaticFrame)s like [`Equatorial`]) do
//! not need a source.
//!
//! A [`FrameSource`] gives each of its frames as one link: the frame relative to the
//! frame it is defined against, which may be any frame. The provided methods follow
//! the links to the J2000 frame, so the chain logic is written once for every source.

use super::definitions::{DynFrameId, Equatorial, InstantFrame};
use crate::errors::{Error, KeteResult};
use crate::time::{TDB, Time};
use nalgebra::Matrix3;

/// Longest chain of reference frames a resolution follows before it reports a cycle.
pub const MAX_FRAME_CHAIN: usize = 16;

/// A source of coordinate frames by ID, such as the loaded SPICE kernels.
///
/// A source implements [`Self::try_link`], one link of a chain. The provided methods
/// resolve chains through it: [`Self::try_frame`] relative to the equatorial frame,
/// [`Self::try_frame_relative`] relative to any frame, and [`Self::try_resolve`] for a
/// frame that the source does not hold. A source may override them with a faster
/// route to the same result, such as one that holds its data for a whole chain.
///
/// `kete_spice` implements this for its kernels, and
/// [`FrameTable`](super::FrameTable) for frames defined in memory.
///
pub trait FrameSource: Sync {
    /// The frame `frame_id` at `time`, relative to the frame it is defined against.
    ///
    /// The reference may be any frame the source provides, or any static frame. The
    /// resolver names the link `frame_id` and stamps it with `time`, so the source
    /// need not, and may give a link from data near `time`, such as CK pointing
    /// within a tolerance. Static frames are never requested.
    ///
    /// # Errors
    /// Fails when the source has no orientation for `frame_id` at `time`.
    fn try_link(&self, frame_id: DynFrameId, time: Time<TDB>) -> KeteResult<InstantFrame>;

    /// The frame `frame_id` at `time`, relative to the equatorial frame.
    ///
    /// Every link in the chain is at `time`. The result has a rotation rate only if
    /// every link has one.
    ///
    /// # Errors
    /// - The errors of [`Self::try_link`] for any frame in the chain.
    /// - [`Error::ValueError`] if a frame is relative to [`DynFrameId::UNNAMED`], or
    ///   the chain loops or holds more than [`MAX_FRAME_CHAIN`] frames.
    fn try_frame(
        &self,
        frame_id: DynFrameId,
        time: Time<TDB>,
    ) -> KeteResult<InstantFrame<Equatorial>> {
        resolve(self, link(self, frame_id, time)?, time)
    }

    /// The frame `frame_id` at `time`, relative to the frame `reference`.
    ///
    /// The two chains are composed only up to the nearest frame they share, so a
    /// camera relative to its spacecraft does not depend on the orientation of the
    /// spacecraft.
    ///
    /// # Errors
    /// As [`Self::try_frame`], for either frame.
    fn try_frame_relative(
        &self,
        frame_id: DynFrameId,
        reference: DynFrameId,
        time: Time<TDB>,
    ) -> KeteResult<InstantFrame> {
        let frame_links = chain(self, frame_id, time)?;
        let reference_links = chain(self, reference, time)?;
        // Each frame has one reference, so once the chains reach a common frame they
        // share every link after it: the shared links are the common tail of both.
        let shared = frame_links
            .iter()
            .rev()
            .zip(reference_links.iter().rev())
            .take_while(|(a, b)| a.frame == b.frame)
            .count();
        let to_common = compose(&frame_links[..frame_links.len() - shared], frame_id, time)?;
        let common_to_reference = compose(
            &reference_links[..reference_links.len() - shared],
            reference,
            time,
        )?
        .inverse();
        to_common.try_relative_to(&common_to_reference)
    }

    /// `link`, a frame relative to any frame, resolved relative to the equatorial
    /// frame.
    ///
    /// This resolves a frame that the source does not hold, such as a
    /// [`OffsetFrame`](super::OffsetFrame) on a body frame. Every link in the chain is
    /// at the time of `link`.
    ///
    /// # Errors
    /// As [`Self::try_frame`].
    fn try_resolve(&self, link: InstantFrame) -> KeteResult<InstantFrame<Equatorial>> {
        resolve(self, link, link.time)
    }
}

/// The link of frame `frame_id` at `time`, named `frame_id` and stamped with `time`: a
/// static inertial frame relative to J2000, any other frame from `source`.
///
/// J2000, the root, links to itself by the identity.
fn link<S: FrameSource + ?Sized>(
    source: &S,
    frame_id: DynFrameId,
    time: Time<TDB>,
) -> KeteResult<InstantFrame> {
    if let Some(rotation) = frame_id.static_rotation_to_equatorial() {
        return Ok(InstantFrame::from_rotations(
            time,
            *rotation,
            Some(Matrix3::zeros()),
            frame_id,
            DynFrameId::J2000,
        ));
    }
    let link = source.try_link(frame_id, time)?;
    Ok(InstantFrame {
        frame: frame_id,
        time,
        ..link
    })
}

/// Follow the reference frames of `first` to J2000, passing the link of each, at
/// `time`, to `visit`.
///
/// # Errors
/// - The errors of `visit`, and of [`FrameSource::try_link`] for any frame in the
///   chain.
/// - [`Error::ValueError`] if a frame is relative to [`DynFrameId::UNNAMED`], or
///   the chain loops or holds more than [`MAX_FRAME_CHAIN`] frames.
fn walk<S: FrameSource + ?Sized>(
    source: &S,
    first: &InstantFrame,
    time: Time<TDB>,
    mut visit: impl FnMut(InstantFrame) -> KeteResult<()>,
) -> KeteResult<()> {
    let (mut frame, mut reference) = (first.frame, first.reference);
    let mut links = 1;
    while reference != DynFrameId::J2000 {
        if reference == DynFrameId::UNNAMED {
            return Err(Error::ValueError(format!(
                "Frame {frame} is relative to an unnamed frame."
            )));
        }
        if links >= MAX_FRAME_CHAIN {
            return Err(Error::ValueError(format!(
                "Frames form a cycle, or a chain of more than {MAX_FRAME_CHAIN} frames, at \
                 frame {reference}."
            )));
        }
        let next = link(source, reference, time)?;
        (frame, reference) = (next.frame, next.reference);
        links += 1;
        visit(next)?;
    }
    Ok(())
}

/// `first` relative to the equatorial frame, composing each link of its chain at
/// `time` in turn.
///
/// # Errors
/// As [`walk`].
fn resolve<S: FrameSource + ?Sized>(
    source: &S,
    first: InstantFrame,
    time: Time<TDB>,
) -> KeteResult<InstantFrame<Equatorial>> {
    let mut frame = first;
    walk(source, &first, time, |next| {
        frame = frame.try_relative_to(&next)?;
        Ok(())
    })?;
    frame.try_relative_to_equatorial()
}

/// The links from frame `frame_id` to J2000 at `time`: its own link, then that of
/// each reference frame.
///
/// # Errors
/// As [`walk`].
fn chain<S: FrameSource + ?Sized>(
    source: &S,
    frame_id: DynFrameId,
    time: Time<TDB>,
) -> KeteResult<Vec<InstantFrame>> {
    let first = link(source, frame_id, time)?;
    let mut links = vec![first];
    walk(source, &first, time, |next| {
        links.push(next);
        Ok(())
    })?;
    Ok(links)
}

/// The first frame of `links` relative to the reference frame of the last, or
/// `frame_id` relative to itself at `time` if there are none.
fn compose(
    links: &[InstantFrame],
    frame_id: DynFrameId,
    time: Time<TDB>,
) -> KeteResult<InstantFrame> {
    let Some((last, rest)) = links.split_last() else {
        return Ok(InstantFrame::identity(time, frame_id));
    };
    rest.iter()
        .rev()
        .try_fold(*last, |base, link| link.try_relative_to(&base))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::{Ecliptic, FrameTable, StaticFrame};
    use nalgebra::Rotation3;

    const BUS: DynFrameId = DynFrameId(-500);
    const MOUNT: DynFrameId = DynFrameId(-501);
    const CAMERA: DynFrameId = DynFrameId(-502);
    const OTHER: DynFrameId = DynFrameId(-503);

    /// A spinning bus relative to the ecliptic, a fixed mount on it, a camera spinning
    /// on the mount without a known rate, whose pointing is from just after the
    /// requested time, as CK pointing can be, and a frame relative to the camera that
    /// closes a cycle if asked to.
    struct Spacecraft {
        cycle: bool,
    }

    fn bus(time: Time<TDB>) -> InstantFrame {
        let t = (time - Time::j2000()).elapsed;
        InstantFrame::from_euler::<'Z', 'X', 'Z'>(
            time,
            [0.3 + 2.0 * t, 0.2, -0.4],
            [2.0, 0.0, 0.0],
            BUS,
            DynFrameId::ECLIPJ2000,
        )
    }

    fn mount_rotation() -> Rotation3<f64> {
        Rotation3::from_euler_angles(0.1, -0.5, 0.25)
    }

    impl FrameSource for Spacecraft {
        fn try_link(&self, frame_id: DynFrameId, time: Time<TDB>) -> KeteResult<InstantFrame> {
            match frame_id {
                BUS => Ok(bus(time)),
                MOUNT => Ok(InstantFrame::from_rotations(
                    time,
                    mount_rotation(),
                    Some(Matrix3::zeros()),
                    DynFrameId::UNNAMED,
                    BUS,
                )),
                CAMERA => Ok(InstantFrame::from_rotations(
                    time + 1e-9,
                    Rotation3::from_euler_angles(0.0, 0.0, 0.7),
                    None,
                    DynFrameId::UNNAMED,
                    MOUNT,
                )),
                OTHER if self.cycle => Ok(InstantFrame::identity(time, OTHER)),
                OTHER => Ok(InstantFrame::from_rotations(
                    time,
                    Rotation3::identity(),
                    None,
                    OTHER,
                    DynFrameId::UNNAMED,
                )),
                _ => FrameTable::new().try_link(frame_id, time),
            }
        }
    }

    fn epoch() -> Time<TDB> {
        Time::new(2_451_545.25)
    }

    /// A chain composes each link down to equatorial, and names the links.
    #[test]
    fn frame_follows_its_chain() {
        let source = Spacecraft { cycle: false };
        let frame = source.try_frame(MOUNT, epoch()).unwrap();
        assert_eq!(frame.frame, MOUNT);
        let expected =
            Ecliptic::rotation_to_equatorial() * bus(epoch()).rotation * mount_rotation();
        assert!((frame.rotation.matrix() - expected.matrix()).norm() < 1e-15);
        let expected_rate = Ecliptic::rotation_to_equatorial().matrix()
            * bus(epoch()).rotation_rate.unwrap()
            * mount_rotation().matrix();
        assert!((frame.rotation_rate.unwrap() - expected_rate).norm() < 1e-15);

        // The camera has no rate, so neither does its chain. Its pointing is at a time
        // near the request, and the resolved frame is at the request.
        let camera = source.try_frame(CAMERA, epoch()).unwrap();
        assert!(camera.rotation_rate.is_none());
        assert_eq!(camera.time, epoch());
        let camera = source.try_frame_relative(CAMERA, BUS, epoch()).unwrap();
        assert_eq!(camera.time, epoch());
    }

    /// Static frames resolve without a source.
    #[test]
    fn static_frames_need_no_source() {
        let frame = FrameTable::new()
            .try_frame(DynFrameId::ECLIPJ2000, epoch())
            .unwrap();
        assert_eq!(&frame.rotation, Ecliptic::rotation_to_equatorial());
        let frame = FrameTable::new()
            .try_frame(DynFrameId::J2000, epoch())
            .unwrap();
        assert_eq!(frame.rotation, Rotation3::identity());
        assert!(FrameTable::new().try_frame(BUS, epoch()).is_err());
    }

    /// Every static frame links straight to J2000, with its rotation to equatorial,
    /// and relative frames between static frames compose those rotations.
    #[test]
    fn static_frames_link_to_j2000() {
        use crate::frames::{FK4, Galactic};
        let frame = FrameTable::new()
            .try_frame(DynFrameId::GALACTIC, epoch())
            .unwrap();
        let err = (frame.rotation.matrix() - Galactic::rotation_to_equatorial().matrix()).norm();
        assert!(err < 1e-15, "{err:e}");

        // Galactic relative to FK4, through J2000.
        let frame = FrameTable::new()
            .try_frame_relative(DynFrameId::GALACTIC, DynFrameId::FK4, epoch())
            .unwrap();
        let expected = FK4::rotation_to_equatorial().inverse() * Galactic::rotation_to_equatorial();
        assert!((frame.rotation.matrix() - expected.matrix()).norm() < 1e-15);
    }

    /// A frame relative to another stops at the frame they share, so the mount
    /// relative to the bus is its fixed rotation, with a zero rate, and the camera
    /// relative to the mount needs nothing above the mount.
    #[test]
    fn relative_frames_stop_at_the_common_frame() {
        let source = Spacecraft { cycle: false };
        let mount = source.try_frame_relative(MOUNT, BUS, epoch()).unwrap();
        assert_eq!((mount.frame, mount.reference), (MOUNT, BUS));
        assert!((mount.rotation.matrix() - mount_rotation().matrix()).norm() < 1e-15);
        assert_eq!(mount.rotation_rate, Some(Matrix3::zeros()));

        // The bus relative to the mount is the inverse.
        let bus_frame = source.try_frame_relative(BUS, MOUNT, epoch()).unwrap();
        let err = (bus_frame.rotation * mount.rotation).angle();
        assert!(err < 1e-15, "{err:e}");

        // Across chains that only share J2000, it is the composition through
        // equatorial.
        let to_ecliptic = source
            .try_frame_relative(MOUNT, DynFrameId::ECLIPJ2000, epoch())
            .unwrap();
        let expected = bus(epoch()).rotation * mount_rotation();
        assert!((to_ecliptic.rotation.matrix() - expected.matrix()).norm() < 1e-14);

        let same = source.try_frame_relative(CAMERA, CAMERA, epoch()).unwrap();
        assert_eq!(same.rotation, Rotation3::identity());
    }

    /// The rate of a relative frame is the derivative of its rotation.
    #[test]
    fn relative_rate_is_the_derivative() {
        let source = Spacecraft { cycle: false };
        let rotation = |time: Time<TDB>| {
            *source
                .try_frame_relative(DynFrameId::ECLIPJ2000, MOUNT, time)
                .unwrap()
                .rotation
                .matrix()
        };
        let h = 1e-4;
        let numeric = (rotation(epoch() + h) - rotation(epoch() - h)) / (2.0 * h);
        let rate = source
            .try_frame_relative(DynFrameId::ECLIPJ2000, MOUNT, epoch())
            .unwrap()
            .rotation_rate
            .unwrap();
        assert!((rate - numeric).abs().max() < 1e-7, "{rate} {numeric}");
    }

    /// A cycle, or a frame relative to an unnamed frame, is an error.
    #[test]
    fn broken_chains_are_errors() {
        assert!(
            Spacecraft { cycle: true }
                .try_frame(OTHER, epoch())
                .is_err()
        );
        assert!(
            Spacecraft { cycle: false }
                .try_frame(OTHER, epoch())
                .is_err()
        );
    }
}
