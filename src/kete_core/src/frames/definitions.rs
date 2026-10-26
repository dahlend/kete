// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Coordinate frames: their IDs, the static frames, frames at an instant, and offset
//! frames.
//!
//! Every frame is a rotation relative to a reference frame. [`Equatorial`]
//! is the root, its own reference. It is the frame of the DE440 ephemeris,
//! whose states limit the speed of orbit integration.
//!
//! The file is in four parts, each building on those before it:
//!
//! 1. **Frame IDs.** [`FrameId`] tags a frame. [`DynFrameId`] is a SPICE frame ID
//!    carried at runtime. The static inertial frames are zero-sized tags known at
//!    compile time.
//! 2. **Static frames.** [`StaticFrame`]: [`Equatorial`], [`Ecliptic`], [`B1950`],
//!    [`FK4`], [`Galactic`], [`EclipticB1950`] and [`MarsIAU`], inertial frames
//!    known at compile time, each a constant rotation to the equatorial frame. They
//!    are the frame parameter of [`State`](crate::state::State) and
//!    [`Vector`](super::Vector), and their conversions cannot fail.
//! 3. **Frames at an instant.** [`InstantFrame`] is the rotation of a frame, and
//!    its rotation rate if known, relative to its reference at one time. It
//!    transforms positions and velocities, and composes with its reference frame.
//! 4. **Offset frames.** [`OffsetFrame`] is a frame stored without a time: a
//!    constant rotation relative to any reference frame, so it turns with that
//!    reference if the reference rotates. A static frame is an offset frame whose
//!    references are all static, known at compile time. [`OffsetFrame::try_at`] gives it at a time,
//!    relative to the equatorial frame, using a [`FrameSource`] for any reference
//!    that is not a static inertial frame.
//!
//! [`FrameSource`], in `source.rs`, provides frames by ID and resolves chains of
//! reference frames.

use crate::errors::{Error, KeteResult};
use crate::time::{TDB, Time};
use nalgebra::{Matrix3, Rotation3, Vector3};
use std::fmt::Debug;
use std::sync::LazyLock;

use super::earth::OBLIQUITY;
use super::euler_rotation;
use super::rotation::axis_rotation;
use super::source::FrameSource;

// ---------------------------------------------------------------------------
// Frame IDs
// ---------------------------------------------------------------------------

/// The SPICE frame ID of a frame, for the reference type parameter of
/// [`InstantFrame`].
///
/// This mirrors [`CenterBody`](super::CenterBody) for the center of a state. The
/// static inertial frames ([`Equatorial`], [`Ecliptic`], [`Galactic`] and the others) are
/// zero-sized types that guarantee a frame at compile time, see [`StaticFrame`].
/// [`DynFrameId`] carries the ID at runtime, and is the default.
pub trait FrameId: Sized + Sync + Send + Clone + Copy + Debug + PartialEq {
    /// Runtime SPICE frame ID of this frame.
    fn id(&self) -> DynFrameId;
}

/// SPICE frame ID of a frame known at runtime, such as [`Self::J2000`].
///
/// It is the runtime [`FrameId`], the frame counterpart of
/// [`DynCenter`](super::DynCenter). A [`FrameSource`] provides a body frame by its
/// ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DynFrameId(pub i32);

impl DynFrameId {
    /// The ID of a frame that has none.
    pub const UNNAMED: Self = Self(0);

    /// The SPICE J2000 frame, the equatorial frame.
    pub const J2000: Self = Self(1);

    /// The SPICE B1950 frame.
    pub const B1950: Self = Self(2);

    /// The SPICE FK4 frame.
    pub const FK4: Self = Self(3);

    /// The SPICE GALACTIC frame.
    pub const GALACTIC: Self = Self(13);

    /// The SPICE MARSIAU frame.
    pub const MARSIAU: Self = Self(16);

    /// The SPICE ECLIPJ2000 frame, the ecliptic frame.
    pub const ECLIPJ2000: Self = Self(17);

    /// The SPICE ECLIPB1950 frame.
    pub const ECLIPB1950: Self = Self(18);

    /// The SPICE ITRF93 frame, the precise Earth-fixed frame of the Earth
    /// orientation PCK kernels.
    pub const ITRF93: Self = Self(13000);
}

impl FrameId for DynFrameId {
    #[inline(always)]
    fn id(&self) -> DynFrameId {
        *self
    }
}

impl std::fmt::Display for DynFrameId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Static frames
// ---------------------------------------------------------------------------

/// A static inertial frame: a constant rotation to the equatorial frame.
///
/// The implementors are [`Equatorial`], the root, and the frames of the
/// `static_frames!` list: [`Ecliptic`], [`B1950`], [`FK4`], [`Galactic`],
/// [`EclipticB1950`] and [`MarsIAU`]. Conversions between them need no time and no
/// [`FrameSource`], and cannot fail.
pub trait StaticFrame: FrameId {
    /// SPICE frame ID of this frame.
    const ID: DynFrameId;

    /// Rotation from this frame to the equatorial frame.
    fn rotation_to_equatorial() -> &'static Rotation3<f64>;

    /// Convert a vector from input frame to equatorial frame.
    #[inline(always)]
    #[must_use]
    fn to_equatorial(vec: Vector3<f64>) -> Vector3<f64> {
        Self::rotation_to_equatorial().transform_vector(&vec)
    }

    /// Convert a vector from the equatorial frame to this frame.
    #[inline(always)]
    #[must_use]
    fn from_equatorial(vec: Vector3<f64>) -> Vector3<f64> {
        Self::rotation_to_equatorial().inverse_transform_vector(&vec)
    }

    /// Convert between frames.
    #[inline(always)]
    #[must_use]
    fn convert<Target: StaticFrame>(vec: Vector3<f64>) -> Vector3<f64> {
        Target::from_equatorial(Self::to_equatorial(vec))
    }

    /// Rotation matrix from this frame to another inertial frame.
    ///
    /// [`Self::convert`] applies this to a single vector. The matrix itself is what is
    /// needed to rotate a Jacobian or a covariance, where the same rotation acts on every
    /// column rather than on one vector.
    #[inline(always)]
    #[must_use]
    fn rotation_to_frame<Target: StaticFrame>() -> Rotation3<f64> {
        Target::rotation_to_equatorial().inverse() * *Self::rotation_to_equatorial()
    }
}

/// Equatorial frame.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct Equatorial {}

impl FrameId for Equatorial {
    #[inline(always)]
    fn id(&self) -> DynFrameId {
        Self::ID
    }
}

impl StaticFrame for Equatorial {
    const ID: DynFrameId = DynFrameId::J2000;

    #[inline(always)]
    fn rotation_to_equatorial() -> &'static Rotation3<f64> {
        &IDENTITY_ROT
    }

    #[inline(always)]
    fn to_equatorial(vec: Vector3<f64>) -> Vector3<f64> {
        // equatorial is a special case, so we can skip the rotation
        // and just return the vector as is.
        vec
    }

    #[inline(always)]
    fn from_equatorial(vec: Vector3<f64>) -> Vector3<f64> {
        // equatorial is a special case, so we can skip the rotation
        // and just return the vector as is.
        vec
    }
}

/// Define the static inertial frames other than [`Equatorial`], and the runtime lookup
/// of every static frame, [`DynFrameId::static_rotation_to_equatorial`], from one list.
///
/// Each entry is a frame type, its [`DynFrameId`] constant, and the static rotation
/// from it to the equatorial frame.
macro_rules! static_frames {
    ($($(#[$meta:meta])* $name:ident($id:ident): $rotation:ident;)*) => {
        $(
            $(#[$meta])*
            #[derive(Debug, Clone, Copy, PartialEq)]
            #[must_use]
            pub struct $name {}

            impl FrameId for $name {
                #[inline(always)]
                fn id(&self) -> DynFrameId {
                    Self::ID
                }
            }

            impl StaticFrame for $name {
                const ID: DynFrameId = DynFrameId::$id;

                #[inline(always)]
                fn rotation_to_equatorial() -> &'static Rotation3<f64> {
                    &$rotation
                }
            }
        )*

        impl DynFrameId {
            /// The fixed rotation from this frame to the equatorial frame, if this ID
            /// is one of the static inertial frames.
            #[must_use]
            pub fn static_rotation_to_equatorial(self) -> Option<&'static Rotation3<f64>> {
                match self {
                    Self::J2000 => Some(Equatorial::rotation_to_equatorial()),
                    $(Self::$id => Some($name::rotation_to_equatorial()),)*
                    _ => None,
                }
            }
        }
    };
}

static_frames! {
    /// Ecliptic frame: the mean ecliptic and equinox of J2000, the frame of JPL's
    /// orbital elements.
    Ecliptic(ECLIPJ2000): ECLIPTIC_EQUATORIAL_ROT;

    /// B1950 frame: the mean equator and dynamical equinox of B1950.
    B1950(B1950): B1950_EQUATORIAL_ROT;

    /// FK4 frame: the B1950 frame with the FK4 equinox.
    FK4(FK4): FK4_EQUATORIAL_ROT;

    /// Galactic frame: Galactic System II.
    Galactic(GALACTIC): GALACTIC_EQUATORIAL_ROT;

    /// Ecliptic frame of B1950: the mean ecliptic and equinox of B1950.
    EclipticB1950(ECLIPB1950): ECLIPB1950_EQUATORIAL_ROT;

    /// Mars frame: the Mars mean equator and IAU vector of J2000.
    MarsIAU(MARSIAU): MARSIAU_EQUATORIAL_ROT;
}

// ---------------------------------------------------------------------------
// Frames at an instant
// ---------------------------------------------------------------------------

/// A frame at a single instant: its rotation, and rotation rate if known, relative to
/// a reference frame.
///
/// `R` identifies the reference frame, as `C` does the center of a
/// [`State`](crate::state::State). With a static reference, such as
/// [`Equatorial`], the rotation to equatorial cannot fail. With the default
/// [`DynFrameId`] it fails unless the reference is a static inertial frame.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct InstantFrame<R: FrameId = DynFrameId> {
    /// Time of the frame in TDB.
    pub time: Time<TDB>,

    /// Rotation matrix from this frame to the reference frame.
    pub rotation: Rotation3<f64>,

    /// Time derivative of `rotation`, per day.
    ///
    /// It is `None` when the source of the frame holds no rate, such as a CK
    /// segment without angular velocity. A frame without a rate gives its
    /// rotation, but not a velocity transformation.
    pub rotation_rate: Option<Matrix3<f64>>,

    /// This frame, [`DynFrameId::UNNAMED`] if it has no ID.
    pub frame: DynFrameId,

    /// The frame that this frame is defined relative to.
    pub reference: R,
}

impl<R: FrameId> InstantFrame<R> {
    /// A frame from its rotation, and rotation rate if known, relative to `reference`.
    ///
    /// # Arguments
    /// * `time` - Time of the frame in TDB.
    /// * `rotation` - Rotation matrix from this frame to the reference frame.
    /// * `rotation_rate` - Time derivative of `rotation` per day, or `None` if unknown.
    /// * `frame` - This frame.
    /// * `reference` - The frame that this frame is defined relative to.
    pub fn from_rotations(
        time: impl Into<Time<TDB>>,
        rotation: Rotation3<f64>,
        rotation_rate: Option<Matrix3<f64>>,
        frame: DynFrameId,
        reference: R,
    ) -> Self {
        Self {
            time: time.into(),
            rotation,
            rotation_rate,
            frame,
            reference,
        }
    }

    /// A frame from its rotation relative to `reference`, and its angular velocity
    /// relative to `reference` in radians per day, expressed on the axes of
    /// `reference`.
    ///
    /// The rotation rate is `[w]x R`. It is `None` when `angular_velocity` is.
    pub fn from_angular_velocity(
        time: impl Into<Time<TDB>>,
        rotation: Rotation3<f64>,
        angular_velocity: Option<Vector3<f64>>,
        frame: DynFrameId,
        reference: R,
    ) -> Self {
        let rate = angular_velocity.map(|w| w.cross_matrix() * rotation.matrix());
        Self::from_rotations(time, rotation, rate, frame, reference)
    }

    /// A frame from Euler angles about the axes `E1`, `E2` and `E3`, and their rates,
    /// relative to `reference`.
    pub fn from_euler<const E1: char, const E2: char, const E3: char>(
        time: impl Into<Time<TDB>>,
        angles: [f64; 3],
        rates: [f64; 3],
        frame: DynFrameId,
        reference: R,
    ) -> Self {
        let (rot_p, rot_dp) = euler_rotation::<E1, E2, E3>(&angles, &rates);
        Self::from_rotations(time, rot_p, Some(rot_dp), frame, reference)
    }
}

impl InstantFrame {
    /// The frame `frame` relative to itself: the identity, with a zero rate.
    pub fn identity(time: impl Into<Time<TDB>>, frame: DynFrameId) -> Self {
        Self::from_rotations(
            time,
            Rotation3::identity(),
            Some(Matrix3::zeros()),
            frame,
            frame,
        )
    }
}

impl<R: FrameId> InstantFrame<R> {
    /// This frame relative to the reference of `base`, where `base` is the reference
    /// frame of this frame at the same time.
    ///
    /// The IDs and times must match. The result has a rotation rate only if both frames
    /// have one.
    ///
    /// # Errors
    /// [`Error::ValueError`] if `base` is not the reference frame of this frame at its
    /// time, or if the reference frame is [`DynFrameId::UNNAMED`], which matches no
    /// frame.
    pub fn try_relative_to<B: FrameId>(
        &self,
        base: &InstantFrame<B>,
    ) -> KeteResult<InstantFrame<B>> {
        if self.reference.id() == DynFrameId::UNNAMED || base.frame != self.reference.id() {
            return Err(Error::ValueError(format!(
                "Frame {} is defined relative to frame {}, not frame {}.",
                self.frame,
                self.reference.id(),
                base.frame
            )));
        }
        if base.time != self.time {
            return Err(Error::ValueError(format!(
                "Frame {} is at JD {}, but its reference frame {} is at JD {}.",
                self.frame,
                self.time.jd(),
                base.frame,
                base.time.jd()
            )));
        }
        // d(R_base R) / dt = dR_base R + R_base dR; unknown if either rate is.
        let rate = base
            .rotation_rate
            .zip(self.rotation_rate)
            .map(|(base_rate, rate)| {
                base_rate * self.rotation.matrix() + base.rotation.matrix() * rate
            });
        Ok(InstantFrame::from_rotations(
            self.time,
            base.rotation * self.rotation,
            rate,
            self.frame,
            base.reference,
        ))
    }

    /// The reference frame relative to this frame.
    pub fn inverse(&self) -> InstantFrame {
        // d(R^T) / dt = (dR / dt)^T
        InstantFrame::from_rotations(
            self.time,
            self.rotation.inverse(),
            self.rotation_rate.map(|rate| rate.transpose()),
            self.reference.id(),
            self.frame,
        )
    }

    /// This frame relative to the equatorial frame, through its static inertial
    /// reference frame. A static rotation has no rate, so the result has a rotation
    /// rate if this frame does.
    ///
    /// # Errors
    /// [`Error::ValueError`] if the reference frame is not a static inertial frame.
    pub fn try_relative_to_equatorial(&self) -> KeteResult<InstantFrame<Equatorial>> {
        // Relative to J2000, the frame is already relative to equatorial.
        if self.reference.id() == DynFrameId::J2000 {
            return Ok(InstantFrame::from_rotations(
                self.time,
                self.rotation,
                self.rotation_rate,
                self.frame,
                Equatorial {},
            ));
        }
        let to_equatorial = self.reference_to_equatorial()?;
        Ok(InstantFrame::from_rotations(
            self.time,
            to_equatorial * self.rotation,
            self.rotation_rate.map(|rate| to_equatorial * rate),
            self.frame,
            Equatorial {},
        ))
    }

    /// Convert a position and velocity from this frame to the equatorial frame.
    ///
    /// # Errors
    /// As [`Self::from_equatorial`].
    pub fn to_equatorial(
        &self,
        pos: impl Into<Vector3<f64>>,
        vel: impl Into<Vector3<f64>>,
    ) -> KeteResult<(Vector3<f64>, Vector3<f64>)> {
        let pos = pos.into();
        let (rot_p, rot_dp) = self.rotation_and_rate_to_equatorial()?;

        let new_pos = rot_p.transform_vector(&pos);
        let new_vel = rot_dp * pos + rot_p.transform_vector(&vel.into());
        Ok((new_pos, new_vel))
    }

    /// Convert a position and velocity from the equatorial frame to this frame.
    ///
    /// # Errors
    /// - [`Error::ValueError`] if the reference frame is not a static inertial frame.
    /// - [`Error::ValueError`] if the frame has no rotation rate.
    pub fn from_equatorial(
        &self,
        pos: impl Into<Vector3<f64>>,
        vel: impl Into<Vector3<f64>>,
    ) -> KeteResult<(Vector3<f64>, Vector3<f64>)> {
        let pos = pos.into();
        let (rot_p, rot_dp) = self.rotation_and_rate_to_equatorial()?;

        let new_pos = rot_p.inverse_transform_vector(&pos);
        let new_vel = rot_dp.transpose() * pos + rot_p.inverse_transform_vector(&vel.into());

        Ok((new_pos, new_vel))
    }

    /// The rotation from the reference frame to the equatorial frame.
    ///
    /// # Errors
    /// [`Error::ValueError`] if the reference frame is not a static inertial frame.
    fn reference_to_equatorial(&self) -> KeteResult<&'static Rotation3<f64>> {
        let reference = self.reference.id();
        reference.static_rotation_to_equatorial().ok_or_else(|| {
            Error::ValueError(format!(
                "Frame {} is relative to frame {reference}, which is not a static \
                 inertial frame, such as J2000 (1) or ECLIPJ2000 (17); a frame relative to \
                 another frame is resolved by its FrameSource.",
                self.frame,
            ))
        })
    }

    /// The rotation from this frame to the equatorial frame, and its time
    /// derivative per day.
    ///
    /// # Errors
    /// - [`Error::ValueError`] if the reference frame is not a static inertial frame.
    /// - [`Error::ValueError`] if the frame has no rotation rate.
    fn rotation_and_rate_to_equatorial(&self) -> KeteResult<(Rotation3<f64>, Matrix3<f64>)> {
        let rate = self.rotation_rate.ok_or_else(|| {
            Error::ValueError(
                "The frame has no rotation rate, so it rotates positions but cannot \
                 transform velocities. A CK segment without angular velocity gives such a \
                 frame."
                    .into(),
            )
        })?;
        // Relative to equatorial, the rotation is already the one to equatorial.
        if self.reference.id() == DynFrameId::J2000 {
            return Ok((self.rotation, rate));
        }
        let to_equatorial = self.reference_to_equatorial()?;
        Ok((to_equatorial * self.rotation, to_equatorial * rate))
    }
}

impl<R: StaticFrame> InstantFrame<R> {
    /// The rotation from this frame to the equatorial frame.
    #[must_use]
    pub fn rotation_to_equatorial(&self) -> Rotation3<f64> {
        R::rotation_to_equatorial() * self.rotation
    }
}

// ---------------------------------------------------------------------------
// Offset frames
// ---------------------------------------------------------------------------

/// A frame at a constant rotation relative to a reference frame, at any time.
///
/// It is the runtime form of a [`StaticFrame`], which converts to one with [`From`],
/// and of a SPICE TK frame. Unlike a static frame, its reference may be any frame, so
/// it is inertial only if its reference is. A static reference needs no source; any
/// other is provided by a [`FrameSource`] at each time, so the frame turns with its
/// reference. The identity
/// rotation relative to a body frame is that body frame.
///
/// This is what is stored when there is not yet a time, such as the body frame of a
/// shaped gravity field.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct OffsetFrame {
    /// This frame, [`DynFrameId::UNNAMED`] if it has no ID.
    pub frame: DynFrameId,

    /// Rotation from this frame to the reference frame.
    pub rotation: Rotation3<f64>,

    /// The frame that this frame is defined relative to.
    pub reference: DynFrameId,
}

impl<F: StaticFrame> From<F> for OffsetFrame {
    fn from(_: F) -> Self {
        Self::new(F::ID, *F::rotation_to_equatorial(), DynFrameId::J2000)
    }
}

impl OffsetFrame {
    /// The frame `frame` at `rotation` from itself to `reference`.
    pub fn new(frame: DynFrameId, rotation: Rotation3<f64>, reference: DynFrameId) -> Self {
        Self {
            frame,
            rotation,
            reference,
        }
    }

    /// This frame at `time`, relative to its reference frame, which it does not
    /// rotate against.
    pub fn relative_to_reference(&self, time: Time<TDB>) -> InstantFrame {
        InstantFrame::from_rotations(
            time,
            self.rotation,
            Some(Matrix3::zeros()),
            self.frame,
            self.reference,
        )
    }

    /// This frame at `time`, relative to the equatorial frame.
    ///
    /// It has a rotation rate if its reference frame does.
    ///
    /// # Errors
    /// As [`FrameSource::try_resolve`].
    pub fn try_at<S: FrameSource>(
        &self,
        time: Time<TDB>,
        source: &S,
    ) -> KeteResult<InstantFrame<Equatorial>> {
        source.try_resolve(self.relative_to_reference(time))
    }
}

// ---------------------------------------------------------------------------
// Static frame rotations
// ---------------------------------------------------------------------------

// The rotation of each static frame to its reference frame.

static IDENTITY_ROT: LazyLock<Rotation3<f64>> = LazyLock::new(Rotation3::identity);

static ECLIPTIC_EQUATORIAL_ROT: LazyLock<Rotation3<f64>> =
    LazyLock::new(|| axis_rotation::<'X'>(OBLIQUITY));

// The other rotations are written as products of rotations about the coordinate axes,
// with their angles in arcseconds. FK4, Galactic and the B1950 ecliptic are defined
// from B1950 and FK4, and compose those rotations once, so every static frame is a
// single rotation to equatorial.

static B1950_EQUATORIAL_ROT: LazyLock<Rotation3<f64>> = LazyLock::new(|| {
    // B1950 precessed to J2000 with the IAU 1976 precession: z, theta and zeta.
    let (rot, _) = euler_rotation::<'Z', 'Y', 'Z'>(
        &[
            arcsec(1153.04066200330),
            arcsec(-1002.26108439117),
            arcsec(1152.84248596724),
        ],
        &[0.0; 3],
    );
    rot
});

static FK4_EQUATORIAL_ROT: LazyLock<Rotation3<f64>> =
    // B1950 with the equinox offset of Fricke.
    LazyLock::new(|| *B1950_EQUATORIAL_ROT * axis_rotation::<'Z'>(arcsec(0.525)));

static GALACTIC_EQUATORIAL_ROT: LazyLock<Rotation3<f64>> = LazyLock::new(|| {
    // Galactic System II, defined from FK4.
    let (galactic_fk4, _) = euler_rotation::<'Z', 'X', 'Z'>(
        &[arcsec(1_016_100.0), arcsec(225_360.0), arcsec(1_177_200.0)],
        &[0.0; 3],
    );
    *FK4_EQUATORIAL_ROT * galactic_fk4
});

static ECLIPB1950_EQUATORIAL_ROT: LazyLock<Rotation3<f64>> =
    // B1950 with the obliquity of the ecliptic at B1950.
    LazyLock::new(|| *B1950_EQUATORIAL_ROT * axis_rotation::<'X'>(arcsec(84_404.836)));

static MARSIAU_EQUATORIAL_ROT: LazyLock<Rotation3<f64>> = LazyLock::new(|| {
    let (rot, _) = euler_rotation::<'Z', 'Y', 'Z'>(
        &[arcsec(-152_348.4), arcsec(133_610.4), arcsec(324_000.0)],
        &[0.0; 3],
    );
    rot
});

/// `angle` in arcseconds, in radians.
fn arcsec(angle: f64) -> f64 {
    (angle / 3600.0).to_radians()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::FrameTable;

    #[test]
    fn test_ecliptic_rot_roundtrip() {
        let vec = Ecliptic::to_equatorial([1.0, 2.0, 3.0].into());
        let vec_return = Ecliptic::from_equatorial(vec);
        assert!((1.0 - vec_return.x).abs() <= 10.0 * f64::EPSILON);
        assert!((2.0 - vec_return.y).abs() <= 10.0 * f64::EPSILON);
        assert!((3.0 - vec_return.z).abs() <= 10.0 * f64::EPSILON);
    }
    #[test]
    fn test_fk4_roundtrip() {
        let vec = FK4::to_equatorial([1.0, 2.0, 3.0].into());
        let vec_return = FK4::from_equatorial(vec);
        assert!((1.0 - vec_return.x).abs() <= 10.0 * f64::EPSILON);
        assert!((2.0 - vec_return.y).abs() <= 10.0 * f64::EPSILON);
        assert!((3.0 - vec_return.z).abs() <= 10.0 * f64::EPSILON);
    }
    #[test]
    fn test_galactic_rot_roundtrip() {
        let vec = Galactic::to_equatorial([1.0, 2.0, 3.0].into());
        let vec_return = Galactic::from_equatorial(vec);
        assert!((1.0 - vec_return.x).abs() <= 10.0 * f64::EPSILON);
        assert!((2.0 - vec_return.y).abs() <= 10.0 * f64::EPSILON);
        assert!((3.0 - vec_return.z).abs() <= 10.0 * f64::EPSILON);
    }

    #[test]
    fn test_instant_frame_rot_roundtrip() {
        let angles = [0.11, 0.21, 0.31];
        let rates = [0.41, 0.51, 0.61];
        let pos = [1.0, 2.0, 3.0];
        let vel = [0.1, 0.2, 0.3];
        let frame = InstantFrame::from_euler::<'Z', 'X', 'Z'>(
            0_f64,
            angles,
            rates,
            DynFrameId::UNNAMED,
            DynFrameId::ECLIPJ2000,
        );
        let (r_pos, r_vel) = frame.to_equatorial(pos, vel).unwrap();
        let (pos_return, vel_return) = frame.from_equatorial(r_pos, r_vel).unwrap();

        assert!((1.0 - pos_return.x).abs() <= 10.0 * f64::EPSILON);
        assert!((2.0 - pos_return.y).abs() <= 10.0 * f64::EPSILON);
        assert!((3.0 - pos_return.z).abs() <= 10.0 * f64::EPSILON);
        assert!((0.1 - vel_return.x).abs() <= 10.0 * f64::EPSILON);
        assert!((0.2 - vel_return.y).abs() <= 10.0 * f64::EPSILON);
        assert!((0.3 - vel_return.z).abs() <= 10.0 * f64::EPSILON);
    }

    /// A frame with no rotation rate gives its rotation, but a velocity transformation
    /// is an error rather than one that assumes the frame does not rotate.
    #[test]
    fn missing_rate_rotates_but_does_not_transform_velocity() {
        let rotation = euler_rotation::<'Z', 'X', 'Z'>(&[0.11, 0.21, 0.31], &[0.0; 3]).0;
        let pos = Vector3::new(1.0, 2.0, 3.0);
        let vel = Vector3::new(0.1, 0.2, 0.3);
        for reference in [DynFrameId::J2000, DynFrameId::ECLIPJ2000] {
            let frame =
                InstantFrame::from_rotations(0_f64, rotation, None, DynFrameId::UNNAMED, reference);
            let with_rate = InstantFrame::from_rotations(
                0_f64,
                rotation,
                Some(Matrix3::zeros()),
                DynFrameId::UNNAMED,
                reference,
            );
            assert_eq!(
                frame.try_relative_to_equatorial().unwrap().rotation,
                with_rate.rotation_and_rate_to_equatorial().unwrap().0
            );
            assert!(frame.rotation_and_rate_to_equatorial().is_err());
            assert!(frame.to_equatorial(pos, vel).is_err());
            assert!(frame.from_equatorial(pos, vel).is_err());
        }
    }

    /// Each supported inertial reference applies its frame's rotation to equatorial;
    /// any other reference is an error.
    #[test]
    fn rotation_and_rate_to_equatorial_by_reference() {
        let rot = Rotation3::from_euler_angles(0.1, -0.2, 0.3);
        let rate = Matrix3::new(0.0, -1e-3, 0.0, 1e-3, 0.0, 0.0, 0.0, 0.0, 0.0);
        let frame = |id| {
            InstantFrame::from_rotations(
                Time::<TDB>::new(2_451_545.0),
                rot,
                Some(rate),
                DynFrameId::UNNAMED,
                id,
            )
        };
        let cases = [
            (DynFrameId::J2000, Equatorial::rotation_to_equatorial()),
            (DynFrameId::FK4, FK4::rotation_to_equatorial()),
            (DynFrameId::GALACTIC, Galactic::rotation_to_equatorial()),
            (DynFrameId::ECLIPJ2000, Ecliptic::rotation_to_equatorial()),
            (DynFrameId::B1950, B1950::rotation_to_equatorial()),
            (
                DynFrameId::ECLIPB1950,
                EclipticB1950::rotation_to_equatorial(),
            ),
            (DynFrameId::MARSIAU, MarsIAU::rotation_to_equatorial()),
        ];
        for (id, to_eq) in cases {
            let (r, dr) = frame(id).rotation_and_rate_to_equatorial().unwrap();
            assert!((r.matrix() - (to_eq * rot).matrix()).norm() < 1e-15);
            assert!((dr - to_eq * rate).norm() < 1e-15);
        }
        assert!(
            frame(DynFrameId(4))
                .rotation_and_rate_to_equatorial()
                .is_err()
        );
        assert!(
            frame(DynFrameId(-1000))
                .rotation_and_rate_to_equatorial()
                .is_err()
        );
    }

    /// Each static frame's ID looks up its rotation to equatorial.
    #[test]
    fn static_frames_by_id() {
        fn check<F: StaticFrame>() {
            assert_eq!(
                DynFrameId::static_rotation_to_equatorial(F::ID),
                Some(F::rotation_to_equatorial())
            );
        }
        check::<Equatorial>();
        check::<Ecliptic>();
        check::<B1950>();
        check::<FK4>();
        check::<Galactic>();
        check::<EclipticB1950>();
        check::<MarsIAU>();
    }

    /// A frame source holding one body frame, `BODY`, relative to the ecliptic.
    struct OneFrame {
        rotation: Rotation3<f64>,
        rate: Matrix3<f64>,
    }

    const BODY: DynFrameId = DynFrameId(-500);

    impl FrameSource for OneFrame {
        fn try_link(&self, frame_id: DynFrameId, time: Time<TDB>) -> KeteResult<InstantFrame> {
            if frame_id != BODY {
                return FrameTable::new().try_link(frame_id, time);
            }
            Ok(InstantFrame::from_rotations(
                time,
                self.rotation,
                Some(self.rate),
                BODY,
                DynFrameId::ECLIPJ2000,
            ))
        }
    }

    fn one_frame() -> OneFrame {
        OneFrame {
            rotation: Rotation3::from_euler_angles(0.4, 0.1, -0.7),
            rate: Matrix3::new(0.0, -2e-3, 0.0, 2e-3, 0.0, 0.0, 0.0, 0.0, 0.0),
        }
    }

    /// An offset frame on a body frame turns with the body; on a static frame it needs
    /// no source.
    #[test]
    fn fixed_frame_resolves_its_reference() {
        let time = Time::<TDB>::new(2_451_545.0);
        let source = one_frame();
        let offset = Rotation3::from_euler_angles(-0.2, 0.3, 0.05);

        let on_body = OffsetFrame::new(DynFrameId(-501), offset, BODY);
        let rot = on_body.try_at(time, &source).unwrap().rotation;
        let expected = Ecliptic::rotation_to_equatorial() * source.rotation * offset;
        assert!((rot.matrix() - expected.matrix()).norm() < 1e-15);
        assert!(on_body.try_at(time, &FrameTable::new()).is_err());

        let on_static = OffsetFrame::new(DynFrameId::UNNAMED, offset, DynFrameId::FK4);
        let rot = on_static.try_at(time, &FrameTable::new()).unwrap().rotation;
        let expected = FK4::rotation_to_equatorial() * offset;
        assert!((rot.matrix() - expected.matrix()).norm() < 1e-15);
    }

    /// A static frame as an offset frame keeps its ID, reference and rotation.
    #[test]
    fn static_frame_as_fixed_frame() {
        let time = Time::<TDB>::new(2_451_545.0);
        let galactic = OffsetFrame::from(Galactic {});
        assert_eq!(galactic.frame, DynFrameId::GALACTIC);
        assert_eq!(galactic.reference, DynFrameId::J2000);
        let rot = galactic.try_at(time, &FrameTable::new()).unwrap().rotation;
        let err = (rot.matrix() - Galactic::rotation_to_equatorial().matrix()).norm();
        assert!(err < 1e-15, "{err:e}");
    }

    /// An offset frame on a body frame carries the body's rate.
    #[test]
    fn fixed_frame_turns_with_its_reference() {
        let time = Time::<TDB>::new(2_451_545.0);
        let source = one_frame();
        let offset = Rotation3::from_euler_angles(-0.2, 0.3, 0.05);
        let frame = OffsetFrame::new(DynFrameId(-501), offset, BODY)
            .try_at(time, &source)
            .unwrap();
        let to_eq = Ecliptic::rotation_to_equatorial();
        assert_eq!(frame.frame, DynFrameId(-501));
        let rate = frame.rotation_rate.unwrap();
        assert!((rate - to_eq * source.rate * offset.matrix()).norm() < 1e-15);
    }

    /// Composition follows the reference: a frame on the body frame, composed with the
    /// body frame, is relative to the body's reference, with the product rate. A frame
    /// that is not the reference, or not at the same time, is an error, and an unnamed
    /// reference matches no frame.
    #[test]
    fn relative_to_follows_the_reference() {
        let time = Time::<TDB>::new(2_451_545.0);
        let source = one_frame();
        let offset = Rotation3::from_euler_angles(-0.2, 0.3, 0.05);
        let frame = InstantFrame::from_rotations(
            time,
            offset,
            Some(Matrix3::zeros()),
            DynFrameId(-501),
            BODY,
        );
        let body = source.try_frame(BODY, time).unwrap();
        let composed = frame.try_relative_to(&body).unwrap();
        let to_eq = Ecliptic::rotation_to_equatorial();
        assert_eq!(composed.frame, DynFrameId(-501));
        let expected = to_eq * source.rotation * offset;
        assert!((composed.rotation.matrix() - expected.matrix()).norm() < 1e-15);
        let rate = composed.rotation_rate.unwrap();
        assert!((rate - to_eq * source.rate * offset.matrix()).norm() < 1e-15);

        let other = InstantFrame::identity(time, DynFrameId(-502));
        assert!(frame.try_relative_to(&other).is_err());

        let later = source.try_frame(BODY, time + 1.0).unwrap();
        assert!(frame.try_relative_to(&later).is_err());

        let unnamed =
            InstantFrame::from_rotations(time, offset, None, DynFrameId(-501), DynFrameId::UNNAMED);
        let base = InstantFrame::identity(time, DynFrameId::UNNAMED);
        assert!(unnamed.try_relative_to(&base).is_err());
    }

    /// With a static reference the rotation to equatorial needs no check, and agrees
    /// with the runtime reference.
    #[test]
    fn static_reference_matches_runtime_reference() {
        let time = Time::<TDB>::new(2_451_545.0);
        let rot = Rotation3::from_euler_angles(0.4, 0.1, -0.7);
        let typed = InstantFrame::from_rotations(time, rot, None, DynFrameId(-501), Galactic {});
        let dynamic =
            InstantFrame::from_rotations(time, rot, None, DynFrameId(-501), DynFrameId::GALACTIC);
        assert_eq!(
            typed.rotation_to_equatorial(),
            dynamic.try_relative_to_equatorial().unwrap().rotation
        );
    }

    /// The rate from an angular velocity is the derivative of a rotation turning at
    /// that angular velocity.
    #[test]
    fn angular_velocity_rate_is_the_derivative() {
        let w = Vector3::new(0.3, -0.2, 0.5);
        let start = Rotation3::from_euler_angles(0.4, 0.1, -0.7);
        let at = |dt: f64| *(Rotation3::new(w * dt) * start).matrix();
        let frame = InstantFrame::from_angular_velocity(
            0_f64,
            start,
            Some(w),
            DynFrameId::UNNAMED,
            DynFrameId::J2000,
        );
        let h = 1e-6;
        let numeric = (at(h) - at(-h)) / (2.0 * h);
        assert!((frame.rotation_rate.unwrap() - numeric).abs().max() < 1e-9);
    }

    /// A reference that is not a static frame is a structural error, not missing data,
    /// so it is never mistaken for a time a source does not cover.
    #[test]
    fn non_static_reference_is_a_value_error() {
        let frame = InstantFrame::from_rotations(
            Time::<TDB>::new(2_451_545.0),
            Rotation3::identity(),
            None,
            DynFrameId(-501),
            DynFrameId(-500),
        );
        assert!(matches!(
            frame.try_relative_to_equatorial(),
            Err(Error::ValueError(_))
        ));
    }

    /// Each static frame matches reference rotations from the frame to J2000.
    #[test]
    fn static_frames_match_spice() {
        fn check<F: StaticFrame>(expected: [[f64; 3]; 3]) {
            let expected = Matrix3::from_fn(|i, j| expected[i][j]);
            let err = (F::rotation_to_equatorial().matrix() - expected)
                .abs()
                .max();
            assert!(err < 1e-15, "{:?}: {err:e}", F::ID);
        }
        check::<B1950>([
            [
                0.9999257079523629,
                -0.011178938137770135,
                -0.00485900381535927,
            ],
            [
                0.01117893812642769,
                0.9999375133499887,
                -2.716259471424704e-05,
            ],
            [
                0.0048590038414544285,
                -2.7157926258510777e-05,
                0.9999881946023742,
            ],
        ]);
        check::<FK4>([
            [
                0.9999256794956877,
                -0.01118148322046629,
                -0.00485900381535927,
            ],
            [
                0.011181483239171792,
                0.9999374848933135,
                -2.716259471424704e-05,
            ],
            [
                0.004859003772314385,
                -2.7170293744002025e-05,
                0.9999881946023742,
            ],
        ]);
        check::<Galactic>([
            [
                -0.05487553939574253,
                0.4941094536277439,
                -0.8676661356833737,
            ],
            [
                -0.8734371047275962,
                -0.44482959429757496,
                -0.19807638961301985,
            ],
            [-0.4838349917700252, 0.7469822486998919, 0.4559837945214199],
        ]);
        check::<Ecliptic>([
            [1.0, 0.0, 0.0],
            [0.0, 0.9174820620691818, -0.3977771559319137],
            [0.0, 0.3977771559319137, 0.9174820620691818],
        ]);
        check::<EclipticB1950>([
            [
                0.9999257079523629,
                -0.012189277138214924,
                -9.94050092035086e-06,
            ],
            [0.01117893812642769, 0.9173688178789828, -0.3978812427417045],
            [
                0.0048590038414544285,
                0.39785157220522016,
                0.9174369278459982,
            ],
        ]);
        check::<MarsIAU>([
            [0.673257747460025, -0.5896308378262533, 0.44616082366044196],
            [0.739407874914146, 0.536880310821634, -0.40624564781301037],
            [
                -3.6947768825436786e-17,
                0.6034028562547383,
                0.7974365135003686,
            ],
        ]);
    }
}
