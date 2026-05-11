//! Center body types for [`State`](crate::state::State).
//!
//! These types encode the gravitational center of a `State` at compile time,
//! turning center-mismatch errors into type errors.
//!
//! The default type parameter is [`DynCenter`], which carries the center NAIF
//! id at runtime.
//! The typed variants [`SSB`], [`SunCenter`], and [`EarthCenter`] are zero-sized
//! and guarantee a specific center at compile time.
use std::fmt::Debug;

/// Trait for center-body types used in [`State`](crate::state::State).
///
/// Implementors provide the NAIF id of the center body at runtime.
pub trait CenterBody: Sized + Sync + Send + Clone + Copy + Debug + PartialEq
where
    DynCenter: From<Self>,
{
    /// NAIF id of the center body.
    fn center_id(&self) -> i32;
}

/// Marker trait for center-body types that know their NAIF id at compile
/// time.
///
/// Distinct from [`CenterBody`] because it has no `DynCenter: From<Self>`
/// requirement, which keeps it dyn-compatible -- the [`Force`] trait's
/// `Center` associated type can require `NaifBody` without breaking
/// `Box<dyn Force<...>>`. Used by the `Recenter` adapter
/// (in `kete_spice`) to look up reference-body shifts from SPK at the
/// type level rather than via runtime IDs.
///
/// [`Force`]: crate::forces::Force
pub trait NaifBody: Send + Sync + Copy + 'static {
    /// NAIF id of this center body, known at compile time.
    const NAIF_ID: i32;
}

impl NaifBody for SSB {
    const NAIF_ID: i32 = 0;
}

impl NaifBody for SunCenter {
    const NAIF_ID: i32 = 10;
}

impl NaifBody for EarthCenter {
    const NAIF_ID: i32 = 399;
}

/// Runtime-determined center body -- the default.
///
/// Carries the NAIF center id at runtime; states may be re-centered via
/// ``SpkCollection::try_change_center`` in `kete_spice`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DynCenter(pub i32);

/// Solar System Barycenter (NAIF ID 0).
///
/// States typed with `SSB` are guaranteed to be SSB-centered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SSB;

/// Sun-centered (NAIF ID 10).
///
/// States typed with `SunCenter` are guaranteed to be heliocentric.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SunCenter;

/// Earth-centered (NAIF ID 399).
///
/// States typed with `EarthCenter` are guaranteed to be geocentric.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EarthCenter;

impl CenterBody for DynCenter {
    #[inline(always)]
    fn center_id(&self) -> i32 {
        self.0
    }
}

impl From<i32> for DynCenter {
    fn from(id: i32) -> Self {
        Self(id)
    }
}

impl From<SSB> for DynCenter {
    fn from(_: SSB) -> Self {
        Self(0)
    }
}

impl From<SunCenter> for DynCenter {
    fn from(_: SunCenter) -> Self {
        Self(10)
    }
}

impl From<EarthCenter> for DynCenter {
    fn from(_: EarthCenter) -> Self {
        Self(399)
    }
}

impl CenterBody for SSB {
    #[inline(always)]
    fn center_id(&self) -> i32 {
        0
    }
}

impl CenterBody for SunCenter {
    #[inline(always)]
    fn center_id(&self) -> i32 {
        10
    }
}

impl CenterBody for EarthCenter {
    #[inline(always)]
    fn center_id(&self) -> i32 {
        399
    }
}
