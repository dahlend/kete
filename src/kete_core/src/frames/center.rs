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
/// Implementors expose a NAIF id, either as a compile-time const
/// ([`NAIF_ID`](Self::NAIF_ID)) for the typed centers ([`SSB`],
/// [`SunCenter`], [`EarthCenter`]) or via a runtime instance method
/// ([`center_id`](Self::center_id)) for [`DynCenter`].
///
/// `Into<DynCenter>` is a supertrait so any [`CenterBody`] can be
/// type-erased to [`DynCenter`] without callers repeating the bound.
pub trait CenterBody:
    Sized + Sync + Send + Clone + Copy + Debug + PartialEq + Into<DynCenter>
{
    /// NAIF id of this center body, known at compile time.
    ///
    /// Typed centers report their actual id. [`DynCenter`] reports `i32::MIN`
    /// as a sentinel since its id is per-instance; consult
    /// [`center_id`](Self::center_id) for the runtime value instead.
    const NAIF_ID: i32;

    /// Runtime NAIF id of this center body.
    ///
    /// Defaults to [`Self::NAIF_ID`]; [`DynCenter`] overrides to return
    /// its stored runtime value.
    #[inline(always)]
    fn center_id(&self) -> i32 {
        Self::NAIF_ID
    }
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
    const NAIF_ID: i32 = i32::MIN;

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
    const NAIF_ID: i32 = 0;
}

impl CenterBody for SunCenter {
    const NAIF_ID: i32 = 10;
}

impl CenterBody for EarthCenter {
    const NAIF_ID: i32 = 399;
}
