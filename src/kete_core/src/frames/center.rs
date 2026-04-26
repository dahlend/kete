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
pub trait CenterBody: Sized + Sync + Send + Clone + Copy + Debug + PartialEq {
    /// NAIF id of the center body.
    fn center_id(&self) -> i32;
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
