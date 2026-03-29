//! # `kete_spice`
//!
//! SPICE kernel I/O, SPK-dependent propagation, and SPICE-related extensions for kete.
//!
//! This crate provides:
//! - SPICE kernel reading (SPK, PCK, CK, SCLK) via the [`spice`] module
//! - SPK-dependent N-body propagation via the [`propagation`] module
//! - FOV SPICE-dependent visibility checks via the [`fov_ext`] module
//! - CK-dependent frame rotation via the [`frame_ext`] module
//!
//! Dependency direction: `kete_spice -> kete_core` (one-way, no cycles).

#![deny(missing_docs)]
#![deny(missing_debug_implementations)]

pub mod fov_ext;
pub mod frame_ext;
pub mod propagation;
pub mod spice;

mod jacobian;
mod state_transition;

pub use state_transition::compute_state_transition;
