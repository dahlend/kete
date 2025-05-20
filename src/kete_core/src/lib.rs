//! # kete Core
//! This library contains all of the mathematical functions required for kete to
//! function.
//!
//! This crate is left as a stand alone Rust crate, completely independent of the
//! Python wrappers. This is done intentionally, as it makes these functions available
//! outside of the python module so that wrappers may be written for other languages
//! later.
//!

#![deny(
    bad_style,
    dead_code,
    improper_ctypes,
    non_shorthand_field_patterns,
    no_mangle_generic_items,
    overflowing_literals,
    path_statements,
    patterns_in_fns_without_body,
    unconditional_recursion,
    unused,
    while_true,
    missing_debug_implementations,
    missing_docs,
    trivial_casts,
    trivial_numeric_casts,
    unused_extern_crates,
    unused_import_braces,
    unused_qualifications,
    unused_results
)]

pub mod cache;
pub mod constants;
pub mod elements;
pub mod errors;
pub mod fitting;
pub mod flux;
pub mod fov;
pub mod frames;
pub mod io;
pub mod propagation;
pub mod simult_states;
pub mod spice;
pub mod state;
pub mod stats;
pub mod time;

/// Common useful imports
pub mod prelude {
    pub use crate::elements::CometElements;
    pub use crate::errors::{Error, KeteResult};
    pub use crate::flux::{
        black_body_flux, frm_facet_temperature, lambertian_flux, neatm_facet_temperature,
        CometMKParams, FrmParams, HGParams, NeatmParams,
    };
    pub use crate::frames::{Ecliptic, EclipticNonInertial, Equatorial, Galactic, FK4};
    pub use crate::propagation::{propagate_n_body_spk, propagate_two_body};
    pub use crate::simult_states::SimultaneousStates;
    pub use crate::spice::{LOADED_PCK, LOADED_SPK};
    pub use crate::state::{Desig, State};
}
