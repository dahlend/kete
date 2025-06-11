//! # kete Core
//! This library contains all of the mathematical functions required for kete to
//! function.
//!
//! This crate is left as a stand alone Rust crate, completely independent of the
//! Python wrappers. This is done intentionally, as it makes these functions available
//! outside of the python module so that wrappers may be written for other languages
//! later.
//!

pub mod cache;
pub mod constants;
pub mod desigs;
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
pub mod util;

/// Common useful imports
pub mod prelude {
    pub use crate::desigs::Desig;
    pub use crate::elements::CometElements;
    pub use crate::errors::{Error, KeteResult};
    pub use crate::flux::{
        CometMKParams, FrmParams, HGParams, NeatmParams, black_body_flux, frm_facet_temperature,
        lambertian_flux, neatm_facet_temperature,
    };
    pub use crate::frames::{Ecliptic, Equatorial, FK4, Galactic, NonInertialFrame};
    pub use crate::propagation::{propagate_n_body_spk, propagate_two_body};
    pub use crate::simult_states::SimultaneousStates;
    pub use crate::spice::{LOADED_PCK, LOADED_SPK};
    pub use crate::state::State;
}
