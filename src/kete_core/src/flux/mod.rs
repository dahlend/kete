//! # Flux
//! Flux calculations including thermal and reflected light models.
//!
//! There are a few flux calculation models contained here:
//! [`HGParams`] - Flux calculations of an object using the HG system.
//! [`NeatmParams`] - The NEATM thermal model to compute black body flux.
//! [`FrmParams`] - The FRM thermal model to compute black body flux.
//!
mod comets;
mod common;
mod frm;
mod neatm;
mod reflected;
mod shapes;
mod sun;

pub use self::comets::CometMKParams;
pub use self::common::{
    ColorCorrFn, ModelResults, ObserverBands, black_body_flux, flux_to_mag, lambertian_flux,
    lambertian_vis_scale_factor, mag_to_flux, sub_solar_temperature,
};
pub use self::frm::{FrmParams, frm_facet_temperature};
pub use self::neatm::{NeatmParams, neatm_facet_temperature};
pub use self::reflected::{
    HGParams, cometary_dust_phase_curve_correction, hg_phase_curve_correction,
};
pub use self::shapes::{ConvexShape, DEFAULT_SHAPE, Facet};
pub use self::sun::{solar_flux, solar_flux_black_body};
