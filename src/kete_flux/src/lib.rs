// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! # Flux
//! Thermal and reflected light flux models.
//!
//! Available models:
//! - HG system: [`hg_apparent_mag`], [`hg_apparent_flux`], [`hg_phase_curve_correction`]
//! - NEATM thermal model: [`neatm_thermal_flux`], [`neatm_total_flux`]
//! - FRM thermal model: [`frm_thermal_flux`], [`frm_total_flux`]

mod comets;
mod common;
pub mod fitting;
mod frm;
mod neatm;
mod reflected;
mod sun;

pub use self::comets::CometMKParams;
pub(crate) use self::common::assemble_total;
pub use self::common::{
    BandInfo, ColorCorrFn, ModelResults, black_body_flux, bond_albedo, flux_to_mag,
    lambertian_flux, mag_to_flux, sub_solar_temperature,
};
pub use self::frm::{frm_facet_temperature, frm_thermal_flux, frm_total_flux};
pub use self::neatm::{neatm_facet_temperature, neatm_thermal_flux, neatm_total_flux};
pub use self::reflected::{
    albedo_from_h_mag_diam, cometary_dust_phase_curve_correction, diam_from_h_mag_albedo,
    h_mag_from_diam_albedo, hg_apparent_flux, hg_apparent_mag, hg_phase_curve_correction,
    resolve_hg_params,
};
pub use self::sun::{solar_flux, solar_flux_black_body};
