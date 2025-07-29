//! # Flux
//! Flux calculations including thermal and reflected light models.
//!
//! There are a few flux calculation models contained here:
//! [`HGParams`] - Flux calculations of an object using the HG system.
//! [`NeatmParams`] - The NEATM thermal model to compute black body flux.
//! [`FrmParams`] - The FRM thermal model to compute black body flux.
//!
//
// BSD 3-Clause License
//
// Copyright (c) 2025, Dar Dahlen
// Copyright (c) 2025, California Institute of Technology
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
