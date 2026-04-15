//! # Spitzer Space Telescope constants.
// BSD 3-Clause License
//
// Copyright (c) 2026, Dar Dahlen
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

/// Spitzer/IRAC effective channel wavelengths in nanometers for channels 1-4.
/// Ch1 = 3.6 um, Ch2 = 4.5 um, Ch3 = 5.8 um, Ch4 = 8.0 um.
/// Source: IRAC Instrument Handbook v2.1, Table 4.1 (Reach et al. 2005).
pub const IRAC_BANDS: [f64; 4] = [3550.0, 4493.0, 5731.0, 7872.0];

/// Vega-system zero-magnitude flux densities in Jy for IRAC channels 1-4.
/// Source: IRAC Instrument Handbook v2.1, Table 4.1.
pub const IRAC_ZERO_MAG: [f64; 4] = [280.9, 179.7, 115.0, 64.13];

/// IRAC field of view width, 5.21 arcminutes in radians.
/// All 4 channels share the same 5.21' x 5.21' FOV size.
pub const IRAC_WIDTH: f64 = 0.001515527;

/// Spitzer/MIPS effective wavelengths in nanometers for the 24, 70, and 160 um bands.
/// Source: MIPS Instrument Handbook (Rieke et al. 2004).
pub const MIPS_BANDS: [f64; 3] = [23680.0, 71420.0, 155900.0];

/// Vega-system zero-magnitude flux densities in Jy for MIPS 24, 70, and 160 um.
/// Sources: Engelbracht et al. 2007 (24um), Gordon et al. 2007 (70um),
/// Stansberry et al. 2007 (160um).
pub const MIPS_ZERO_MAG: [f64; 3] = [7.17, 0.778, 0.159];

/// Spitzer/IRS Peak-Up Imager effective wavelengths in nanometers for Blue and Red.
/// Blue = 13.3-18.7 um bandpass, Red = 18.5-26.0 um bandpass.
/// Source: IRS Instrument Handbook v5.0.
pub const IRS_PU_BANDS: [f64; 2] = [15800.0, 22300.0];

/// Vega-system zero-magnitude flux densities in Jy for IRS Peak-Up Blue and Red.
/// Derived from the Cohen (1999) Vega spectral model at the effective wavelengths.
pub const IRS_PU_ZERO_MAG: [f64; 2] = [15.6, 7.80];
