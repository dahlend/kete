// BSD 3-Clause License
//
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

/// effective surface temperature of the sun in K.
pub const SUN_TEMP: f64 = 5778.0;

/// Sun diameter in km
pub const SUN_DIAMETER: f64 = 1.3914e6;

/// W/m^2 measured at 1 AU.
pub const SOLAR_FLUX: f64 = 1360.8;

/// Speed of light in AU / Day
pub const C_AU_PER_DAY: f64 = 173.14463267424034;

/// Speed of light in meters / second (Definition)
pub const C_M_PER_S: f64 = 299792458.0;

/// Inverse of the Speed of light in Day / AU
pub const C_AU_PER_DAY_INV: f64 = 0.005775518331436995;

/// Square of the inverse of the speed of light in Day^2 / AU^2
pub const C_AU_PER_DAY_INV_SQUARED: f64 = 3.33566119967647e-05;

/// Km per AU (Definition)
pub const AU_KM: f64 = 149597870.7;

/// Stefan Boltzmann constant in W / (m^2 * K^4)
pub const STEFAN_BOLTZMANN: f64 = 5.670374419e-8;

/// Golden Ratio, unit-less
pub const GOLDEN_RATIO: f64 = 1.618033988749894;

/// Zero point V band magnitude in Jy. This is approximately equivalent to Vega's flux.
/// <https://coolwiki.ipac.caltech.edu/index.php/Central_wavelengths_and_zero_points>
pub const V_MAG_ZERO: f64 = 3597.28;

/// V-band constant for the relationship between `D`, `H_V`, and `p_v`, in km.
pub const C_V: f64 = 1329.0;
