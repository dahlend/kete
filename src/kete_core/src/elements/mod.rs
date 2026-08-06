//! # Orbital Elements
//!
//! Two-body orbital element sets and their conversions to and from
//! [`State`](crate::state::State).
//!
//! [`CometElements`] is the classical set, defined by the perihelion distance,
//! eccentricity and three angles. It is what external catalogs publish and what
//! kete reports.
//!
//! [`EquinoctialElements`] is the modified equinoctial set: six unconstrained floats
//! with no singularity at zero eccentricity or zero inclination, which is what
//! [`UncertainState`](crate::state::UncertainState) stores a covariance over.
//
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

mod cometary;
mod equinoctial;

pub use cometary::CometElements;
pub use equinoctial::EquinoctialElements;

use crate::errors::Error;
use crate::forces::GravParams;
use crate::prelude::KeteResult;

use nalgebra::Vector3;

/// Square root of the gravitational parameter of a NAIF center id.
///
/// Elements carry `gm_sqrt` rather than `mu`, since that is the form every conversion
/// below uses. The lookup and its error live on [`GravParams::try_mass_from_naif_id`].
///
/// # Errors
/// Fails if `center_id` has no entry in the mass table.
fn gm_sqrt_for_center(center_id: i32) -> KeteResult<f64> {
    Ok(GravParams::try_mass_from_naif_id(center_id)?.sqrt())
}

/// [`CometElements`] is defined by ecliptic angles, which is the frame
/// [`EquinoctialElements`] stores, so the conversion needs no rotation.
impl TryFrom<&CometElements> for EquinoctialElements {
    type Error = Error;

    fn try_from(elem: &CometElements) -> KeteResult<Self> {
        let [pos, vel] = elem.to_pos_vel()?;
        Self::from_pos_vel(
            elem.desig.clone(),
            elem.epoch,
            &Vector3::from(pos),
            &Vector3::from(vel),
            elem.center_id,
            elem.gm_sqrt,
        )
    }
}

impl TryFrom<&EquinoctialElements> for CometElements {
    type Error = Error;

    fn try_from(elem: &EquinoctialElements) -> KeteResult<Self> {
        let [pos, vel] = elem.to_pos_vel()?;
        Ok(Self::from_pos_vel(
            elem.desig.clone(),
            elem.epoch,
            &Vector3::from(pos),
            &Vector3::from(vel),
            elem.center_id,
            elem.gm_sqrt,
        ))
    }
}
