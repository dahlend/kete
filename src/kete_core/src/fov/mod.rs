//! # Field of View
//! On-Sky field of view checks.
// BSD 3-Clause License
//
// Copyright (c) 2026, Dar Dahlen
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

mod fov_like;
mod generic;
mod neos;
mod patches;
mod ptf;
mod spherex;
mod wise;
mod ztf;

pub use self::fov_like::FovLike;
pub use self::generic::{GenericCone, GenericRectangle, OmniDirectional};
pub use self::neos::{NeosCmos, NeosVisit};
pub use self::patches::{Contains, OnSkyRectangle, SkyPatch, SphericalCone, SphericalPolygon};
pub use self::ptf::{PTFFilter, PtfCcd, PtfField};
pub use self::spherex::{SpherexCmos, SpherexField};
pub use self::wise::WiseCmos;
pub use self::ztf::{ZtfCcdQuad, ZtfField};

use serde::{Deserialize, Serialize};

use crate::{
    errors::Error,
    frames::{Equatorial, Vector},
    prelude::*,
    propagation::{light_time_correct, propagate_two_body},
};

/// Allowed FOV objects, either contiguous or joint.
/// Many of these exist solely to carry additional metadata.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[non_exhaustive]
pub enum FOV {
    /// Omni-Directional FOV.
    OmniDirectional(OmniDirectional),

    /// Generic cone FOV without any additional metadata.
    GenericCone(GenericCone),

    /// Generic rectangle FOV without any additional metadata.
    GenericRectangle(GenericRectangle),

    /// WISE or NEOWISE FOV.
    Wise(WiseCmos),

    /// NEOS single cmos FOV.
    NeosCmos(NeosCmos),

    /// NEOS Visit.
    NeosVisit(NeosVisit),

    /// ZTF Single Quad of single CCD FOV.
    ZtfCcdQuad(ZtfCcdQuad),

    /// Full ZTF field of up to 64 individual files.
    ZtfField(ZtfField),

    /// Single PTF CCD image.
    PtfCcd(PtfCcd),

    /// Full PTF field of multiple ccd images.
    PtfField(PtfField),

    /// Spherex CMOS
    SpherexCmos(SpherexCmos),

    /// Spherex Field, containing up to 6 CMOS frames.
    SpherexField(SpherexField),
}

impl FOV {
    /// Observer position in this FOV
    pub fn observer(&self) -> &State<Equatorial> {
        match self {
            Self::Wise(fov) => fov.observer(),
            Self::NeosCmos(fov) => fov.observer(),
            Self::ZtfCcdQuad(fov) => fov.observer(),
            Self::GenericCone(fov) => fov.observer(),
            Self::GenericRectangle(fov) => fov.observer(),
            Self::ZtfField(fov) => fov.observer(),
            Self::NeosVisit(fov) => fov.observer(),
            Self::OmniDirectional(fov) => fov.observer(),
            Self::PtfCcd(fov) => fov.observer(),
            Self::PtfField(fov) => fov.observer(),
            Self::SpherexCmos(fov) => fov.observer(),
            Self::SpherexField(fov) => fov.observer(),
        }
    }

    /// Check if static sources are visible in this FOV.
    #[must_use]
    pub fn check_statics(&self, pos: &[Vector<Equatorial>]) -> Vec<Option<(Vec<usize>, Self)>> {
        match self {
            Self::Wise(fov) => fov.check_statics(pos),
            Self::NeosCmos(fov) => fov.check_statics(pos),
            Self::ZtfCcdQuad(fov) => fov.check_statics(pos),
            Self::GenericCone(fov) => fov.check_statics(pos),
            Self::GenericRectangle(fov) => fov.check_statics(pos),
            Self::ZtfField(fov) => fov.check_statics(pos),
            Self::NeosVisit(fov) => fov.check_statics(pos),
            Self::OmniDirectional(fov) => fov.check_statics(pos),
            Self::PtfCcd(fov) => fov.check_statics(pos),
            Self::PtfField(fov) => fov.check_statics(pos),
            Self::SpherexCmos(fov) => fov.check_statics(pos),
            Self::SpherexField(fov) => fov.check_statics(pos),
        }
    }
}

macro_rules! dispatch_fov {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            Self::Wise(fov) => fov.$method($($arg),*),
            Self::NeosCmos(fov) => fov.$method($($arg),*),
            Self::ZtfCcdQuad(fov) => fov.$method($($arg),*),
            Self::GenericCone(fov) => fov.$method($($arg),*),
            Self::GenericRectangle(fov) => fov.$method($($arg),*),
            Self::ZtfField(fov) => fov.$method($($arg),*),
            Self::NeosVisit(fov) => fov.$method($($arg),*),
            Self::OmniDirectional(fov) => fov.$method($($arg),*),
            Self::PtfCcd(fov) => fov.$method($($arg),*),
            Self::PtfField(fov) => fov.$method($($arg),*),
            Self::SpherexCmos(fov) => fov.$method($($arg),*),
            Self::SpherexField(fov) => fov.$method($($arg),*),
        }
    };
}

impl FovLike for FOV {
    fn get_fov(&self, index: usize) -> FOV {
        dispatch_fov!(self, get_fov, index)
    }

    fn observer(&self) -> &State<Equatorial> {
        dispatch_fov!(self, observer)
    }

    fn contains(&self, obs_to_obj: &Vector<Equatorial>) -> (usize, Contains) {
        dispatch_fov!(self, contains, obs_to_obj)
    }

    fn n_patches(&self) -> usize {
        dispatch_fov!(self, n_patches)
    }

    fn pointing(&self) -> KeteResult<Vector<Equatorial>> {
        match self {
            Self::Wise(fov) => fov.pointing(),
            Self::NeosCmos(fov) => fov.pointing(),
            Self::ZtfCcdQuad(fov) => fov.pointing(),
            Self::GenericCone(fov) => fov.pointing(),
            Self::GenericRectangle(fov) => fov.pointing(),
            Self::ZtfField(fov) => fov.pointing(),
            Self::NeosVisit(fov) => <NeosVisit as FovLike>::pointing(fov),
            Self::OmniDirectional(fov) => fov.pointing(),
            Self::PtfCcd(fov) => fov.pointing(),
            Self::PtfField(fov) => fov.pointing(),
            Self::SpherexCmos(fov) => fov.pointing(),
            Self::SpherexField(fov) => fov.pointing(),
        }
    }

    fn corners(&self) -> KeteResult<Vec<Vector<Equatorial>>> {
        match self {
            Self::Wise(fov) => fov.corners(),
            Self::NeosCmos(fov) => fov.corners(),
            Self::ZtfCcdQuad(fov) => fov.corners(),
            Self::GenericCone(fov) => fov.corners(),
            Self::GenericRectangle(fov) => fov.corners(),
            Self::ZtfField(fov) => fov.corners(),
            Self::NeosVisit(fov) => <NeosVisit as FovLike>::corners(fov),
            Self::OmniDirectional(fov) => fov.corners(),
            Self::PtfCcd(fov) => fov.corners(),
            Self::PtfField(fov) => fov.corners(),
            Self::SpherexCmos(fov) => fov.corners(),
            Self::SpherexField(fov) => fov.corners(),
        }
    }
}

/// Assuming the object undergoes two-body motion, check to see if it is within the
/// field of view.
///
/// Both the state and the FOV observer must be Sun-centered (`center_id = 10`).
///
/// # Errors
/// Returns an error if `state.center_id != 10` or if the Kepler solver fails.
pub fn check_two_body<F: FovLike>(
    fov: &F,
    state: &State<Equatorial>,
) -> KeteResult<(usize, Contains, State<Equatorial>)> {
    if state.center_id != 10 {
        return Err(Error::ValueError(
            "check_two_body requires center_id = 10 (Sun).".into(),
        ));
    }
    let obs = fov.observer();

    let final_state = propagate_two_body(state, obs.epoch)?;
    let dist = (final_state.pos - obs.pos).norm();
    let final_state = light_time_correct(&final_state, dist)?;
    let rel_pos = final_state.pos - obs.pos;

    let (idx, contains) = fov.contains(&rel_pos);
    Ok((idx, contains, final_state))
}
