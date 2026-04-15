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
mod spitzer;
mod wise;
mod ztf;

use crate::errors::KeteResult;
use crate::frames::{Equatorial, Vector};
use crate::state::State;

pub use self::fov_like::{FovLike, check_linear, check_statics, check_two_body};
pub use self::generic::{GenericCone, GenericRectangle, OmniDirectional};
pub use self::neos::{NeosCmos, NeosVisit};
pub use self::patches::{Contains, OnSkyRectangle, SkyPatch, SphericalCone, SphericalPolygon};
pub use self::ptf::{PTFFilter, PtfCcd, PtfField};
pub use self::spherex::{SpherexCmos, SpherexField};
pub use self::spitzer::{SpitzerBand, SpitzerFrame};
pub use self::wise::WiseCmos;
pub use self::ztf::{ZtfCcdQuad, ZtfField};

/// Allowed FOV objects, either contiguous or joint.
/// Many of these exist solely to carry additional metadata.
#[derive(Debug, Clone)]
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

    /// Spitzer BCD frame (IRAC or MIPS).
    Spitzer(SpitzerFrame),
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
            Self::Spitzer(fov) => fov.$method($($arg),*),
        }
    };
}

impl FovLike for FOV {
    type ChildFov = Self;

    fn corners(&self) -> KeteResult<Vec<Vector<Equatorial>>> {
        dispatch_fov!(self, corners)
    }

    fn get_child(&self, index: usize) -> Self {
        match self {
            Self::Wise(fov) => Self::Wise(fov.get_child(index)),
            Self::NeosCmos(fov) => Self::NeosCmos(fov.get_child(index)),
            Self::ZtfCcdQuad(fov) => Self::ZtfCcdQuad(fov.get_child(index)),
            Self::GenericCone(fov) => Self::GenericCone(fov.get_child(index)),
            Self::GenericRectangle(fov) => Self::GenericRectangle(fov.get_child(index)),
            Self::ZtfField(fov) => Self::ZtfCcdQuad(fov.get_child(index)),
            Self::NeosVisit(fov) => Self::NeosCmos(fov.get_child(index)),
            Self::OmniDirectional(fov) => Self::OmniDirectional(fov.get_child(index)),
            Self::PtfCcd(fov) => Self::PtfCcd(fov.get_child(index)),
            Self::PtfField(fov) => Self::PtfCcd(fov.get_child(index)),
            Self::SpherexCmos(fov) => Self::SpherexCmos(fov.get_child(index)),
            Self::SpherexField(fov) => Self::SpherexCmos(fov.get_child(index)),
            Self::Spitzer(fov) => Self::Spitzer(fov.get_child(index)),
        }
    }

    fn pointing(&self) -> KeteResult<Vector<Equatorial>> {
        dispatch_fov!(self, pointing)
    }

    fn into_fov(self) -> FOV {
        self
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
}
