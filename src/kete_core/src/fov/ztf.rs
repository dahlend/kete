//! # ZTF Fov definitions.

use super::{closest_inside, Contains, FovLike, OnSkyRectangle, SkyPatch, FOV};
use crate::{frames::Vector, prelude::*};
use serde::{Deserialize, Serialize};

/// ZTF frame data, single quad of a single chip
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ZtfCcdQuad {
    /// State of the observer
    observer: State<Equatorial>,

    /// Patch of sky
    patch: OnSkyRectangle,

    /// Field ID
    pub field: u32,

    /// File Frac Day
    /// String representation of the filename for this frame.
    pub filefracday: u64,

    /// Magnitude limit of this frame
    pub maglimit: f64,

    /// Filter ID
    pub fid: u64,

    /// Filter code used for the frame
    pub filtercode: Box<str>,

    /// Image Type Code
    pub imgtypecode: Box<str>,

    /// Which CCID was the frame taken with
    pub ccdid: u8,

    /// Quadrant ID
    pub qid: u8,
}

impl ZtfCcdQuad {
    /// Create a ZTF field of view
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        corners: [Vector<Equatorial>; 4],
        observer: State<Equatorial>,
        field: u32,
        filefracday: u64,
        ccdid: u8,
        filtercode: Box<str>,
        imgtypecode: Box<str>,
        qid: u8,
        maglimit: f64,
        fid: u64,
    ) -> Self {
        let patch = OnSkyRectangle::from_corners(corners);
        Self {
            patch,
            observer,
            field,
            filefracday,
            ccdid,
            filtercode,
            imgtypecode,
            qid,
            maglimit,
            fid,
        }
    }
}

impl FovLike for ZtfCcdQuad {
    fn get_fov(&self, index: usize) -> FOV {
        if index != 0 {
            panic!("FOV only has a single patch")
        }
        FOV::ZtfCcdQuad(self.clone())
    }

    #[inline]
    fn observer(&self) -> &State<Equatorial> {
        &self.observer
    }

    #[inline]
    fn contains(&self, obs_to_obj: &Vector<Equatorial>) -> (usize, Contains) {
        (0, self.patch.contains(obs_to_obj))
    }

    #[inline]
    fn n_patches(&self) -> usize {
        1
    }

    #[inline]
    fn pointing(&self) -> KeteResult<Vector<Equatorial>> {
        Ok(self.patch.pointing())
    }

    #[inline]
    fn corners(&self) -> KeteResult<Vec<Vector<Equatorial>>> {
        Ok(self.patch.corners().into())
    }
}

/// ZTF frame data, single quad of a single chip
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ZtfField {
    /// Individual CCD quads
    ccd_quads: Vec<ZtfCcdQuad>,

    /// Observer position
    observer: State<Equatorial>,

    /// Field ID
    pub field: u32,

    /// Filter ID
    pub fid: u64,

    /// Filter code used for the frame
    pub filtercode: Box<str>,

    /// Image Type Code
    pub imgtypecode: Box<str>,
}

impl ZtfField {
    /// Construct a new ZtfField from a list of ccd quads.
    /// These ccd quads must be from the same field and having matching value as
    /// appropriate.
    pub fn new(ccd_quads: Vec<ZtfCcdQuad>) -> KeteResult<Self> {
        if ccd_quads.is_empty() {
            Err(Error::ValueError(
                "Ztf Field must contains ZtfCcdQuads".into(),
            ))?;
        }

        let first = ccd_quads.first().unwrap();

        let observer = first.observer().clone();
        let field = first.field;
        let fid = first.fid;
        let filtercode = first.filtercode.clone();
        let imgtypecode = first.imgtypecode.clone();

        for ccd in ccd_quads.iter() {
            if ccd.field != field
                || ccd.fid != fid
                || ccd.filtercode != filtercode
                || ccd.imgtypecode != imgtypecode
                || ccd.observer().jd != observer.jd
            {
                Err(Error::ValueError(
                    "All ZtfCcdQuads must have matching values except CCD ID etc.".into(),
                ))?;
            }
        }
        Ok(Self {
            ccd_quads,
            observer,
            field,
            fid,
            filtercode,
            imgtypecode,
        })
    }
}

impl FovLike for ZtfField {
    fn get_fov(&self, index: usize) -> FOV {
        FOV::ZtfCcdQuad(self.ccd_quads[index].clone())
    }

    fn observer(&self) -> &State<Equatorial> {
        &self.observer
    }

    fn contains(&self, obs_to_obj: &Vector<Equatorial>) -> (usize, Contains) {
        closest_inside(
            &self
                .ccd_quads
                .iter()
                .map(|x| x.contains(obs_to_obj).1)
                .collect::<Vec<_>>(),
        )
    }

    fn n_patches(&self) -> usize {
        self.ccd_quads.len()
    }

    #[inline]
    fn pointing(&self) -> KeteResult<Vector<Equatorial>> {
        if self.ccd_quads.is_empty() {
            Err(Error::ValueError("ZtfField has no ccd quads".into()))
        } else {
            // return the average pointing of all ccd quads
            Ok(self
                .ccd_quads
                .iter()
                .fold(Vector::new([0.0; 3]), |acc, x| acc + x.pointing().unwrap()))
        }
    }

    #[inline]
    fn corners(&self) -> KeteResult<Vec<Vector<Equatorial>>> {
        if self.ccd_quads.is_empty() {
            Err(Error::ValueError("ZtfField has no ccd quads".into()))
        } else {
            // return all the corners of all ccd quads
            Ok(self
                .ccd_quads
                .iter()
                .flat_map(|x| x.corners().unwrap())
                .collect())
        }
    }
}
