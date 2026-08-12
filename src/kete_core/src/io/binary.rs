//! Binary codec for kete file format.
//!
//! Defines [`KeteWrite`] / [`KeteRead`] traits and implementations for all types
//! that participate in the kete binary file format. Also provides file-level
//! [`write_single_kete_file`], [`write_vec_kete_file`], and [`read_kete_file`]
//! functions that handle the header (magic bytes, version, content type, entry count).
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

use crate::desigs::Desig;
use crate::elements::EquinoctialElements;
use crate::errors::{Error, KeteResult};
use crate::forces::{
    DustNonGrav, FarnocchiaNonGrav, JplCometNonGrav, NonGravKind, ParameterMask, ParameterizedForce,
};
use crate::fov::{
    FOV, GenericCone, GenericRectangle, NeosCmos, NeosVisit, OmniDirectional, OnSkyRectangle,
    PTFFilter, PtfCcd, PtfField, SpherexCmos, SpherexField, SphericalCone, SpitzerBand,
    SpitzerFrame, WiseCmos, ZtfCcdQuad, ZtfField,
};
use crate::frames::{Equatorial, Vector};
use crate::state::{DiffuseState, ProbeSet, SimultaneousStates, State, UncertainState};
use crate::time::{TDB, Time};
use nalgebra::{DMatrix, DVector, Vector3};
use std::io::{self, Cursor, Read, Write};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAGIC: &[u8; 4] = b"KETE";
const VERSION: u16 = 1;
const CONTENT_TYPE_SINGLE: u8 = 0;
const CONTENT_TYPE_VEC: u8 = 1;
const CONTENT_TYPE_UNCERTAIN: u8 = 2;
const CONTENT_TYPE_UNCERTAIN_VEC: u8 = 3;
const CONTENT_TYPE_DIFFUSE: u8 = 4;
const CONTENT_TYPE_DIFFUSE_VEC: u8 = 5;

/// The payload read from a kete binary file.
///
/// The content type in the file header determines which variant is returned.
/// New content types are added rather than changing the version, so a reader
/// keeps accepting every file an older writer produced.
#[derive(Debug, Clone)]
pub enum KeteFileType {
    /// A single [`SimultaneousStates`] (content type 0).
    Single(Box<SimultaneousStates>),
    /// A collection of [`SimultaneousStates`] (content type 1).
    Vec(Vec<SimultaneousStates>),
    /// A single [`UncertainState`] (content type 2).
    Uncertain(Box<UncertainState>),
    /// A collection of [`UncertainState`] (content type 3).
    UncertainVec(Vec<UncertainState>),
    /// A single [`DiffuseState`] (content type 4).
    Diffuse(Box<DiffuseState>),
    /// A collection of [`DiffuseState`] (content type 5).
    DiffuseVec(Vec<DiffuseState>),
}

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

/// Serialize a value into a byte stream (little-endian).
pub trait KeteWrite {
    /// Write `self` into the given writer.
    ///
    /// # Errors
    /// Returns an error if the write operation fails.
    fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()>;
}

/// Deserialize a value from a byte stream (little-endian).
pub trait KeteRead: Sized {
    /// Read a value from the given reader.
    ///
    /// # Errors
    /// Returns an error if the read operation fails or the data is invalid.
    fn read_from<R: Read>(r: &mut R) -> KeteResult<Self>;
}

// ---------------------------------------------------------------------------
// Primitive implementations
// ---------------------------------------------------------------------------

macro_rules! impl_primitive {
    ($($ty:ty),+) => { $(
        impl KeteWrite for $ty {
            #[inline]
            fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
                w.write_all(&self.to_le_bytes())
            }
        }
        impl KeteRead for $ty {
            #[inline]
            fn read_from<R: Read>(r: &mut R) -> KeteResult<Self> {
                let mut buf = [0_u8; std::mem::size_of::<$ty>()];
                r.read_exact(&mut buf)?;
                Ok(Self::from_le_bytes(buf))
            }
        }
    )+ };
}

impl_primitive!(u8, u16, u32, u64, i32, f32, f64);

// ---------------------------------------------------------------------------
// char, bool, Option<T>, str, String, Box<str>, Vec<T>
// ---------------------------------------------------------------------------

impl KeteWrite for char {
    fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        (*self as u32).write_to(w)
    }
}

impl KeteRead for char {
    fn read_from<R: Read>(r: &mut R) -> KeteResult<Self> {
        let v = u32::read_from(r)?;
        Self::from_u32(v).ok_or_else(|| Error::IOError(format!("Invalid char value: {v}")))
    }
}

impl KeteWrite for bool {
    fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        u8::from(*self).write_to(w)
    }
}

impl KeteRead for bool {
    fn read_from<R: Read>(r: &mut R) -> KeteResult<Self> {
        match u8::read_from(r)? {
            0 => Ok(false),
            1 => Ok(true),
            t => Err(Error::IOError(format!("Invalid bool value: {t}"))),
        }
    }
}

impl<T: KeteWrite> KeteWrite for Option<T> {
    fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        match self {
            Some(v) => {
                1_u8.write_to(w)?;
                v.write_to(w)
            }
            None => 0_u8.write_to(w),
        }
    }
}

impl<T: KeteRead> KeteRead for Option<T> {
    fn read_from<R: Read>(r: &mut R) -> KeteResult<Self> {
        match u8::read_from(r)? {
            0 => Ok(None),
            1 => Ok(Some(T::read_from(r)?)),
            t => Err(Error::IOError(format!("Invalid Option tag: {t}"))),
        }
    }
}

impl KeteWrite for str {
    fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        (self.len() as u16).write_to(w)?;
        w.write_all(self.as_bytes())
    }
}

impl KeteRead for String {
    fn read_from<R: Read>(r: &mut R) -> KeteResult<Self> {
        let len = u16::read_from(r)? as usize;
        let mut buf = vec![0_u8; len];
        r.read_exact(&mut buf)?;
        Self::from_utf8(buf).map_err(|e| Error::IOError(e.to_string()))
    }
}

impl KeteRead for Box<str> {
    fn read_from<R: Read>(r: &mut R) -> KeteResult<Self> {
        Ok(String::read_from(r)?.into_boxed_str())
    }
}

impl<T: KeteWrite> KeteWrite for Vec<T> {
    fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        (self.len() as u32).write_to(w)?;
        for item in self {
            item.write_to(w)?;
        }
        Ok(())
    }
}

impl<T: KeteRead> KeteRead for Vec<T> {
    fn read_from<R: Read>(r: &mut R) -> KeteResult<Self> {
        let count = u32::read_from(r)? as usize;
        let mut vec = Self::with_capacity(count);
        for _ in 0..count {
            vec.push(T::read_from(r)?);
        }
        Ok(vec)
    }
}

// ---------------------------------------------------------------------------
// DMatrix<f64>, Vector3<f64>
// ---------------------------------------------------------------------------

// These are written by plain functions rather than through `KeteWrite`.
// Implementing the traits for nalgebra's own types would make every matrix in
// the ecosystem serializable in kete's format, which is a far broader promise
// than the few fields below need, and one this format would then owe forever.

/// Write a dense matrix as its shape and then its entries, column major.
fn write_matrix<W: Write>(matrix: &DMatrix<f64>, w: &mut W) -> io::Result<()> {
    (matrix.nrows() as u32).write_to(w)?;
    (matrix.ncols() as u32).write_to(w)?;
    for v in matrix.iter() {
        v.write_to(w)?;
    }
    Ok(())
}

/// Read a dense matrix written by [`write_matrix`].
fn read_matrix<R: Read>(r: &mut R) -> KeteResult<DMatrix<f64>> {
    let rows = u32::read_from(r)? as usize;
    let cols = u32::read_from(r)? as usize;
    let count = rows
        .checked_mul(cols)
        .ok_or_else(|| Error::IOError(format!("Matrix dimensions {rows}x{cols} overflow")))?;
    let mut data = Vec::with_capacity(count);
    for _ in 0..count {
        data.push(f64::read_from(r)?);
    }
    Ok(DMatrix::from_vec(rows, cols, data))
}

/// Write three components of a cartesian vector.
fn write_xyz<W: Write>(vector: &Vector3<f64>, w: &mut W) -> io::Result<()> {
    for v in vector.iter() {
        v.write_to(w)?;
    }
    Ok(())
}

/// Read a vector written by [`write_xyz`].
fn read_xyz<R: Read>(r: &mut R) -> KeteResult<Vector3<f64>> {
    let x = f64::read_from(r)?;
    let y = f64::read_from(r)?;
    let z = f64::read_from(r)?;
    Ok(Vector3::new(x, y, z))
}

// ---------------------------------------------------------------------------
// Vector<Equatorial>
// ---------------------------------------------------------------------------

impl KeteWrite for Vector<Equatorial> {
    fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        let raw: [f64; 3] = (*self).into();
        for val in &raw {
            val.write_to(w)?;
        }
        Ok(())
    }
}

impl KeteRead for Vector<Equatorial> {
    fn read_from<R: Read>(r: &mut R) -> KeteResult<Self> {
        Ok(Self::new([
            f64::read_from(r)?,
            f64::read_from(r)?,
            f64::read_from(r)?,
        ]))
    }
}

// ---------------------------------------------------------------------------
// Time<TDB>
// ---------------------------------------------------------------------------

impl KeteWrite for Time<TDB> {
    fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        self.jd.write_to(w)
    }
}

impl KeteRead for Time<TDB> {
    fn read_from<R: Read>(r: &mut R) -> KeteResult<Self> {
        Ok(Self::new(f64::read_from(r)?))
    }
}

// ---------------------------------------------------------------------------
// Desig
// ---------------------------------------------------------------------------

impl KeteWrite for Desig {
    fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        let mut payload = Vec::new();
        let tag: u8 = match self {
            Self::Empty => 0,
            Self::Perm(v) => {
                v.write_to(&mut payload)?;
                1
            }
            Self::Prov(s) => {
                s.as_str().write_to(&mut payload)?;
                2
            }
            Self::CometPerm(c, n, opt_c) => {
                c.write_to(&mut payload)?;
                n.write_to(&mut payload)?;
                opt_c.write_to(&mut payload)?;
                3
            }
            Self::CometProv(opt_c, s, opt_c2) => {
                opt_c.write_to(&mut payload)?;
                s.as_str().write_to(&mut payload)?;
                opt_c2.write_to(&mut payload)?;
                4
            }
            Self::PlanetSat(a, b) => {
                a.write_to(&mut payload)?;
                b.write_to(&mut payload)?;
                5
            }
            Self::Name(s) => {
                s.as_str().write_to(&mut payload)?;
                6
            }
            Self::Naif(v) => {
                v.write_to(&mut payload)?;
                7
            }
            Self::ObservatoryCode(s) => {
                s.as_str().write_to(&mut payload)?;
                8
            }
        };
        tag.write_to(w)?;
        (payload.len() as u8).write_to(w)?;
        w.write_all(&payload)
    }
}

impl KeteRead for Desig {
    fn read_from<R: Read>(r: &mut R) -> KeteResult<Self> {
        let tag = u8::read_from(r)?;
        let payload_len = u8::read_from(r)? as usize;
        let mut payload = vec![0_u8; payload_len];
        r.read_exact(&mut payload)?;
        let mut cursor = Cursor::new(&payload);
        match tag {
            0 => Ok(Self::Empty),
            1 => Ok(Self::Perm(u32::read_from(&mut cursor)?)),
            2 => Ok(Self::Prov(String::read_from(&mut cursor)?)),
            3 => Ok(Self::CometPerm(
                char::read_from(&mut cursor)?,
                u32::read_from(&mut cursor)?,
                Option::<char>::read_from(&mut cursor)?,
            )),
            4 => Ok(Self::CometProv(
                Option::<char>::read_from(&mut cursor)?,
                String::read_from(&mut cursor)?,
                Option::<char>::read_from(&mut cursor)?,
            )),
            5 => Ok(Self::PlanetSat(
                i32::read_from(&mut cursor)?,
                u32::read_from(&mut cursor)?,
            )),
            6 => Ok(Self::Name(String::read_from(&mut cursor)?)),
            7 => Ok(Self::Naif(i32::read_from(&mut cursor)?)),
            8 => Ok(Self::ObservatoryCode(String::read_from(&mut cursor)?)),
            t => Err(Error::IOError(format!("Unknown Desig tag: {t}"))),
        }
    }
}

// ---------------------------------------------------------------------------
// State<Equatorial>
// ---------------------------------------------------------------------------

impl KeteWrite for State<Equatorial> {
    fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        self.desig.write_to(w)?;
        self.epoch.write_to(w)?;
        self.pos.write_to(w)?;
        self.vel.write_to(w)?;
        self.center_id().write_to(w)
    }
}

impl KeteRead for State<Equatorial> {
    fn read_from<R: Read>(r: &mut R) -> KeteResult<Self> {
        let desig = Desig::read_from(r)?;
        let epoch = Time::read_from(r)?;
        let pos = Vector::read_from(r)?;
        let vel = Vector::read_from(r)?;
        let center_id = i32::read_from(r)?;
        Ok(Self::new(desig, epoch, pos, vel, center_id))
    }
}

// ---------------------------------------------------------------------------
// SphericalCone, OnSkyRectangle, PTFFilter
// ---------------------------------------------------------------------------

impl KeteWrite for SphericalCone {
    fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        self.pointing.write_to(w)?;
        self.angle.write_to(w)
    }
}

impl KeteRead for SphericalCone {
    fn read_from<R: Read>(r: &mut R) -> KeteResult<Self> {
        Ok(Self {
            pointing: Vector::read_from(r)?,
            angle: f64::read_from(r)?,
        })
    }
}

impl KeteWrite for OnSkyRectangle {
    fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        for normal in &self.edge_normals {
            normal.write_to(w)?;
        }
        Ok(())
    }
}

impl KeteRead for OnSkyRectangle {
    fn read_from<R: Read>(r: &mut R) -> KeteResult<Self> {
        let normals = [
            Vector::read_from(r)?,
            Vector::read_from(r)?,
            Vector::read_from(r)?,
            Vector::read_from(r)?,
        ];
        Ok(Self::from_normals(normals))
    }
}

impl KeteWrite for PTFFilter {
    fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        let tag: u8 = match self {
            Self::G => 0,
            Self::R => 1,
            Self::HA656 => 2,
            Self::HA663 => 3,
        };
        tag.write_to(w)
    }
}

impl KeteRead for PTFFilter {
    fn read_from<R: Read>(r: &mut R) -> KeteResult<Self> {
        match u8::read_from(r)? {
            0 => Ok(Self::G),
            1 => Ok(Self::R),
            2 => Ok(Self::HA656),
            3 => Ok(Self::HA663),
            t => Err(Error::IOError(format!("Invalid PTFFilter tag: {t}"))),
        }
    }
}

// ---------------------------------------------------------------------------
// FOV variant structs
// ---------------------------------------------------------------------------

impl KeteWrite for OmniDirectional {
    fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        self.observer.write_to(w)
    }
}

impl KeteRead for OmniDirectional {
    fn read_from<R: Read>(r: &mut R) -> KeteResult<Self> {
        Ok(Self {
            observer: State::read_from(r)?,
        })
    }
}

impl KeteWrite for GenericCone {
    fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        self.observer.write_to(w)?;
        self.patch.write_to(w)
    }
}

impl KeteRead for GenericCone {
    fn read_from<R: Read>(r: &mut R) -> KeteResult<Self> {
        Ok(Self {
            observer: State::read_from(r)?,
            patch: SphericalCone::read_from(r)?,
        })
    }
}

impl KeteWrite for GenericRectangle {
    fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        self.observer.write_to(w)?;
        self.patch.write_to(w)?;
        self.rotation.write_to(w)
    }
}

impl KeteRead for GenericRectangle {
    fn read_from<R: Read>(r: &mut R) -> KeteResult<Self> {
        Ok(Self {
            observer: State::read_from(r)?,
            patch: OnSkyRectangle::read_from(r)?,
            rotation: f64::read_from(r)?,
        })
    }
}

impl KeteWrite for WiseCmos {
    fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        self.observer.write_to(w)?;
        self.patch.write_to(w)?;
        self.frame_num.write_to(w)?;
        self.scan_id.as_ref().write_to(w)
    }
}

impl KeteRead for WiseCmos {
    fn read_from<R: Read>(r: &mut R) -> KeteResult<Self> {
        Ok(Self {
            observer: State::read_from(r)?,
            patch: OnSkyRectangle::read_from(r)?,
            frame_num: u64::read_from(r)?,
            scan_id: Box::<str>::read_from(r)?,
        })
    }
}

impl KeteWrite for NeosCmos {
    fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        self.observer.write_to(w)?;
        self.patch.write_to(w)?;
        self.rotation.write_to(w)?;
        self.side_id.write_to(w)?;
        self.stack_id.write_to(w)?;
        self.quad_id.write_to(w)?;
        self.loop_id.write_to(w)?;
        self.subloop_id.write_to(w)?;
        self.exposure_id.write_to(w)?;
        self.band.write_to(w)?;
        self.cmos_id.write_to(w)
    }
}

impl KeteRead for NeosCmos {
    fn read_from<R: Read>(r: &mut R) -> KeteResult<Self> {
        Ok(Self {
            observer: State::read_from(r)?,
            patch: OnSkyRectangle::read_from(r)?,
            rotation: f64::read_from(r)?,
            side_id: u16::read_from(r)?,
            stack_id: u8::read_from(r)?,
            quad_id: u8::read_from(r)?,
            loop_id: u8::read_from(r)?,
            subloop_id: u8::read_from(r)?,
            exposure_id: u8::read_from(r)?,
            band: u8::read_from(r)?,
            cmos_id: u8::read_from(r)?,
        })
    }
}

impl KeteWrite for NeosVisit {
    fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        for chip in self.chips.as_ref() {
            chip.write_to(w)?;
        }
        self.observer.write_to(w)?;
        self.rotation.write_to(w)?;
        self.side_id.write_to(w)?;
        self.stack_id.write_to(w)?;
        self.quad_id.write_to(w)?;
        self.loop_id.write_to(w)?;
        self.subloop_id.write_to(w)?;
        self.exposure_id.write_to(w)?;
        self.band.write_to(w)
    }
}

impl KeteRead for NeosVisit {
    fn read_from<R: Read>(r: &mut R) -> KeteResult<Self> {
        let chips = Box::new([
            NeosCmos::read_from(r)?,
            NeosCmos::read_from(r)?,
            NeosCmos::read_from(r)?,
            NeosCmos::read_from(r)?,
        ]);
        Ok(Self {
            chips,
            observer: State::read_from(r)?,
            rotation: f64::read_from(r)?,
            side_id: u16::read_from(r)?,
            stack_id: u8::read_from(r)?,
            quad_id: u8::read_from(r)?,
            loop_id: u8::read_from(r)?,
            subloop_id: u8::read_from(r)?,
            exposure_id: u8::read_from(r)?,
            band: u8::read_from(r)?,
        })
    }
}

impl KeteWrite for ZtfCcdQuad {
    fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        self.observer.write_to(w)?;
        self.patch.write_to(w)?;
        self.field.write_to(w)?;
        self.filefracday.write_to(w)?;
        self.maglimit.write_to(w)?;
        self.fid.write_to(w)?;
        self.filtercode.as_ref().write_to(w)?;
        self.imgtypecode.as_ref().write_to(w)?;
        self.ccdid.write_to(w)?;
        self.qid.write_to(w)
    }
}

impl KeteRead for ZtfCcdQuad {
    fn read_from<R: Read>(r: &mut R) -> KeteResult<Self> {
        Ok(Self {
            observer: State::read_from(r)?,
            patch: OnSkyRectangle::read_from(r)?,
            field: u32::read_from(r)?,
            filefracday: u64::read_from(r)?,
            maglimit: f64::read_from(r)?,
            fid: u64::read_from(r)?,
            filtercode: Box::<str>::read_from(r)?,
            imgtypecode: Box::<str>::read_from(r)?,
            ccdid: u8::read_from(r)?,
            qid: u8::read_from(r)?,
        })
    }
}

impl KeteWrite for ZtfField {
    fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        self.ccd_quads.write_to(w)?;
        self.observer.write_to(w)?;
        self.field.write_to(w)?;
        self.fid.write_to(w)?;
        self.filtercode.as_ref().write_to(w)?;
        self.imgtypecode.as_ref().write_to(w)
    }
}

impl KeteRead for ZtfField {
    fn read_from<R: Read>(r: &mut R) -> KeteResult<Self> {
        Ok(Self {
            ccd_quads: Vec::read_from(r)?,
            observer: State::read_from(r)?,
            field: u32::read_from(r)?,
            fid: u64::read_from(r)?,
            filtercode: Box::<str>::read_from(r)?,
            imgtypecode: Box::<str>::read_from(r)?,
        })
    }
}

impl KeteWrite for PtfCcd {
    fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        self.observer.write_to(w)?;
        self.patch.write_to(w)?;
        self.field.write_to(w)?;
        self.ccdid.write_to(w)?;
        self.filter.write_to(w)?;
        self.filename.as_ref().write_to(w)?;
        self.info_bits.write_to(w)?;
        self.seeing.write_to(w)
    }
}

impl KeteRead for PtfCcd {
    fn read_from<R: Read>(r: &mut R) -> KeteResult<Self> {
        Ok(Self {
            observer: State::read_from(r)?,
            patch: OnSkyRectangle::read_from(r)?,
            field: u32::read_from(r)?,
            ccdid: u8::read_from(r)?,
            filter: PTFFilter::read_from(r)?,
            filename: Box::<str>::read_from(r)?,
            info_bits: u32::read_from(r)?,
            seeing: f32::read_from(r)?,
        })
    }
}

impl KeteWrite for PtfField {
    fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        self.ccds.write_to(w)?;
        self.observer.write_to(w)?;
        self.field.write_to(w)?;
        self.filter.write_to(w)
    }
}

impl KeteRead for PtfField {
    fn read_from<R: Read>(r: &mut R) -> KeteResult<Self> {
        Ok(Self {
            ccds: Vec::read_from(r)?,
            observer: State::read_from(r)?,
            field: u32::read_from(r)?,
            filter: PTFFilter::read_from(r)?,
        })
    }
}

impl KeteWrite for SpherexCmos {
    fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        self.observer.write_to(w)?;
        self.patch.write_to(w)?;
        self.uri.as_ref().write_to(w)?;
        self.plane_id.as_ref().write_to(w)
    }
}

impl KeteRead for SpherexCmos {
    fn read_from<R: Read>(r: &mut R) -> KeteResult<Self> {
        Ok(Self {
            observer: State::read_from(r)?,
            patch: OnSkyRectangle::read_from(r)?,
            uri: Box::<str>::read_from(r)?,
            plane_id: Box::<str>::read_from(r)?,
        })
    }
}

impl KeteWrite for SpherexField {
    fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        self.cmos_frames.write_to(w)?;
        self.observer.write_to(w)?;
        self.obsid.as_ref().write_to(w)?;
        self.observationid.as_ref().write_to(w)
    }
}

impl KeteRead for SpherexField {
    fn read_from<R: Read>(r: &mut R) -> KeteResult<Self> {
        Ok(Self {
            cmos_frames: Vec::read_from(r)?,
            observer: State::read_from(r)?,
            obsid: Box::<str>::read_from(r)?,
            observationid: Box::<str>::read_from(r)?,
        })
    }
}

impl KeteWrite for SpitzerBand {
    fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        let tag: u8 = match self {
            Self::Irac1 => 0,
            Self::Irac2 => 1,
            Self::Irac3 => 2,
            Self::Irac4 => 3,
            Self::Mips24 => 4,
            Self::Mips70 => 5,
            Self::Mips160 => 6,
            Self::IrsPeakUpBlue => 7,
            Self::IrsPeakUpRed => 8,
        };
        tag.write_to(w)
    }
}

impl KeteRead for SpitzerBand {
    fn read_from<R: Read>(r: &mut R) -> KeteResult<Self> {
        match u8::read_from(r)? {
            0 => Ok(Self::Irac1),
            1 => Ok(Self::Irac2),
            2 => Ok(Self::Irac3),
            3 => Ok(Self::Irac4),
            4 => Ok(Self::Mips24),
            5 => Ok(Self::Mips70),
            6 => Ok(Self::Mips160),
            7 => Ok(Self::IrsPeakUpBlue),
            8 => Ok(Self::IrsPeakUpRed),
            t => Err(Error::IOError(format!("Invalid SpitzerBand tag: {t}"))),
        }
    }
}

impl KeteWrite for SpitzerFrame {
    fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        self.observer.write_to(w)?;
        self.patch.write_to(w)?;
        self.obs_id.as_ref().write_to(w)?;
        self.band.write_to(w)?;
        self.artifact_uri.as_ref().write_to(w)?;
        self.duration.write_to(w)
    }
}

impl KeteRead for SpitzerFrame {
    fn read_from<R: Read>(r: &mut R) -> KeteResult<Self> {
        Ok(Self {
            observer: State::read_from(r)?,
            patch: OnSkyRectangle::read_from(r)?,
            obs_id: Box::<str>::read_from(r)?,
            band: SpitzerBand::read_from(r)?,
            artifact_uri: Box::<str>::read_from(r)?,
            duration: f64::read_from(r)?,
        })
    }
}

// ---------------------------------------------------------------------------
// FOV enum
// ---------------------------------------------------------------------------

/// Write a value to a `Vec<u8>` buffer.
fn write_to_vec<T: KeteWrite>(val: &T) -> io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    val.write_to(&mut buf)?;
    Ok(buf)
}

impl KeteWrite for FOV {
    fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        let (tag, payload) = match self {
            Self::OmniDirectional(v) => (0_u8, write_to_vec(v)?),
            Self::GenericCone(v) => (1_u8, write_to_vec(v)?),
            Self::GenericRectangle(v) => (2_u8, write_to_vec(v)?),
            Self::Wise(v) => (3_u8, write_to_vec(v)?),
            Self::NeosCmos(v) => (4_u8, write_to_vec(v)?),
            Self::NeosVisit(v) => (5_u8, write_to_vec(v)?),
            Self::ZtfCcdQuad(v) => (6_u8, write_to_vec(v)?),
            Self::ZtfField(v) => (7_u8, write_to_vec(v)?),
            Self::PtfCcd(v) => (8_u8, write_to_vec(v)?),
            Self::PtfField(v) => (9_u8, write_to_vec(v)?),
            Self::SpherexCmos(v) => (10_u8, write_to_vec(v)?),
            Self::SpherexField(v) => (11_u8, write_to_vec(v)?),
            Self::Spitzer(v) => (12_u8, write_to_vec(v)?),
        };
        tag.write_to(w)?;
        (payload.len() as u32).write_to(w)?;
        w.write_all(&payload)
    }
}

/// Read an FOV from the stream. Returns `None` if the tag is unknown
/// (forward compatibility -- unknown variants are skipped).
///
/// # Errors
/// Returns an error if the stream cannot be read or contains invalid data.
pub fn read_fov<R: Read>(r: &mut R) -> KeteResult<Option<FOV>> {
    let tag = u8::read_from(r)?;
    let payload_len = u32::read_from(r)? as usize;
    let mut payload = vec![0_u8; payload_len];
    r.read_exact(&mut payload)?;

    let mut cursor = Cursor::new(&payload);
    let fov = match tag {
        0 => Some(FOV::OmniDirectional(OmniDirectional::read_from(
            &mut cursor,
        )?)),
        1 => Some(FOV::GenericCone(GenericCone::read_from(&mut cursor)?)),
        2 => Some(FOV::GenericRectangle(GenericRectangle::read_from(
            &mut cursor,
        )?)),
        3 => Some(FOV::Wise(WiseCmos::read_from(&mut cursor)?)),
        4 => Some(FOV::NeosCmos(NeosCmos::read_from(&mut cursor)?)),
        5 => Some(FOV::NeosVisit(NeosVisit::read_from(&mut cursor)?)),
        6 => Some(FOV::ZtfCcdQuad(ZtfCcdQuad::read_from(&mut cursor)?)),
        7 => Some(FOV::ZtfField(ZtfField::read_from(&mut cursor)?)),
        8 => Some(FOV::PtfCcd(PtfCcd::read_from(&mut cursor)?)),
        9 => Some(FOV::PtfField(PtfField::read_from(&mut cursor)?)),
        10 => Some(FOV::SpherexCmos(SpherexCmos::read_from(&mut cursor)?)),
        11 => Some(FOV::SpherexField(SpherexField::read_from(&mut cursor)?)),
        12 => Some(FOV::Spitzer(SpitzerFrame::read_from(&mut cursor)?)),
        _ => None,
    };
    Ok(fov)
}

impl KeteRead for FOV {
    fn read_from<R: Read>(r: &mut R) -> KeteResult<Self> {
        read_fov(r)?.ok_or_else(|| Error::IOError("Unknown FOV variant tag".into()))
    }
}

// ---------------------------------------------------------------------------
// SimultaneousStates
// ---------------------------------------------------------------------------

impl KeteWrite for SimultaneousStates {
    fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        let mut payload = Vec::new();
        self.epoch().write_to(&mut payload)?;
        self.center_id().write_to(&mut payload)?;
        self.fov.write_to(&mut payload)?;
        self.states.write_to(&mut payload)?;
        (payload.len() as u32).write_to(w)?;
        w.write_all(&payload)
    }
}

impl KeteRead for SimultaneousStates {
    fn read_from<R: Read>(r: &mut R) -> KeteResult<Self> {
        let entry_len = u32::read_from(r)? as usize;
        let mut payload = vec![0_u8; entry_len];
        r.read_exact(&mut payload)?;
        let mut cursor = Cursor::new(&payload);

        // Read header fields first (format compatibility: these bytes must
        // stay in this position). Then validate them against what new_exact
        // derives from the state data, catching corrupt files.
        let epoch_file = Time::read_from(&mut cursor)?;
        let center_id_file = i32::read_from(&mut cursor)?;
        let fov = Option::<FOV>::read_from(&mut cursor)?;
        let states = Vec::read_from(&mut cursor)?;
        let result = Self::new_exact(states, fov)?;
        if result.epoch() != epoch_file || result.center_id() != center_id_file {
            return Err(Error::IOError(
                "SimultaneousStates header fields do not match state data".into(),
            ));
        }
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// ProbeSet
// ---------------------------------------------------------------------------

/// Write the probes a component was carrying.
fn write_probes<W: Write>(probes: &ProbeSet, w: &mut W) -> io::Result<()> {
    // One count for all four arrays, which the reader checks against, since they
    // are one per probe pair by construction.
    (probes.directions.len() as u32).write_to(w)?;
    for direction in &probes.directions {
        (direction.len() as u32).write_to(w)?;
        for v in direction.iter() {
            v.write_to(w)?;
        }
    }
    probes.displacements.write_to(w)?;

    // A pair is either both probes alive, written as four vectors, or dead,
    // which is the tag alone. A dead pair reports infinite nonlinearity for the
    // rest of the run, so the distinction has to survive the trip.
    for pair in &probes.states {
        match pair {
            Some([(plus_pos, plus_vel), (minus_pos, minus_vel)]) => {
                1_u8.write_to(w)?;
                write_xyz(plus_pos, w)?;
                write_xyz(plus_vel, w)?;
                write_xyz(minus_pos, w)?;
                write_xyz(minus_vel, w)?;
            }
            None => 0_u8.write_to(w)?,
        }
    }

    for [plus, minus] in &probes.params {
        plus.write_to(w)?;
        minus.write_to(w)?;
    }

    write_matrix(&probes.stm, w)?;
    probes.epoch.write_to(w)
}

/// Read the probes written by [`write_probes`].
fn read_probes<R: Read>(r: &mut R) -> KeteResult<ProbeSet> {
    let pairs = u32::read_from(r)? as usize;
    let mut directions = Vec::with_capacity(pairs);
    for _ in 0..pairs {
        let len = u32::read_from(r)? as usize;
        let mut entries = Vec::with_capacity(len);
        for _ in 0..len {
            entries.push(f64::read_from(r)?);
        }
        directions.push(DVector::from_vec(entries));
    }

    let displacements = Vec::<f64>::read_from(r)?;
    if displacements.len() != pairs {
        return Err(Error::IOError(
            "ProbeSet displacements disagree with the number of probe pairs".into(),
        ));
    }

    let mut states = Vec::with_capacity(pairs);
    for _ in 0..pairs {
        states.push(match u8::read_from(r)? {
            0 => None,
            1 => {
                let plus = (read_xyz(r)?, read_xyz(r)?);
                let minus = (read_xyz(r)?, read_xyz(r)?);
                Some([plus, minus])
            }
            t => return Err(Error::IOError(format!("Invalid probe pair tag: {t}"))),
        });
    }

    let mut params = Vec::with_capacity(pairs);
    for _ in 0..pairs {
        let plus = Vec::<f64>::read_from(r)?;
        let minus = Vec::<f64>::read_from(r)?;
        params.push([plus, minus]);
    }

    Ok(ProbeSet {
        directions,
        displacements,
        states,
        params,
        stm: read_matrix(r)?,
        epoch: Time::read_from(r)?,
    })
}

// ---------------------------------------------------------------------------
// NonGravKind
// ---------------------------------------------------------------------------

const NONGRAV_DUST: u8 = 0;
const NONGRAV_JPL_COMET: u8 = 1;
const NONGRAV_FARNOCCHIA: u8 = 2;

/// Write a non-gravitational model and which of its parameters are free.
fn write_non_grav<W: Write>(mask: &ParameterMask<NonGravKind>, w: &mut W) -> io::Result<()> {
    match &mask.inner {
        NonGravKind::Dust(_) => NONGRAV_DUST.write_to(w)?,
        NonGravKind::JplComet(model) => {
            NONGRAV_JPL_COMET.write_to(w)?;
            model.alpha.write_to(w)?;
            model.r_0.write_to(w)?;
            model.m.write_to(w)?;
            model.n.write_to(w)?;
            model.k.write_to(w)?;
            model.dt.write_to(w)?;
        }
        NonGravKind::Farnocchia(model) => {
            NONGRAV_FARNOCCHIA.write_to(w)?;
            model.albedo.write_to(w)?;
            model.absorptivity.write_to(w)?;
            model.flattening.write_to(w)?;
            model.spin_pole.write_to(w)?;
        }
    }
    mask.mask.write_to(w)
}

/// Read a model written by [`write_non_grav`].
fn read_non_grav<R: Read>(r: &mut R) -> KeteResult<ParameterMask<NonGravKind>> {
    let inner = match u8::read_from(r)? {
        NONGRAV_DUST => NonGravKind::Dust(DustNonGrav),
        NONGRAV_JPL_COMET => NonGravKind::JplComet(JplCometNonGrav::new(
            f64::read_from(r)?,
            f64::read_from(r)?,
            f64::read_from(r)?,
            f64::read_from(r)?,
            f64::read_from(r)?,
            f64::read_from(r)?,
        )),
        NONGRAV_FARNOCCHIA => NonGravKind::Farnocchia(FarnocchiaNonGrav::new(
            f64::read_from(r)?,
            f64::read_from(r)?,
            f64::read_from(r)?,
            Vector::read_from(r)?,
        )?),
        t => return Err(Error::IOError(format!("Invalid non-grav model tag: {t}"))),
    };
    let mask = Vec::<Option<f64>>::read_from(r)?;
    Ok(ParameterMask { inner, mask })
}

// ---------------------------------------------------------------------------
// UncertainState
// ---------------------------------------------------------------------------

// Matrices and vectors are written by these helpers rather than through
// `KeteWrite`. Implementing the traits for nalgebra's types would make every
// matrix in the ecosystem serializable in kete's format, which is a far broader
// promise than the three fields below need, and one this format would then owe
// forever.

impl KeteWrite for UncertainState {
    fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        let elements = &self.elements;
        elements.desig.write_to(w)?;
        elements.epoch.write_to(w)?;
        elements.semi_latus.write_to(w)?;
        elements.ecc_f.write_to(w)?;
        elements.ecc_g.write_to(w)?;
        elements.pole_h.write_to(w)?;
        elements.pole_k.write_to(w)?;
        elements.true_lon.write_to(w)?;
        elements.center_id.write_to(w)?;
        elements.gm_sqrt.write_to(w)?;

        write_matrix(&self.cov_matrix, w)?;
        self.free_params.write_to(w)?;

        // The measurement travels with the state. Dropping the probes would
        // silently restart `eta` on the next leg, which is the one way a march
        // can lose its history without saying so.
        match &self.probes {
            Some(probes) => {
                1_u8.write_to(w)?;
                write_probes(probes, w)?;
            }
            None => 0_u8.write_to(w)?,
        }
        match &self.whitening_cov {
            Some(reference) => {
                1_u8.write_to(w)?;
                write_matrix(reference, w)?;
            }
            None => 0_u8.write_to(w)?,
        }
        match &self.non_grav {
            Some(mask) => {
                1_u8.write_to(w)?;
                write_non_grav(mask, w)?;
            }
            None => 0_u8.write_to(w)?,
        }
        self.eta.write_to(w)?;
        self.residual_meters.write_to(w)
    }
}

impl KeteRead for UncertainState {
    fn read_from<R: Read>(r: &mut R) -> KeteResult<Self> {
        let elements = EquinoctialElements {
            desig: Desig::read_from(r)?,
            epoch: Time::read_from(r)?,
            semi_latus: f64::read_from(r)?,
            ecc_f: f64::read_from(r)?,
            ecc_g: f64::read_from(r)?,
            pole_h: f64::read_from(r)?,
            pole_k: f64::read_from(r)?,
            true_lon: f64::read_from(r)?,
            center_id: i32::read_from(r)?,
            gm_sqrt: f64::read_from(r)?,
        };
        let cov_matrix = read_matrix(r)?;
        let free_params = Vec::<f64>::read_from(r)?;
        let mut state = Self::new(elements, cov_matrix, free_params)?;

        state.probes = match u8::read_from(r)? {
            0 => None,
            1 => Some(read_probes(r)?),
            t => return Err(Error::IOError(format!("Invalid probe set tag: {t}"))),
        };
        state.whitening_cov = match u8::read_from(r)? {
            0 => None,
            1 => Some(read_matrix(r)?),
            t => return Err(Error::IOError(format!("Invalid whitening tag: {t}"))),
        };
        state.non_grav = match u8::read_from(r)? {
            0 => None,
            1 => Some(read_non_grav(r)?),
            t => return Err(Error::IOError(format!("Invalid non-grav tag: {t}"))),
        };
        if let Some(mask) = &state.non_grav {
            let free = mask.free_param_names().len();
            if free != state.free_params.len() {
                return Err(Error::IOError(format!(
                    "force model leaves {free} parameters free, but the state carries {}",
                    state.free_params.len()
                )));
            }
        }
        state.eta = Option::<f64>::read_from(r)?;
        state.residual_meters = Option::<f64>::read_from(r)?;
        Ok(state)
    }
}

// ---------------------------------------------------------------------------
// DiffuseState
// ---------------------------------------------------------------------------

impl KeteWrite for DiffuseState {
    fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        let mut payload = Vec::new();
        self.weights.write_to(&mut payload)?;
        self.components.write_to(&mut payload)?;
        self.include_asteroids.write_to(&mut payload)?;
        (payload.len() as u32).write_to(w)?;
        w.write_all(&payload)
    }
}

impl KeteRead for DiffuseState {
    fn read_from<R: Read>(r: &mut R) -> KeteResult<Self> {
        let entry_len = u32::read_from(r)? as usize;
        let mut payload = vec![0_u8; entry_len];
        r.read_exact(&mut payload)?;
        let mut cursor = Cursor::new(&payload);

        let weights = Vec::<f64>::read_from(&mut cursor)?;
        let components = Vec::<UncertainState>::read_from(&mut cursor)?;
        // `new` re-checks the weights against the components and the components
        // against each other, so a corrupt file fails here rather than producing
        // a mixture that does not describe a density.
        let mut mixture = Self::new(weights, components)?;
        mixture.include_asteroids = bool::read_from(&mut cursor)?;
        Ok(mixture)
    }
}

// ---------------------------------------------------------------------------
// File-level functions
// ---------------------------------------------------------------------------

/// Write the file header (magic, version, content type).
fn write_header<W: Write>(w: &mut W, content_type: u8) -> KeteResult<()> {
    w.write_all(MAGIC)?;
    VERSION.write_to(w)?;
    content_type.write_to(w)?;
    Ok(())
}

/// Read and validate the file header. Returns the content type byte.
fn read_header<R: Read>(r: &mut R) -> KeteResult<u8> {
    let mut magic = [0_u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(Error::IOError(
            "Invalid file: missing KETE magic bytes".into(),
        ));
    }
    let version = u16::read_from(r)?;
    if version > VERSION {
        return Err(Error::IOError(format!(
            "File version {version} is newer than supported version {VERSION}. \
             Please update kete."
        )));
    }
    u8::read_from(r)
}

/// Write a single [`SimultaneousStates`] to a kete binary file.
///
/// # Errors
/// Returns an error if writing to the stream fails.
pub fn write_single_kete_file<W: Write>(entry: &SimultaneousStates, w: &mut W) -> KeteResult<()> {
    write_header(w, CONTENT_TYPE_SINGLE)?;
    entry.write_to(w)?;
    Ok(())
}

/// Write a collection of [`SimultaneousStates`] entries to a kete binary file.
///
/// # Errors
/// Returns an error if writing to the stream fails.
pub fn write_vec_kete_file<W: Write>(entries: &[SimultaneousStates], w: &mut W) -> KeteResult<()> {
    write_header(w, CONTENT_TYPE_VEC)?;
    (entries.len() as u32).write_to(w)?;
    for entry in entries {
        entry.write_to(w)?;
    }
    Ok(())
}

/// Write a single [`UncertainState`] to a kete binary file.
///
/// # Errors
/// Returns an error if writing to the stream fails.
pub fn write_uncertain_kete_file<W: Write>(entry: &UncertainState, w: &mut W) -> KeteResult<()> {
    write_header(w, CONTENT_TYPE_UNCERTAIN)?;
    entry.write_to(w)?;
    Ok(())
}

/// Write a collection of [`UncertainState`] entries to a kete binary file.
///
/// One file holding many records is the intended shape for a set of states that
/// belong together, since a directory of one-record files costs a file handle
/// and a header per state.
///
/// # Errors
/// Returns an error if writing to the stream fails.
pub fn write_uncertain_vec_kete_file<W: Write>(
    entries: &[UncertainState],
    w: &mut W,
) -> KeteResult<()> {
    write_header(w, CONTENT_TYPE_UNCERTAIN_VEC)?;
    (entries.len() as u32).write_to(w)?;
    for entry in entries {
        entry.write_to(w)?;
    }
    Ok(())
}

/// Write a single [`DiffuseState`] to a kete binary file.
///
/// # Errors
/// Returns an error if writing to the stream fails.
pub fn write_diffuse_kete_file<W: Write>(entry: &DiffuseState, w: &mut W) -> KeteResult<()> {
    write_header(w, CONTENT_TYPE_DIFFUSE)?;
    entry.write_to(w)?;
    Ok(())
}

/// Write a collection of [`DiffuseState`] entries to a kete binary file.
///
/// One file holding many mixtures is the intended shape for a set that belongs
/// together, such as the cells of one dust simulation.
///
/// # Errors
/// Returns an error if writing to the stream fails.
pub fn write_diffuse_vec_kete_file<W: Write>(
    entries: &[DiffuseState],
    w: &mut W,
) -> KeteResult<()> {
    write_header(w, CONTENT_TYPE_DIFFUSE_VEC)?;
    (entries.len() as u32).write_to(w)?;
    for entry in entries {
        entry.write_to(w)?;
    }
    Ok(())
}

/// Read the contents of a kete binary file.
///
/// Returns a [`KeteFileType`] enum whose variant reflects what the file
/// header declares: which type it holds, and whether it holds one or many.
///
/// # Errors
/// Returns an error if the stream cannot be read, has invalid magic bytes,
/// an unsupported version, or contains corrupt data.
pub fn read_kete_file<R: Read>(r: &mut R) -> KeteResult<KeteFileType> {
    let content_type = read_header(r)?;
    match content_type {
        CONTENT_TYPE_SINGLE => Ok(KeteFileType::Single(Box::new(
            SimultaneousStates::read_from(r)?,
        ))),
        CONTENT_TYPE_VEC => {
            let n_entries = u32::read_from(r)? as usize;
            let mut entries = Vec::with_capacity(n_entries);
            for _ in 0..n_entries {
                entries.push(SimultaneousStates::read_from(r)?);
            }
            Ok(KeteFileType::Vec(entries))
        }
        CONTENT_TYPE_UNCERTAIN => Ok(KeteFileType::Uncertain(Box::new(
            UncertainState::read_from(r)?,
        ))),
        CONTENT_TYPE_UNCERTAIN_VEC => {
            let n_entries = u32::read_from(r)? as usize;
            let mut entries = Vec::with_capacity(n_entries);
            for _ in 0..n_entries {
                entries.push(UncertainState::read_from(r)?);
            }
            Ok(KeteFileType::UncertainVec(entries))
        }
        CONTENT_TYPE_DIFFUSE => Ok(KeteFileType::Diffuse(Box::new(DiffuseState::read_from(r)?))),
        CONTENT_TYPE_DIFFUSE_VEC => {
            let n_entries = u32::read_from(r)? as usize;
            let mut entries = Vec::with_capacity(n_entries);
            for _ in 0..n_entries {
                entries.push(DiffuseState::read_from(r)?);
            }
            Ok(KeteFileType::DiffuseVec(entries))
        }
        _ => Err(Error::IOError(format!(
            "Unsupported content type: {content_type}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::DiffuseState;

    /// Round-trip helper: write then read, assert equality.
    fn round_trip_write_read<T: KeteWrite + KeteRead + std::fmt::Debug + PartialEq>(val: &T) {
        let mut buf = Vec::new();
        val.write_to(&mut buf).unwrap();
        let mut cursor = Cursor::new(&buf);
        let recovered = T::read_from(&mut cursor).unwrap();
        assert_eq!(*val, recovered);
    }

    /// Round-trip helper for types that only implement Debug (not `PartialEq`).
    fn round_trip_debug<T: KeteWrite + KeteRead + std::fmt::Debug>(val: &T) {
        let mut buf = Vec::new();
        val.write_to(&mut buf).unwrap();
        let mut cursor = Cursor::new(&buf);
        let recovered = T::read_from(&mut cursor).unwrap();
        assert_eq!(format!("{val:?}"), format!("{recovered:?}"));
    }

    // -- Primitives --

    #[test]
    fn test_primitives() {
        round_trip_write_read(&42_u8);
        round_trip_write_read(&1234_u16);
        round_trip_write_read(&0xDEAD_BEEF_u32);
        round_trip_write_read(&0xCAFE_BABE_DEAD_BEEF_u64);
        round_trip_write_read(&-42_i32);
        round_trip_write_read(&1.5_f32);
        round_trip_write_read(&std::f64::consts::PI);
    }

    #[test]
    fn test_char_bool() {
        round_trip_write_read(&'A');
        round_trip_write_read(&'\u{1F680}');
        round_trip_write_read(&true);
        round_trip_write_read(&false);
    }

    #[test]
    fn test_option() {
        round_trip_write_read(&Some(42_u32));
        round_trip_write_read(&None::<u32>);
        round_trip_write_read(&Some('X'));
        round_trip_write_read(&None::<char>);
    }

    #[test]
    fn test_string() {
        let s = String::from("hello world");
        let mut buf = Vec::new();
        s.as_str().write_to(&mut buf).unwrap();
        let mut cursor = Cursor::new(&buf);
        let recovered = String::read_from(&mut cursor).unwrap();
        assert_eq!(s, recovered);
    }

    #[test]
    fn test_box_str() {
        let s: Box<str> = "test string".into();
        let mut buf = Vec::new();
        s.as_ref().write_to(&mut buf).unwrap();
        let mut cursor = Cursor::new(&buf);
        let recovered = Box::<str>::read_from(&mut cursor).unwrap();
        assert_eq!(s, recovered);
    }

    #[test]
    fn test_vec() {
        let v: Vec<u32> = vec![1, 2, 3, 4, 5];
        round_trip_write_read(&v);
        round_trip_write_read(&Vec::<u32>::new());
    }

    // -- Domain types --

    fn sample_vec() -> Vector<Equatorial> {
        Vector::new([1.0, 2.0, 3.0])
    }

    fn sample_time() -> Time<TDB> {
        Time::new(2451545.0)
    }

    fn sample_state() -> State<Equatorial> {
        State::new(
            Desig::Naif(399),
            sample_time(),
            Vector::new([1.0, 0.0, 0.0]),
            Vector::new([0.0, 1.0, 0.0]),
            10,
        )
    }

    /// An uncertain state carrying probes, which is what a marched component
    /// looks like and therefore what has to survive a file.
    fn sample_uncertain() -> UncertainState {
        let ecliptic = sample_state().into_frame::<crate::frames::Ecliptic>();
        let elements = EquinoctialElements::from_state(&ecliptic).unwrap();
        let mut cov = DMatrix::<f64>::zeros(7, 7);
        for i in 0..6 {
            cov[(i, i)] = 1e-12 * f64::from(i as u32 + 1);
        }
        cov[(6, 6)] = 1e-6;
        let mut state = UncertainState::new(elements, cov, vec![0.01]).unwrap();
        // The elements' center and the force center coincide here, so the
        // resolver is zero and no ephemeris is needed.
        let resolver = |_: Time<TDB>| Ok((Vector3::zeros(), Vector3::zeros()));
        state.non_grav = Some(ParameterMask {
            inner: NonGravKind::Dust(DustNonGrav),
            mask: vec![None],
        });
        state.probes = Some(ProbeSet::seed(&state, &resolver).unwrap());
        state.whitening_cov = Some(state.cov_matrix.clone());
        state.eta = Some(0.0123);
        state.residual_meters = Some(45.6);
        state
    }

    /// A state, its covariance, its probes and its reported numbers all come
    /// back. The probes are the part that matters: a reloaded component without
    /// them silently restarts its `eta` on the next leg.
    #[test]
    fn test_uncertain_state_round_trip() {
        let original = sample_uncertain();

        let mut buf = Vec::new();
        write_uncertain_kete_file(&original, &mut buf).unwrap();
        let mut cursor = Cursor::new(&buf);
        let KeteFileType::Uncertain(recovered) = read_kete_file(&mut cursor).unwrap() else {
            panic!("expected an Uncertain variant");
        };

        let got = &*recovered;
        assert_eq!(got.elements.epoch, original.elements.epoch);
        assert_eq!(got.elements.center_id, original.elements.center_id);
        assert!((got.elements.semi_latus - original.elements.semi_latus).abs() < 1e-15);
        assert_eq!(got.cov_matrix, original.cov_matrix);
        assert_eq!(got.free_params, original.free_params);
        assert_eq!(got.eta, original.eta);
        assert_eq!(got.residual_meters, original.residual_meters);
        assert_eq!(got.whitening_cov, original.whitening_cov);

        let (before, after) = (original.probes.unwrap(), got.probes.clone().unwrap());
        assert_eq!(before.directions.len(), after.directions.len());
        assert_eq!(before.cov_dim(), after.cov_dim());
        assert_eq!(before.epoch(), after.epoch());
        assert_eq!(before.stm, after.stm);
        assert_eq!(before.directions, after.directions);
        assert_eq!(before.displacements, after.displacements);
        assert_eq!(before.params, after.params);
        assert_eq!(
            before.states.iter().filter(|s| s.is_some()).count(),
            after.states.iter().filter(|s| s.is_some()).count()
        );

        assert!(got.non_grav.is_some());
    }

    /// Many records in one file, which is the shape a set of states that belong
    /// together needs. A file holding one reads back as a collection of one.
    #[test]
    fn test_uncertain_state_vec_round_trip() {
        let records: Vec<UncertainState> = (0..3).map(|_| sample_uncertain()).collect();

        let mut buf = Vec::new();
        write_uncertain_vec_kete_file(&records, &mut buf).unwrap();
        let mut cursor = Cursor::new(&buf);
        let KeteFileType::UncertainVec(recovered) = read_kete_file(&mut cursor).unwrap() else {
            panic!("expected an UncertainVec variant");
        };
        assert_eq!(recovered.len(), 3);
        assert!(recovered.iter().all(|r| r.non_grav.is_some()));
        assert!(recovered.iter().all(|r| r.probes.is_some()));
    }

    /// A model whose free parameter count disagrees with the state it is stored
    /// beside is a corrupt file rather than a state to be propagated wrongly.
    #[test]
    fn test_uncertain_state_rejects_mismatched_model() {
        let mut state = sample_uncertain();
        state.non_grav = Some(ParameterMask {
            inner: NonGravKind::Dust(DustNonGrav),
            // Frozen rather than free, so the model exposes no parameters while
            // the state carries one.
            mask: vec![Some(0.01)],
        });
        let mut buf = Vec::new();
        state.write_to(&mut buf).unwrap();
        let mut cursor = Cursor::new(&buf);
        assert!(UncertainState::read_from(&mut cursor).is_err());
    }

    /// A mixture, its components' probes, and the model it was propagated under
    /// all come back.
    #[test]
    fn test_diffuse_state_round_trip() {
        let mixture = DiffuseState::new(
            vec![0.25, 0.75],
            vec![sample_uncertain(), sample_uncertain()],
        )
        .unwrap();
        let mut record = mixture;
        record.include_asteroids = true;

        let mut buf = Vec::new();
        write_diffuse_kete_file(&record, &mut buf).unwrap();
        let mut cursor = Cursor::new(&buf);
        let KeteFileType::Diffuse(recovered) = read_kete_file(&mut cursor).unwrap() else {
            panic!("expected a Diffuse variant");
        };

        assert_eq!(recovered.weights, record.weights);
        assert_eq!(recovered.n_components(), 2);
        assert!(recovered.include_asteroids);
        for (got, want) in recovered.components.iter().zip(&record.components) {
            assert_eq!(got.cov_matrix, want.cov_matrix);
            assert_eq!(got.eta, want.eta);
            assert_eq!(got.free_params, want.free_params);
            assert!(got.probes.is_some());
            assert!(got.non_grav.is_some());
        }
    }

    /// Many mixtures in one file, which is the shape a dust cell set needs.
    #[test]
    fn test_diffuse_state_vec_round_trip() {
        let records: Vec<DiffuseState> = (0..3)
            .map(|_| DiffuseState::from_uncertain(sample_uncertain()))
            .collect();

        let mut buf = Vec::new();
        write_diffuse_vec_kete_file(&records, &mut buf).unwrap();
        let mut cursor = Cursor::new(&buf);
        let KeteFileType::DiffuseVec(recovered) = read_kete_file(&mut cursor).unwrap() else {
            panic!("expected a DiffuseVec variant");
        };
        assert_eq!(recovered.len(), 3);
        assert!(recovered.iter().all(|r| !r.include_asteroids));
    }

    /// Weights that do not describe the components are a corrupt file, caught by
    /// the same check that guards the constructor.
    #[test]
    fn test_diffuse_state_rejects_bad_weights() {
        let mut buf = Vec::new();
        // Two weights against one component, written past the constructor.
        let mut payload = Vec::new();
        vec![0.5_f64, 0.5].write_to(&mut payload).unwrap();
        vec![sample_uncertain()].write_to(&mut payload).unwrap();
        false.write_to(&mut payload).unwrap();
        (payload.len() as u32).write_to(&mut buf).unwrap();
        buf.extend_from_slice(&payload);
        let mut cursor = Cursor::new(&buf);
        assert!(DiffuseState::read_from(&mut cursor).is_err());
    }

    #[test]
    fn test_vector() {
        round_trip_write_read(&sample_vec());
    }

    #[test]
    fn test_time() {
        round_trip_write_read(&sample_time());
    }

    #[test]
    fn test_desig_all_variants() {
        round_trip_write_read(&Desig::Empty);
        round_trip_write_read(&Desig::Perm(12345));
        round_trip_write_read(&Desig::Prov("2024 AB".into()));
        round_trip_write_read(&Desig::CometPerm('C', 1, Some('a')));
        round_trip_write_read(&Desig::CometPerm('P', 2, None));
        round_trip_write_read(&Desig::CometProv(Some('D'), "2024 A1".into(), None));
        round_trip_write_read(&Desig::CometProv(None, "2024 B2".into(), Some('f')));
        round_trip_write_read(&Desig::PlanetSat(5, 1));
        round_trip_write_read(&Desig::Name("Ceres".into()));
        round_trip_write_read(&Desig::Naif(-42));
        round_trip_write_read(&Desig::ObservatoryCode("500".into()));
    }

    #[test]
    fn test_state() {
        round_trip_write_read(&sample_state());
    }

    #[test]
    fn test_ptf_filter() {
        round_trip_write_read(&PTFFilter::G);
        round_trip_write_read(&PTFFilter::R);
        round_trip_write_read(&PTFFilter::HA656);
        round_trip_write_read(&PTFFilter::HA663);
    }

    // -- FOV variants --

    fn sample_rectangle() -> OnSkyRectangle {
        let n = |x: f64, y: f64, z: f64| Vector::new([x, y, z]);
        OnSkyRectangle::from_normals([
            n(0.0, 0.0, 1.0),
            n(0.0, 1.0, 0.0),
            n(0.0, 0.0, -1.0),
            n(0.0, -1.0, 0.0),
        ])
    }

    fn sample_cone() -> SphericalCone {
        SphericalCone {
            pointing: Vector::new([1.0, 0.0, 0.0]),
            angle: 0.1,
        }
    }

    #[test]
    fn test_spherical_cone() {
        round_trip_debug(&sample_cone());
    }

    #[test]
    fn test_on_sky_rectangle() {
        round_trip_debug(&sample_rectangle());
    }

    #[test]
    fn test_fov_omni() {
        let fov = OmniDirectional {
            observer: sample_state(),
        };
        round_trip_debug(&fov);
    }

    #[test]
    fn test_fov_generic_cone() {
        let fov = GenericCone {
            observer: sample_state(),
            patch: sample_cone(),
        };
        round_trip_debug(&fov);
    }

    #[test]
    fn test_fov_generic_rectangle() {
        let fov = GenericRectangle {
            observer: sample_state(),
            patch: sample_rectangle(),
            rotation: 0.5,
        };
        round_trip_debug(&fov);
    }

    #[test]
    fn test_fov_wise() {
        let fov = WiseCmos {
            observer: sample_state(),
            patch: sample_rectangle(),
            frame_num: 12345,
            scan_id: "scan_001".into(),
        };
        round_trip_debug(&fov);
    }

    #[test]
    fn test_fov_neos_cmos() {
        let fov = NeosCmos {
            observer: sample_state(),
            patch: sample_rectangle(),
            rotation: 0.1,
            side_id: 1,
            stack_id: 2,
            quad_id: 3,
            loop_id: 4,
            subloop_id: 5,
            exposure_id: 6,
            band: 7,
            cmos_id: 8,
        };
        round_trip_debug(&fov);
    }

    fn sample_neos_cmos(cmos_id: u8) -> NeosCmos {
        NeosCmos {
            observer: sample_state(),
            patch: sample_rectangle(),
            rotation: 0.1,
            side_id: 1,
            stack_id: 2,
            quad_id: 3,
            loop_id: 4,
            subloop_id: 5,
            exposure_id: 6,
            band: 7,
            cmos_id,
        }
    }

    #[test]
    fn test_fov_neos_visit() {
        let fov = NeosVisit {
            chips: Box::new([
                sample_neos_cmos(0),
                sample_neos_cmos(1),
                sample_neos_cmos(2),
                sample_neos_cmos(3),
            ]),
            observer: sample_state(),
            rotation: 0.2,
            side_id: 10,
            stack_id: 11,
            quad_id: 12,
            loop_id: 13,
            subloop_id: 14,
            exposure_id: 15,
            band: 16,
        };
        round_trip_debug(&fov);
    }

    #[test]
    fn test_fov_ztf_ccd_quad() {
        let fov = ZtfCcdQuad {
            observer: sample_state(),
            patch: sample_rectangle(),
            field: 100,
            filefracday: 20240101,
            maglimit: 21.5,
            fid: 1,
            filtercode: "zr".into(),
            imgtypecode: "o".into(),
            ccdid: 5,
            qid: 2,
        };
        round_trip_debug(&fov);
    }

    fn sample_ztf_ccd_quad(qid: u8) -> ZtfCcdQuad {
        ZtfCcdQuad {
            observer: sample_state(),
            patch: sample_rectangle(),
            field: 100,
            filefracday: 20240101,
            maglimit: 21.5,
            fid: 1,
            filtercode: "zr".into(),
            imgtypecode: "o".into(),
            ccdid: 5,
            qid,
        }
    }

    #[test]
    fn test_fov_ztf_field() {
        let fov = ZtfField {
            ccd_quads: vec![sample_ztf_ccd_quad(1), sample_ztf_ccd_quad(2)],
            observer: sample_state(),
            field: 100,
            fid: 1,
            filtercode: "zr".into(),
            imgtypecode: "o".into(),
        };
        round_trip_debug(&fov);
    }

    #[test]
    fn test_fov_ptf_ccd() {
        let fov = PtfCcd {
            observer: sample_state(),
            patch: sample_rectangle(),
            field: 200,
            ccdid: 3,
            filter: PTFFilter::R,
            filename: "ptf_file.fits".into(),
            info_bits: 0,
            seeing: 2.5,
        };
        round_trip_debug(&fov);
    }

    fn sample_ptf_ccd(ccdid: u8) -> PtfCcd {
        PtfCcd {
            observer: sample_state(),
            patch: sample_rectangle(),
            field: 200,
            ccdid,
            filter: PTFFilter::R,
            filename: "ptf_file.fits".into(),
            info_bits: 0,
            seeing: 2.5,
        }
    }

    #[test]
    fn test_fov_ptf_field() {
        let fov = PtfField {
            ccds: vec![sample_ptf_ccd(1), sample_ptf_ccd(2)],
            observer: sample_state(),
            field: 200,
            filter: PTFFilter::R,
        };
        round_trip_debug(&fov);
    }

    #[test]
    fn test_fov_spherex_cmos() {
        let fov = SpherexCmos {
            observer: sample_state(),
            patch: sample_rectangle(),
            uri: "spx://data/001".into(),
            plane_id: "plane_A".into(),
        };
        round_trip_debug(&fov);
    }

    fn sample_spherex_cmos(plane: &str) -> SpherexCmos {
        SpherexCmos {
            observer: sample_state(),
            patch: sample_rectangle(),
            uri: "spx://data/001".into(),
            plane_id: plane.into(),
        }
    }

    #[test]
    fn test_fov_spherex_field() {
        let fov = SpherexField {
            cmos_frames: vec![sample_spherex_cmos("A"), sample_spherex_cmos("B")],
            observer: sample_state(),
            obsid: "obs_001".into(),
            observationid: "obsrv_001".into(),
        };
        round_trip_debug(&fov);
    }

    // -- FOV enum round-trip --

    #[test]
    fn test_fov_enum_round_trip() {
        let cases: Vec<FOV> = vec![
            FOV::OmniDirectional(OmniDirectional {
                observer: sample_state(),
            }),
            FOV::GenericCone(GenericCone {
                observer: sample_state(),
                patch: sample_cone(),
            }),
            FOV::GenericRectangle(GenericRectangle {
                observer: sample_state(),
                patch: sample_rectangle(),
                rotation: 0.5,
            }),
            FOV::Wise(WiseCmos {
                observer: sample_state(),
                patch: sample_rectangle(),
                frame_num: 1,
                scan_id: "s".into(),
            }),
        ];
        for fov in &cases {
            let mut buf = Vec::new();
            fov.write_to(&mut buf).unwrap();
            let mut cursor = Cursor::new(&buf);
            let recovered = read_fov(&mut cursor).unwrap().unwrap();
            // Compare debug representations since FOV doesn't impl PartialEq
            assert_eq!(format!("{fov:?}"), format!("{recovered:?}"));
        }
    }

    // -- Unknown FOV tag skipping --

    #[test]
    fn test_unknown_fov_tag_skipped() {
        let mut buf = Vec::new();
        // Write a fake FOV with tag 255, payload of 10 bytes
        255_u8.write_to(&mut buf).unwrap();
        10_u32.write_to(&mut buf).unwrap();
        buf.extend_from_slice(&[0_u8; 10]);
        // Append a sentinel byte to ensure stream is correctly positioned
        42_u8.write_to(&mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let result = read_fov(&mut cursor).unwrap();
        assert!(result.is_none());
        // Verify the cursor consumed exactly the FOV payload
        assert_eq!(u8::read_from(&mut cursor).unwrap(), 42);
    }

    // -- SimultaneousStates --

    #[test]
    fn test_simult_states_no_fov() {
        let ss = SimultaneousStates::new_exact(vec![sample_state()], None).unwrap();
        round_trip_debug(&ss);
    }

    #[test]
    fn test_simult_states_with_fov() {
        let fov = Some(FOV::OmniDirectional(OmniDirectional {
            observer: sample_state(),
        }));
        let ss = SimultaneousStates::new_exact(vec![sample_state()], fov).unwrap();
        let mut buf = Vec::new();
        ss.write_to(&mut buf).unwrap();
        let mut cursor = Cursor::new(&buf);
        let recovered = SimultaneousStates::read_from(&mut cursor).unwrap();
        assert_eq!(ss.epoch().jd, recovered.epoch().jd);
        assert_eq!(ss.center_id(), recovered.center_id());
        assert_eq!(ss.states.len(), recovered.states.len());
        assert!(recovered.fov.is_some());
    }

    #[test]
    fn test_simult_states_empty() {
        assert!(SimultaneousStates::new_exact(vec![], None).is_err());
    }

    // -- File-level round-trip (single) --

    #[test]
    fn test_single_file_round_trip() {
        let fov = Some(FOV::OmniDirectional(OmniDirectional {
            observer: sample_state(),
        }));
        let entry = SimultaneousStates::new_exact(vec![sample_state()], fov).unwrap();
        let mut buf = Vec::new();
        write_single_kete_file(&entry, &mut buf).unwrap();
        let mut cursor = Cursor::new(&buf);
        let data = read_kete_file(&mut cursor).unwrap();
        match data {
            KeteFileType::Single(recovered) => {
                assert_eq!(entry.epoch().jd, recovered.epoch().jd);
                assert_eq!(entry.center_id(), recovered.center_id());
                assert_eq!(entry.states.len(), recovered.states.len());
                assert!(recovered.fov.is_some());
            }
            KeteFileType::Vec(_)
            | KeteFileType::Uncertain(_)
            | KeteFileType::UncertainVec(_)
            | KeteFileType::Diffuse(_)
            | KeteFileType::DiffuseVec(_) => panic!("expected Single variant"),
        }
    }

    // -- File-level round-trip (vec) --

    #[test]
    fn test_vec_file_round_trip() {
        let state2 = State::new(
            Desig::Naif(399),
            Time::new(2460000.0),
            Vector::new([2.0, 0.0, 0.0]),
            Vector::new([0.0, 1.0, 0.0]),
            10,
        );
        let obs2 = State::new(
            Desig::Naif(399),
            Time::new(2460000.0),
            Vector::new([1.0, 0.0, 0.0]),
            Vector::new([0.0, 0.5, 0.0]),
            10,
        );
        let entries = vec![
            SimultaneousStates::new_exact(vec![sample_state()], None).unwrap(),
            SimultaneousStates::new_exact(
                vec![state2],
                Some(FOV::GenericCone(GenericCone {
                    observer: obs2,
                    patch: sample_cone(),
                })),
            )
            .unwrap(),
        ];
        let mut buf = Vec::new();
        write_vec_kete_file(&entries, &mut buf).unwrap();
        let mut cursor = Cursor::new(&buf);
        let data = read_kete_file(&mut cursor).unwrap();
        match data {
            KeteFileType::Vec(recovered) => {
                assert_eq!(entries.len(), recovered.len());
                for (orig, rec) in entries.iter().zip(recovered.iter()) {
                    assert_eq!(orig.epoch().jd, rec.epoch().jd);
                    assert_eq!(orig.center_id(), rec.center_id());
                    assert_eq!(orig.states.len(), rec.states.len());
                }
            }
            KeteFileType::Single(_)
            | KeteFileType::Uncertain(_)
            | KeteFileType::UncertainVec(_)
            | KeteFileType::Diffuse(_)
            | KeteFileType::DiffuseVec(_) => panic!("expected Vec variant"),
        }
    }

    #[test]
    fn test_file_bad_magic() {
        let buf = b"NOPE\x01\x00\x00\x00\x00\x00\x00";
        let mut cursor = Cursor::new(&buf[..]);
        let result = read_kete_file(&mut cursor);
        assert!(result.is_err());
    }

    #[test]
    fn test_file_future_version() {
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        99_u16.write_to(&mut buf).unwrap();
        0_u8.write_to(&mut buf).unwrap();
        0_u32.write_to(&mut buf).unwrap();
        let mut cursor = Cursor::new(&buf);
        let result = read_kete_file(&mut cursor);
        assert!(result.is_err());
    }

    #[test]
    fn test_file_unsupported_content_type() {
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        VERSION.write_to(&mut buf).unwrap();
        99_u8.write_to(&mut buf).unwrap();
        0_u32.write_to(&mut buf).unwrap();
        let mut cursor = Cursor::new(&buf);
        let result = read_kete_file(&mut cursor);
        assert!(result.is_err());
    }
}
