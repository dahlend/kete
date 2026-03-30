//! Binary codec for kete file format.
//!
//! Defines [`KeteWrite`] / [`KeteRead`] traits and implementations for all types
//! that participate in the kete binary file format. Also provides file-level
//! [`write_kete_file`] and [`read_kete_file`] functions that handle the header
//! (magic bytes, version, content type, entry count).
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
use crate::errors::{Error, KeteResult};
use crate::fov::{
    FOV, GenericCone, GenericRectangle, NeosCmos, NeosVisit, OmniDirectional, OnSkyRectangle,
    PTFFilter, PtfCcd, PtfField, SpherexCmos, SpherexField, SphericalCone, WiseCmos, ZtfCcdQuad,
    ZtfField,
};
use crate::frames::{Equatorial, Vector};
use crate::simult_states::SimultaneousStates;
use crate::state::State;
use crate::time::{TDB, Time};
use std::io::{self, Cursor, Read, Write};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAGIC: &[u8; 4] = b"KETE";
const VERSION: u16 = 1;
const CONTENT_TYPE_SINGLE: u8 = 0;
const CONTENT_TYPE_VEC: u8 = 1;

/// The payload read from a kete binary file.
///
/// The content type in the file header determines which variant is returned.
#[derive(Debug, Clone)]
pub enum KeteFileType {
    /// A single [`SimultaneousStates`] (content type 0).
    Single(Box<SimultaneousStates>),
    /// A collection of [`SimultaneousStates`] (content type 1).
    Vec(Vec<SimultaneousStates>),
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
        self.center_id.write_to(w)
    }
}

impl KeteRead for State<Equatorial> {
    fn read_from<R: Read>(r: &mut R) -> KeteResult<Self> {
        Ok(Self {
            desig: Desig::read_from(r)?,
            epoch: Time::read_from(r)?,
            pos: Vector::read_from(r)?,
            vel: Vector::read_from(r)?,
            center_id: i32::read_from(r)?,
        })
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
        };
        tag.write_to(w)?;
        (payload.len() as u32).write_to(w)?;
        w.write_all(&payload)
    }
}

/// Read an FOV from the stream. Returns `None` if the tag is unknown
/// (forward compatibility — unknown variants are skipped).
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
        self.epoch.write_to(&mut payload)?;
        self.center_id.write_to(&mut payload)?;
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

        let epoch = Time::read_from(&mut cursor)?;
        let center_id = i32::read_from(&mut cursor)?;
        let fov = Option::<FOV>::read_from(&mut cursor)?;
        let states = Vec::read_from(&mut cursor)?;
        Ok(Self {
            states,
            epoch,
            center_id,
            fov,
        })
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

/// Read [`SimultaneousStates`] from a kete binary file.
///
/// Returns a [`KeteFileType`] enum whose variant reflects what the file
/// header declares: a single entry or a collection.
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
        round_trip_write_read(&'🚀');
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
        State {
            desig: Desig::Naif(399),
            epoch: sample_time(),
            pos: Vector::new([1.0, 0.0, 0.0]),
            vel: Vector::new([0.0, 1.0, 0.0]),
            center_id: 10,
        }
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
        let ss = SimultaneousStates {
            states: vec![sample_state()],
            epoch: sample_time(),
            center_id: 10,
            fov: None,
        };
        round_trip_debug(&ss);
    }

    #[test]
    fn test_simult_states_with_fov() {
        let ss = SimultaneousStates {
            states: vec![sample_state()],
            epoch: sample_time(),
            center_id: 10,
            fov: Some(FOV::OmniDirectional(OmniDirectional {
                observer: sample_state(),
            })),
        };
        let mut buf = Vec::new();
        ss.write_to(&mut buf).unwrap();
        let mut cursor = Cursor::new(&buf);
        let recovered = SimultaneousStates::read_from(&mut cursor).unwrap();
        assert_eq!(ss.epoch.jd, recovered.epoch.jd);
        assert_eq!(ss.center_id, recovered.center_id);
        assert_eq!(ss.states.len(), recovered.states.len());
        assert!(recovered.fov.is_some());
    }

    #[test]
    fn test_simult_states_empty() {
        let ss = SimultaneousStates {
            states: vec![],
            epoch: sample_time(),
            center_id: 0,
            fov: None,
        };
        round_trip_debug(&ss);
    }

    // -- File-level round-trip (single) --

    #[test]
    fn test_single_file_round_trip() {
        let entry = SimultaneousStates {
            states: vec![sample_state()],
            epoch: sample_time(),
            center_id: 10,
            fov: Some(FOV::OmniDirectional(OmniDirectional {
                observer: sample_state(),
            })),
        };
        let mut buf = Vec::new();
        write_single_kete_file(&entry, &mut buf).unwrap();
        let mut cursor = Cursor::new(&buf);
        let data = read_kete_file(&mut cursor).unwrap();
        match data {
            KeteFileType::Single(recovered) => {
                assert_eq!(entry.epoch.jd, recovered.epoch.jd);
                assert_eq!(entry.center_id, recovered.center_id);
                assert_eq!(entry.states.len(), recovered.states.len());
                assert!(recovered.fov.is_some());
            }
            KeteFileType::Vec(_) => panic!("expected Single variant"),
        }
    }

    // -- File-level round-trip (vec) --

    #[test]
    fn test_vec_file_round_trip() {
        let entries = vec![
            SimultaneousStates {
                states: vec![sample_state()],
                epoch: sample_time(),
                center_id: 10,
                fov: None,
            },
            SimultaneousStates {
                states: vec![],
                epoch: Time::new(2460000.0),
                center_id: 0,
                fov: Some(FOV::GenericCone(GenericCone {
                    observer: sample_state(),
                    patch: sample_cone(),
                })),
            },
        ];
        let mut buf = Vec::new();
        write_vec_kete_file(&entries, &mut buf).unwrap();
        let mut cursor = Cursor::new(&buf);
        let data = read_kete_file(&mut cursor).unwrap();
        match data {
            KeteFileType::Vec(recovered) => {
                assert_eq!(entries.len(), recovered.len());
                for (orig, rec) in entries.iter().zip(recovered.iter()) {
                    assert_eq!(orig.epoch.jd, rec.epoch.jd);
                    assert_eq!(orig.center_id, rec.center_id);
                    assert_eq!(orig.states.len(), rec.states.len());
                }
            }
            KeteFileType::Single(_) => panic!("expected Vec variant"),
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
