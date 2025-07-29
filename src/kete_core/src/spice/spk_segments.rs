//! Most users should interface with `spk.rs`, not this module.
//!
//! SPK Files are collections of `Segments`, which are ranges of times where the state
//! of an object is recorded. These segments are typically made up of many individual
//! `Records`, with an associated maximum and minimum time where they are valid for.
//!
//! There are unique structs for each possible segment type, not all are currently
//! supported.
//!
//! <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/spk.html#Supported%20Data%20Types>
//!
//! There is a lot of repetition in this file, as many of the segment types have very
//! similar internal structures.
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

use super::SpkArray;
use super::interpolation::{
    chebyshev_evaluate, chebyshev_evaluate_both, hermite_interpolation, lagrange_interpolation,
};
use super::{jd_to_spice_jd, spice_jd_to_jd};
use crate::constants::AU_KM;
use crate::errors::Error;
use crate::frames::{Ecliptic, Equatorial, FK4, Galactic, InertialFrame};
use crate::prelude::{Desig, KeteResult};
use crate::state::State;
use crate::time::{TDB, Time};
use itertools::Itertools;
use sgp4::{Constants, Geopotential, MinutesSinceEpoch, Orbit, julian_years_since_j2000};
use std::fmt::Debug;

#[derive(Debug)]
pub(in crate::spice) enum SpkSegment {
    Type1(SpkSegmentType1),
    Type2(SpkSegmentType2),
    Type3(SpkSegmentType3),
    Type9(SpkSegmentType9),
    Type10(SpkSegmentType10),
    Type13(SpkSegmentType13),
    Type18(SpkSegmentType18),
    Type21(SpkSegmentType21),
}

impl TryFrom<SpkArray> for SpkSegment {
    type Error = Error;

    fn try_from(array: SpkArray) -> Result<Self, Self::Error> {
        match array.segment_type {
            1 => Ok(Self::Type1(array.into())),
            2 => Ok(Self::Type2(array.try_into()?)),
            3 => Ok(Self::Type3(array.try_into()?)),
            9 => Ok(Self::Type9(array.into())),
            10 => Ok(Self::Type10(array.try_into()?)),
            13 => Ok(Self::Type13(array.into())),
            18 => Ok(Self::Type18(array.try_into()?)),
            21 => Ok(Self::Type21(array.into())),
            v => Err(Error::IOError(format!(
                "SPK Segment type {v} not supported. Please submit a github issue!",
            ))),
        }
    }
}

impl<'a> From<&'a SpkSegment> for &'a SpkArray {
    fn from(segment: &'a SpkSegment) -> Self {
        match segment {
            SpkSegment::Type1(seg) => &seg.array,
            SpkSegment::Type2(seg) => &seg.array,
            SpkSegment::Type3(v) => &v.array,
            SpkSegment::Type9(v) => &v.array,
            SpkSegment::Type10(v) => &v.array.array,
            SpkSegment::Type13(v) => &v.array,
            SpkSegment::Type18(v) => &v.array,
            SpkSegment::Type21(v) => &v.array,
        }
    }
}

impl SpkSegment {
    /// Return the [`State`] object at the specified JD. If the requested time is
    /// not within the available range, this will fail.
    #[inline(always)]
    pub(in crate::spice) fn try_get_state<T: InertialFrame>(
        &self,
        jd: f64,
    ) -> KeteResult<State<T>> {
        let arr_ref: &SpkArray = self.into();

        let jds = jd_to_spice_jd(jd);

        // this is faster than calling contains, probably because the || instead of &&
        if jds < arr_ref.jds_start || jds > arr_ref.jds_end {
            return Err(Error::DAFLimits(
                "JD is not present in this record.".to_string(),
            ));
        }

        let (pos, vel) = match &self {
            Self::Type1(v) => v.try_get_pos_vel(jds)?,
            Self::Type2(v) => v.try_get_pos_vel(jds)?,
            Self::Type3(v) => v.try_get_pos_vel(jds)?,
            Self::Type9(v) => v.try_get_pos_vel(jds),
            Self::Type10(v) => v.try_get_pos_vel(jds),
            Self::Type13(v) => v.try_get_pos_vel(jds),
            Self::Type18(v) => v.try_get_pos_vel(jds),
            Self::Type21(v) => v.try_get_pos_vel(jds)?,
        };

        match arr_ref.frame_id {
            1 => Ok(State::<Equatorial>::new(
                Desig::Naif(arr_ref.object_id),
                jd,
                pos.into(),
                vel.into(),
                arr_ref.center_id,
            )
            .into_frame()),
            3 => Ok(State::<FK4>::new(
                Desig::Naif(arr_ref.object_id),
                jd,
                pos.into(),
                vel.into(),
                arr_ref.center_id,
            )
            .into_frame()),
            13 => Ok(State::<Galactic>::new(
                Desig::Naif(arr_ref.object_id),
                jd,
                pos.into(),
                vel.into(),
                arr_ref.center_id,
            )
            .into_frame()),
            17 => Ok(State::<Ecliptic>::new(
                Desig::Naif(arr_ref.object_id),
                jd,
                pos.into(),
                vel.into(),
                arr_ref.center_id,
            )
            .into_frame()),
            _ => Err(Error::ValueError(format!(
                "Frame {:?} not supported. Please submit a github issue! Please include the SPK file!",
                arr_ref.frame_id
            )))?,
        }
    }
}

/// Modified Difference Arrays
///
/// <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/spk.html#Type%201:%20Modified%20Difference%20Arrays>
///
// This format might be derived from works related to this paper:
// Recurrence Relations for Computing With Modified Divided Differences*
// Fred Krogh 1979
#[derive(Debug)]
pub(in crate::spice) struct SpkSegmentType1 {
    array: SpkArray,

    n_records: usize,
}

impl SpkSegmentType1 {
    #[inline(always)]
    fn get_record(&self, idx: usize) -> &[f64] {
        unsafe { self.array.daf.data.get_unchecked(idx * 71..(idx + 1) * 71) }
    }

    #[inline(always)]
    fn get_times(&self) -> &[f64] {
        unsafe {
            self.array
                .daf
                .data
                .get_unchecked(self.n_records * 71..(self.n_records * 72))
        }
    }

    #[inline(always)]
    fn try_get_pos_vel(&self, jds: f64) -> KeteResult<([f64; 3], [f64; 3])> {
        // Records are laid out as so:
        //
        // Size      Description
        // ----------------------
        // 1          Reference Epoch for the difference line
        // n_coef     Step size function vector
        // 6          Reference state - x, vx, y, vy, z, vz  (interleaved order)
        // 3*n_coef   Modified divided difference arrays
        // 1          Maximum integration order plus 1
        // 3          Integration order array
        // total: 11 + 4*n_coef
        // we need to find the first record which has a time greater than or equal
        // to the target jd.

        let start_idx = self
            .get_times()
            .binary_search_by(|probe| probe.total_cmp(&jds))
            .unwrap_or_else(|c| c);

        let record = self.get_record(start_idx);

        let ref_time = record[0];

        let func_vec = &record[1..16];
        let ref_state = &record[16..22];

        let divided_diff_array = &record[22..67];

        let kq_max1 = record[67] as usize;
        let kq = &record[68..71];

        // in the spice code ref_time is in seconds from j2000
        let dt = jds - ref_time;

        let mut fc = [0.0; 15];
        let mut wc = [0.0; 15];

        let mut tp = dt;
        for idx in 0..(kq_max1 - 2) {
            let f = func_vec[idx];
            if f == 0.0 {
                // don't divide by 0 below, file was built incorrectly.
                Err(Error::IOError(
                    "SPK File contains segments of type 1 has invalid contents.".into(),
                ))?;
            }

            fc[idx] = tp / f;
            wc[idx] = dt / f;
            tp = dt + f;
        }

        let mut w: Box<[f64]> = { (0..kq_max1).map(|x| (x as f64 + 1.0).recip()).collect() };

        let mut ks = kq_max1 - 1;
        let mut jx = 0;
        let mut ks1 = ks - 1;

        while ks >= 2 {
            jx += 1;
            for j in 0..jx {
                w[j + ks] = fc[j] * w[j + ks1] - wc[j] * w[j + ks];
            }
            ks = ks1;
            ks1 -= 1;
        }

        // position interpolation
        let pos = std::array::from_fn(|idx| {
            let sum: f64 = (1..(kq[idx] as usize + 1))
                .rev()
                .map(|j| divided_diff_array[15 * idx + j - 1] * w[j + ks - 1])
                .sum();
            (ref_state[2 * idx] + dt * (sum * dt + ref_state[2 * idx + 1])) / AU_KM
        });

        // Recompute W for velocities
        for j in 0..jx {
            w[j + ks] = fc[j] * w[j + ks1] - wc[j] * w[j + ks];
        }
        ks -= 1;

        // velocity interpolation
        let vel = std::array::from_fn(|idx| {
            let sum: f64 = (1..=(kq[idx] as usize))
                .rev()
                .map(|j| divided_diff_array[15 * idx + j - 1] * w[j + ks - 1])
                .sum();
            (ref_state[2 * idx + 1] + dt * sum) / AU_KM * 86400.0
        });
        Ok((pos, vel))
    }
}

impl From<SpkArray> for SpkSegmentType1 {
    fn from(array: SpkArray) -> Self {
        let n_records = array.daf[array.daf.len() - 1] as usize;

        Self { array, n_records }
    }
}

/// Chebyshev Polynomials (Position Only)
///
/// <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/spk.html#Type%202:%20Chebyshev%20position%20only>
///
#[derive(Debug)]
pub(in crate::spice) struct SpkSegmentType2 {
    array: SpkArray,
    jds_step: f64,
    n_coef: usize,
    record_len: usize,
}

/// Type 2 Record View
/// A view into a record of type 2, provided mainly for clarity to the underlying
/// data structure.
struct Type2RecordView<'a> {
    t_mid: &'a f64,
    t_step: &'a f64,

    x_coef: &'a [f64],
    y_coef: &'a [f64],
    z_coef: &'a [f64],
}

impl SpkSegmentType2 {
    #[inline(always)]
    fn get_record(&self, idx: usize) -> Type2RecordView<'_> {
        unsafe {
            let vals = self
                .array
                .daf
                .data
                .get_unchecked(idx * self.record_len..(idx + 1) * self.record_len);

            Type2RecordView {
                t_mid: vals.get_unchecked(0),
                t_step: vals.get_unchecked(1),
                x_coef: vals.get_unchecked(2..(self.n_coef + 2)),
                y_coef: vals.get_unchecked((self.n_coef + 2)..(2 * self.n_coef + 2)),
                z_coef: vals.get_unchecked((2 * self.n_coef + 2)..(3 * self.n_coef + 2)),
            }
        }
    }

    #[inline(always)]
    fn try_get_pos_vel(&self, jds: f64) -> KeteResult<([f64; 3], [f64; 3])> {
        let jds_start = self.array.jds_start;
        let record_index = ((jds - jds_start) / self.jds_step).floor() as usize;
        let record = self.get_record(record_index);

        let t_step = record.t_step;

        let t = (jds - record.t_mid) / t_step;

        let t_step_scaled = 86400.0 / t_step / AU_KM;

        let (p, v) = chebyshev_evaluate_both(t, record.x_coef, record.y_coef, record.z_coef)?;
        Ok((
            [p[0] / AU_KM, p[1] / AU_KM, p[2] / AU_KM],
            [
                v[0] * t_step_scaled,
                v[1] * t_step_scaled,
                v[2] * t_step_scaled,
            ],
        ))
    }
}

impl TryFrom<SpkArray> for SpkSegmentType2 {
    type Error = Error;

    fn try_from(array: SpkArray) -> Result<Self, Self::Error> {
        let record_len = array.daf[array.daf.len() - 2] as usize;
        let jds_step = array.daf[array.daf.len() - 3];

        let n_coef = (record_len - 2) / 3;

        if 3 * n_coef + 2 != record_len {
            return Err(Error::ValueError("File incorrectly formatted, found number of Chebyshev coefficients doesn't match expected".into()));
        }

        Ok(Self {
            array,
            jds_step,
            n_coef,
            record_len,
        })
    }
}

/// Type 3 - Chebyshev Polynomials (Position & Velocity)
///
/// <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/req/spk.html#Type%203:%20Chebyshev%20position%20and%20velocity>
///
#[derive(Debug)]
pub(in crate::spice) struct SpkSegmentType3 {
    array: SpkArray,
    jds_step: f64,
    n_coef: usize,
    record_len: usize,
}

/// Type 3 Record View
/// A view into a record of type 3, provided mainly for clarity to the underlying
/// data structure.
struct Type3RecordView<'a> {
    t_mid: &'a f64,
    t_step: &'a f64,

    x_coef: &'a [f64],
    y_coef: &'a [f64],
    z_coef: &'a [f64],

    vx_coef: &'a [f64],
    vy_coef: &'a [f64],
    vz_coef: &'a [f64],
}

impl SpkSegmentType3 {
    #[inline(always)]
    fn get_record(&self, idx: usize) -> Type3RecordView<'_> {
        unsafe {
            let vals = self
                .array
                .daf
                .data
                .get_unchecked(idx * self.record_len..(idx + 1) * self.record_len);

            Type3RecordView {
                t_mid: vals.get_unchecked(0),
                t_step: vals.get_unchecked(1),
                x_coef: vals.get_unchecked(2..(self.n_coef + 2)),
                y_coef: vals.get_unchecked((self.n_coef + 2)..(2 * self.n_coef + 2)),
                z_coef: vals.get_unchecked((2 * self.n_coef + 2)..(3 * self.n_coef + 2)),
                vx_coef: vals.get_unchecked((3 * self.n_coef + 2)..(4 * self.n_coef + 2)),
                vy_coef: vals.get_unchecked((4 * self.n_coef + 2)..(5 * self.n_coef + 2)),
                vz_coef: vals.get_unchecked((5 * self.n_coef + 2)..(6 * self.n_coef + 2)),
            }
        }
    }

    #[inline(always)]
    fn try_get_pos_vel(&self, jds: f64) -> KeteResult<([f64; 3], [f64; 3])> {
        let jds_start = self.array.jds_start;
        let record_index = ((jds - jds_start) / self.jds_step).floor() as usize;
        let record = self.get_record(record_index);

        let t_step = record.t_step;

        let t = (jds - record.t_mid) / t_step;

        let t_scaled = 86400.0 / AU_KM;

        let p = chebyshev_evaluate(t, record.x_coef, record.y_coef, record.z_coef)?;
        let v = chebyshev_evaluate(t, record.vx_coef, record.vy_coef, record.vz_coef)?;
        Ok((
            [p[0] / AU_KM, p[1] / AU_KM, p[2] / AU_KM],
            [v[0] * t_scaled, v[1] * t_scaled, v[2] * t_scaled],
        ))
    }
}

impl TryFrom<SpkArray> for SpkSegmentType3 {
    type Error = Error;
    fn try_from(array: SpkArray) -> KeteResult<Self> {
        let record_len = array.daf[array.daf.len() - 2] as usize;
        let jds_step = array.daf[array.daf.len() - 3];

        let n_coef = (record_len - 2) / 6;

        if 6 * n_coef + 2 != record_len {
            return Err(Error::ValueError("File incorrectly formatted, found number of Chebyshev coefficients doesn't match expected".into()));
        }

        Ok(Self {
            array,
            jds_step,
            n_coef,
            record_len,
        })
    }
}

// TODO: SPK Segment type 8 should be a minor variation on type 9. This was not
// implemented here due to missing a valid SPK file to test against.

/// Lagrange Interpolation (Uneven Time Steps)
///
/// This uses a collection of individual positions/velocities and interpolates between
/// them using Lagrange interpolation.
/// <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/req/spk.html#Type%209:%20Lagrange%20Interpolation%20---%20Unequal%20Time%20Steps>
#[derive(Debug)]
pub(in crate::spice) struct SpkSegmentType9 {
    array: SpkArray,
    poly_degree: usize,
    n_records: usize,
}

impl From<SpkArray> for SpkSegmentType9 {
    fn from(array: SpkArray) -> Self {
        let n_records = array.daf[array.daf.len() - 1] as usize;
        let mut poly_degree = array.daf[array.daf.len() - 2] as usize;

        if poly_degree + 1 > n_records {
            eprintln!(
                "Spk Segment Type 9 must have at least as many records as the polynomial degree, n_records={n_records}, poly_degree={poly_degree}",
            );
            poly_degree = n_records - 1;
        }

        Self {
            array,
            poly_degree,
            n_records,
        }
    }
}

/// Type 9 Record View
/// A view into a record of type 9, provided mainly for clarity to the underlying
/// data structure.
struct Type9RecordView<'a> {
    pos: &'a [f64; 3],
    vel: &'a [f64; 3],
}

impl SpkSegmentType9 {
    #[inline(always)]
    fn get_record(&self, idx: usize) -> Type9RecordView<'_> {
        unsafe {
            let rec = self.array.daf.data.get_unchecked(idx * 6..(idx + 1) * 6);
            Type9RecordView {
                pos: rec[0..3].try_into().unwrap(),
                vel: rec[3..6].try_into().unwrap(),
            }
        }
    }

    #[inline(always)]
    fn get_times(&self) -> &[f64] {
        unsafe {
            self.array
                .daf
                .data
                .get_unchecked(self.n_records * 6..self.n_records * 7)
        }
    }

    #[inline(always)]
    fn try_get_pos_vel(&self, jds: f64) -> ([f64; 3], [f64; 3]) {
        let times = self.get_times();
        let window_size = self.poly_degree + 1;
        let start_idx: isize = match times.binary_search_by(|probe| probe.total_cmp(&jds)) {
            Ok(c) => c as isize - (window_size as isize) / 2,
            Err(c) => {
                if (jds - times[c - 1]).abs() < (jds - times[c]).abs() {
                    c as isize - 1 - window_size as isize / 2
                } else {
                    c as isize - window_size as isize / 2
                }
            }
        };
        let start_idx = start_idx.clamp(0, (self.n_records - window_size) as isize) as usize;

        let mut pos = [0.0; 3];
        let mut vel = [0.0; 3];
        for idx in 0..3 {
            let mut p: Box<[f64]> = (0..window_size)
                .map(|i| self.get_record(i + start_idx).pos[idx])
                .collect();
            let mut dp: Box<[f64]> = (0..window_size)
                .map(|i| self.get_record(i + start_idx).vel[idx])
                .collect();
            let p = lagrange_interpolation(&times[start_idx..start_idx + window_size], &mut p, jds);
            let v =
                lagrange_interpolation(&times[start_idx..start_idx + window_size], &mut dp, jds);
            pos[idx] = p / AU_KM;
            vel[idx] = v / AU_KM * 86400.;
        }

        (pos, vel)
    }
}

/// Space Command two-line elements
///
/// <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/spk.html#Type%2010:%20Space%20Command%20Two-Line%20Elements>
///
#[derive(Debug)]
pub(in crate::spice) struct SpkSegmentType10 {
    /// Generic Segments are a collection of a few different directories:
    /// `Packets` are where Type 10 stores the TLE values.
    /// `Packet Directory` is unused.
    /// `Reference Directory` is where the 100 step JDs are stored.
    /// `Reference Items` is a list of all JDs
    array: GenericSegment,

    /// spg4 uses a geopotential model which is loaded from the spice kernel.
    /// Unfortunately SGP4 doesn't support custom altitude bounds, but this
    /// probably shouldn't be altered from the defaults.
    geopotential: Geopotential,
}

impl SpkSegmentType10 {
    #[inline(always)]
    fn get_times(&self) -> &[f64] {
        self.array.get_reference_items()
    }

    /// Return the SGP4 record stored within the spice kernel.
    #[inline(always)]
    fn get_record(&self, idx: usize) -> Constants {
        let rec = self.array.get_packet::<15>(idx);
        let [
            _,
            _,
            _,
            b_star,
            inclination,
            right_ascension,
            eccentricity,
            argument_of_perigee,
            mean_anomaly,
            kozai_mean_motion,
            epoch,
            _,
            _,
            _,
            _,
        ] = *rec;

        let epoch = julian_years_since_j2000(
            &Time::<TDB>::new(spice_jd_to_jd(epoch))
                .utc()
                .to_datetime()
                .unwrap()
                .naive_utc(),
        );

        // use the provided goepotential even if it is not correct.
        let orbit_0 = Orbit::from_kozai_elements(
            &self.geopotential,
            inclination,
            right_ascension,
            eccentricity,
            argument_of_perigee,
            mean_anomaly,
            kozai_mean_motion,
        )
        .expect("Failed to load orbit values");
        Constants::new(
            self.geopotential,
            sgp4::iau_epoch_to_sidereal_time,
            epoch,
            b_star,
            orbit_0,
        )
        .expect("Failed to load orbit values")
    }

    #[inline(always)]
    fn try_get_pos_vel(&self, jds: f64) -> ([f64; 3], [f64; 3]) {
        // TODO: this does not yet implement the interpolation between two neighboring states
        // which is present in the cSPICE implementation.
        // This currently matches the cspice implementation to within about 20km, where the error
        // is less near the year 2000.

        // There is also an outstanding small time conversion issue.
        // I am somewhat certain that this conversion is incorrect in cSPICE itself.
        // Much of this error may be fixed by applying a small linear offset to time which
        // causes about a 3 second offset in 2024 vs a 0 second offset in 2000.
        // See #66 for more details.
        let times = self.get_times();
        let idx: usize = match times.binary_search_by(|probe| probe.total_cmp(&jds)) {
            Ok(c) => c,
            Err(c) => {
                if c == 0 {
                    c
                } else if c == times.len() || (jds - times[c - 1]).abs() < (jds - times[c]).abs() {
                    c - 1
                } else {
                    c
                }
            }
        };
        let epoch = times[idx];
        let record = self.get_record(idx);
        let prediction = record
            .propagate(MinutesSinceEpoch((jds - epoch) / 60.0))
            .unwrap();

        let [x, y, z] = prediction.position;
        let [vx, vy, vz] = prediction.velocity;
        let v_scale = 86400.0 / AU_KM;
        (
            [x / AU_KM, y / AU_KM, z / AU_KM],
            [vx * v_scale, vy * v_scale, vz * v_scale],
        )
    }
}

impl TryFrom<SpkArray> for SpkSegmentType10 {
    type Error = Error;
    fn try_from(array: SpkArray) -> KeteResult<Self> {
        let array: GenericSegment = array.try_into()?;
        let constants = array.constants();
        let geopotential = Geopotential {
            j2: constants[0],
            j3: constants[1],
            j4: constants[2],
            ke: constants[3],
            ae: constants[6],
        };

        Ok(Self {
            array,
            geopotential,
        })
    }
}

// TODO: SPK Segment type 12 should be a minor variation on type 13. This was not
// implemented here due to missing a valid SPK file to test against.

/// Hermite Interpolation (Uneven Time Steps)
///
/// This uses a collection of individual positions/velocities and interpolates between
/// them using hermite interpolation.
/// <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/spk.html#Type%2013:%20Hermite%20Interpolation%20---%20Unequal%20Time%20Steps>
#[derive(Debug)]
pub(in crate::spice) struct SpkSegmentType13 {
    array: SpkArray,
    window_size: usize,
    n_records: usize,
}

impl From<SpkArray> for SpkSegmentType13 {
    fn from(array: SpkArray) -> Self {
        let n_records = array.daf[array.daf.len() - 1] as usize;
        let mut window_size = array.daf[array.daf.len() - 2] as usize;

        if window_size > n_records {
            eprintln!(
                "Spk Segment Type 13 must have at least as many records as the window size, n_records={n_records}, window_size={window_size}",
            );
            window_size = n_records;
        }

        Self {
            array,
            window_size,
            n_records,
        }
    }
}

/// Type 13 Record View
/// A view into a record of type 13, provided mainly for clarity to the underlying
/// data structure.
struct Type13RecordView<'a> {
    pos: &'a [f64; 3],
    vel: &'a [f64; 3],
}

impl SpkSegmentType13 {
    #[inline(always)]
    fn get_record(&self, idx: usize) -> Type13RecordView<'_> {
        unsafe {
            let rec = self.array.daf.data.get_unchecked(idx * 6..(idx + 1) * 6);
            Type13RecordView {
                pos: rec[0..3].try_into().unwrap(),
                vel: rec[3..6].try_into().unwrap(),
            }
        }
    }

    #[inline(always)]
    fn get_times(&self) -> &[f64] {
        unsafe {
            self.array
                .daf
                .data
                .get_unchecked(self.n_records * 6..self.n_records * 7)
        }
    }

    #[inline(always)]
    fn try_get_pos_vel(&self, jds: f64) -> ([f64; 3], [f64; 3]) {
        let times = self.get_times();
        let start_idx: isize = match times.binary_search_by(|probe| probe.total_cmp(&jds)) {
            Ok(c) => c as isize - (self.window_size as isize) / 2,
            Err(c) => {
                if (jds - times[c - 1]).abs() < (jds - times[c]).abs() {
                    c as isize - 1 - self.window_size as isize / 2
                } else {
                    c as isize - self.window_size as isize / 2
                }
            }
        };
        let start_idx =
            start_idx.clamp(0, self.n_records as isize - self.window_size as isize) as usize;

        let mut pos = [0.0; 3];
        let mut vel = [0.0; 3];
        for idx in 0..3 {
            let p: Box<[f64]> = (0..self.window_size)
                .map(|i| self.get_record(i + start_idx).pos[idx])
                .collect();
            let dp: Box<[f64]> = (0..self.window_size)
                .map(|i| self.get_record(i + start_idx).vel[idx])
                .collect();
            let (p, v) = hermite_interpolation(
                &times[start_idx..start_idx + self.window_size],
                &p,
                &dp,
                jds,
            );
            pos[idx] = p / AU_KM;
            vel[idx] = v / AU_KM * 86400.;
        }

        (pos, vel)
    }
}

/// Type 18 Record
///
/// This is actually 2 types in 1, under the stated goal of reducing the number
/// of unique SPICE kernel types.
///
/// Subtype 0 is a Hermite Interpolation of both position and velocity, a record
/// contains 12 numbers, 3 position, 3 derivative of position, 3 velocity, and 3
/// derivative of velocity. Note that it explicitly allows that the 3 velocity
/// values do not have to match the derivative of the position vectors.
/// Subtype 1 is a Lagrange Interpolation, a record of which contains 6 values,
/// 3 position, and 3 velocity.
///
/// <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/req/spk.html#Type%2018:%20ESOC/DDID%20Hermite/Lagrange%20Interpolation>
#[derive(Debug)]
pub(in crate::spice) struct SpkSegmentType18 {
    array: SpkArray,
    subtype: usize,
    window_size: usize,
    n_records: usize,
    record_size: usize,
}

impl TryFrom<SpkArray> for SpkSegmentType18 {
    type Error = Error;
    fn try_from(array: SpkArray) -> KeteResult<Self> {
        let n_records = array.daf[array.daf.len() - 1] as usize;
        let mut window_size = array.daf[array.daf.len() - 2] as usize;
        let subtype = array.daf[array.daf.len() - 3] as usize;
        let record_size = {
            if subtype == 0 {
                12
            } else if subtype == 1 {
                6
            } else {
                return Err(Error::ValueError(
                    "Spk Segment Type 12 only supports subtype of 0 or 1".into(),
                ));
            }
        };
        if window_size > n_records {
            eprintln!(
                "Spk Segment Type 18 must have at least as many records as the window size, n_records={n_records}, window_size={window_size}",
            );
            window_size = n_records;
        }

        Ok(Self {
            array,
            subtype,
            window_size,
            n_records,
            record_size,
        })
    }
}

/// Type 18 Record View
/// A view into a record of type 18, provided mainly for clarity to the underlying
/// data structure.
struct Type18RecordView<'a> {
    pos: &'a [f64],
    vel: &'a [f64],
}

impl SpkSegmentType18 {
    #[inline(always)]
    fn get_record(&self, idx: usize) -> Type18RecordView<'_> {
        unsafe {
            let rec = self
                .array
                .daf
                .data
                .get_unchecked(idx * self.record_size..(idx + 1) * self.record_size);
            Type18RecordView {
                pos: &rec[0..self.record_size / 2],
                vel: &rec[self.record_size / 2..self.record_size],
            }
        }
    }

    #[inline(always)]
    fn get_times(&self) -> &[f64] {
        unsafe {
            self.array.daf.data.get_unchecked(
                self.n_records * self.record_size..self.n_records * (self.record_size + 1),
            )
        }
    }

    #[inline(always)]
    fn try_get_pos_vel(&self, jds: f64) -> ([f64; 3], [f64; 3]) {
        let times = self.get_times();
        let start_idx: isize = match times.binary_search_by(|probe| probe.total_cmp(&jds)) {
            Ok(c) => c as isize - (self.window_size as isize) / 2,
            Err(c) => {
                if (jds - times[c - 1]).abs() < (jds - times[c]).abs() {
                    c as isize - 1 - self.window_size as isize / 2
                } else {
                    c as isize - self.window_size as isize / 2
                }
            }
        };
        let start_idx =
            start_idx.clamp(0, self.n_records as isize - self.window_size as isize) as usize;

        let mut pos = [0.0; 3];
        let mut vel = [0.0; 3];
        match self.subtype {
            0 => {
                for idx in 0..3 {
                    {
                        let p: Box<[f64]> = (0..self.window_size)
                            .map(|i| self.get_record(i + start_idx).pos[idx])
                            .collect();
                        let dp: Box<[f64]> = (0..self.window_size)
                            .map(|i| self.get_record(i + start_idx).pos[idx + 3])
                            .collect();
                        let (p, _) = hermite_interpolation(
                            &times[start_idx..start_idx + self.window_size],
                            &p,
                            &dp,
                            jds,
                        );
                        pos[idx] = p / AU_KM;
                    }
                    {
                        let p: Box<[f64]> = (0..self.window_size)
                            .map(|i| self.get_record(i + start_idx).vel[idx])
                            .collect();
                        let dp: Box<[f64]> = (0..self.window_size)
                            .map(|i| self.get_record(i + start_idx).vel[idx + 3])
                            .collect();
                        let (v, _) = hermite_interpolation(
                            &times[start_idx..start_idx + self.window_size],
                            &p,
                            &dp,
                            jds,
                        );
                        vel[idx] = v / AU_KM * 86400.;
                    }
                }
            }
            1 => {
                for idx in 0..3 {
                    let mut p: Box<[f64]> = (0..self.window_size)
                        .map(|i| self.get_record(i + start_idx).pos[idx])
                        .collect();
                    let mut dp: Box<[f64]> = (0..self.window_size)
                        .map(|i| self.get_record(i + start_idx).vel[idx])
                        .collect();
                    let p = lagrange_interpolation(
                        &times[start_idx..start_idx + self.window_size],
                        &mut p,
                        jds,
                    );
                    let v = lagrange_interpolation(
                        &times[start_idx..start_idx + self.window_size],
                        &mut dp,
                        jds,
                    );
                    pos[idx] = p / AU_KM;
                    vel[idx] = v / AU_KM * 86400.;
                }
            }
            _ => {
                unreachable!()
            }
        }
        (pos, vel)
    }
}

/// Extended Modified Difference Arrays
///
/// <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/spk.html#Type%2021:%20Extended%20Modified%20Difference%20Arrays>
///
#[derive(Debug)]
pub(in crate::spice) struct SpkSegmentType21 {
    array: SpkArray,
    n_coef: usize,
    n_records: usize,
    record_len: usize,
}

impl From<SpkArray> for SpkSegmentType21 {
    fn from(array: SpkArray) -> Self {
        let n_records = array.daf[array.daf.len() - 1] as usize;
        let n_coef = array.daf[array.daf.len() - 2] as usize;

        let record_len = 4 * n_coef + 11;

        Self {
            array,
            n_coef,
            n_records,
            record_len,
        }
    }
}

impl SpkSegmentType21 {
    #[inline(always)]
    fn get_record(&self, idx: usize) -> &[f64] {
        unsafe {
            self.array
                .daf
                .data
                .get_unchecked(idx * self.record_len..(idx + 1) * self.record_len)
        }
    }

    #[inline(always)]
    fn get_times(&self) -> &[f64] {
        unsafe {
            self.array.daf.data.get_unchecked(
                self.n_records * self.record_len..self.n_records * (self.record_len + 1),
            )
        }
    }

    #[inline(always)]
    fn try_get_pos_vel(&self, jds: f64) -> KeteResult<([f64; 3], [f64; 3])> {
        // Records are laid out as so:
        //
        // Size      Description
        // ----------------------
        // 1          Reference Epoch for the difference line
        // n_coef     Step size function vector
        // 6          Reference state - x, vx, y, vy, z, vz  (interleaved order)
        // 3*n_coef   Modified divided difference arrays
        // 1          Maximum integration order plus 1
        // 3          Integration order array
        // total: 11 + 4*n_coef

        // we need to find the first record which has a time greater than or equal
        // to the target jd.

        let start_idx = self
            .get_times()
            .binary_search_by(|probe| probe.total_cmp(&jds))
            .unwrap_or_else(|c| c);

        let record = self.get_record(start_idx);

        let ref_time = record[0];

        let func_vec = &record[1..self.n_coef + 1];
        let ref_state = &record[self.n_coef + 1..self.n_coef + 7];

        let divided_diff_array = &record[self.n_coef + 7..4 * self.n_coef + 7];

        let kq_max1 = record[4 * self.n_coef + 7] as usize;
        let kq = &record[4 * self.n_coef + 8..4 * self.n_coef + 11];

        // in the spice code ref_time is in seconds from j2000
        let dt = jds - ref_time;

        let mut fc = Vec::<f64>::with_capacity(self.n_coef);
        let mut wc = Vec::<f64>::with_capacity(self.n_coef);

        let mut tp = dt;
        for f in func_vec.iter().take(kq_max1 - 2) {
            if *f == 0.0 {
                // don't divide by 0 below, file was built incorrectly.
                return Err(Error::IOError(
                    "SPK File contains segments of type 21 has invalid contents.".into(),
                ));
            }

            fc.push(tp / f);
            wc.push(dt / f);
            tp = dt + f;
        }

        let mut w: Box<[f64]> = (0..kq_max1).map(|x| (x as f64 + 1.0).recip()).collect();

        let mut ks = kq_max1 - 1;
        let mut jx = 0;
        let mut ks1 = ks - 1;

        while ks >= 2 {
            jx += 1;
            for j in 0..jx {
                w[j + ks] = fc[j] * w[j + ks1] - wc[j] * w[j + ks];
            }
            ks = ks1;
            ks1 -= 1;
        }

        // position interpolation
        let pos = std::array::from_fn(|idx| {
            let sum: f64 = (1..(kq[idx] as usize + 1))
                .rev()
                .map(|j| divided_diff_array[idx * self.n_coef + j - 1] * w[j + ks - 1])
                .sum();
            (ref_state[2 * idx] + dt * (sum * dt + ref_state[2 * idx + 1])) / AU_KM
        });

        // Recompute W for velocities
        for j in 0..jx {
            w[j + ks] = fc[j] * w[j + ks1] - wc[j] * w[j + ks];
        }
        ks -= 1;

        // velocity interpolation
        let vel = std::array::from_fn(|idx| {
            let sum: f64 = (1..(kq[idx] as usize + 1))
                .rev()
                .map(|j| divided_diff_array[idx * self.n_coef + j - 1] * w[j + ks - 1])
                .sum();
            (ref_state[2 * idx + 1] + dt * sum) / AU_KM * 86400.0
        });

        Ok((pos, vel))
    }
}

// This segment type has poor documentation on the NAIF website.
/// Segments of type 10 and 14 use a "generic segment" definition.
/// The DAF Array is big flat vector of floats.
#[derive(Debug)]
#[allow(dead_code, reason = "Some fields are not used in this segment type")]
struct GenericSegment {
    /// Underlying Spk array
    array: SpkArray,

    /// Number of metadata value stored in this segment.
    n_meta: usize,

    // Below meta data is guaranteed to exist.
    /// address of the constant values
    const_addr: usize,

    /// Number of constants
    n_consts: usize,

    /// Address of reference directory
    ref_dir_addr: usize,

    /// Number of reference directory items
    n_item_ref_dir: usize,

    /// Type of reference directory
    ref_dir_type: usize,

    /// Address of reference items
    ref_items_addr: usize,

    /// Number of reference items
    n_ref_items: usize,

    /// Address of the data packets
    packet_dir_addr: usize,

    /// Number of data packets
    n_dir_packets: usize,

    /// Packet directory type
    packet_dir_dype: usize,

    /// Packet address
    packet_addr: usize,

    /// Number of data packets
    n_packets: usize,

    /// Address of reserved area
    res_addr: usize,

    /// number of entries in reserved area.
    n_reserved: usize,
}

impl GenericSegment {
    fn constants(&self) -> &[f64] {
        unsafe {
            self.array
                .daf
                .data
                .get_unchecked(self.const_addr..self.const_addr + self.n_consts)
        }
    }

    /// Slice into the entire reference items array.
    fn get_reference_items(&self) -> &[f64] {
        unsafe {
            self.array
                .daf
                .data
                .get_unchecked(self.ref_items_addr..self.ref_items_addr + self.n_ref_items)
        }
    }

    fn get_packet<const T: usize>(&self, idx: usize) -> &[f64; T] {
        unsafe {
            self.array
                .daf
                .data
                .get_unchecked(self.packet_addr + T * idx..self.packet_addr + T * (idx + 1))
                .try_into()
                .unwrap()
        }
    }
}

impl TryFrom<SpkArray> for GenericSegment {
    type Error = Error;

    fn try_from(array: SpkArray) -> KeteResult<Self> {
        // The very last value of this array is an int (cast to f64) which indicates the number
        // of meta-data values.

        let n_meta = array.daf[array.daf.len() - 1] as usize;

        if n_meta < 15 {
            Err(Error::IOError(
                "PSK File not correctly formatted. There are fewer values found than expected."
                    .into(),
            ))?;
        }
        // there are guaranteed to be 15 meta data values.
        let (
            const_addr,
            n_consts,
            ref_dir_addr,
            n_item_ref_dir,
            ref_dir_type,
            ref_items_addr,
            n_ref_items,
            packet_dir_addr,
            n_dir_packets,
            packet_dir_dype,
            packet_addr,
            n_packets,
        ) = array
            .daf
            .data
            .get(array.daf.len() - n_meta..array.daf.len() - 1)
            .unwrap()
            .iter()
            .map(|x| *x as usize)
            .next_tuple()
            .unwrap();

        let (res_addr, n_reserved) = array
            .daf
            .data
            .get(array.daf.len() - n_meta..array.daf.len() - 1)
            .unwrap()
            .iter()
            .map(|x| *x as usize)
            .next_tuple()
            .unwrap();

        Ok(Self {
            array,
            n_meta,
            const_addr,
            n_consts,
            ref_dir_addr,
            n_item_ref_dir,
            ref_dir_type,
            ref_items_addr,
            n_ref_items,
            packet_dir_addr,
            n_dir_packets,
            packet_dir_dype,
            packet_addr,
            n_packets,
            res_addr,
            n_reserved,
        })
    }
}
