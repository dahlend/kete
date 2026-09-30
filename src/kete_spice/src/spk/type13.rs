//! SPK Segment Type 13 - Hermite Interpolation (Unequal Time Steps).
//!
//! <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/spk.html#Type%2013:%20Hermite%20Interpolation%20---%20Unequal%20Time%20Steps>

use super::SpkArray;
use super::type9::{read_control, window_start};
use crate::interpolation::hermite_interpolation;
use crate::jd_to_spice_jd;
use kete_core::constants::AU_KM;
use kete_core::errors::Error;
use kete_core::frames::InertialFrame;
use kete_core::prelude::{Desig, KeteResult, State};

/// Hermite Interpolation (Uneven Time Steps)
///
/// This uses a collection of individual positions/velocities and interpolates between
/// them using hermite interpolation.
/// <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/spk.html#Type%2013:%20Hermite%20Interpolation%20---%20Unequal%20Time%20Steps>
#[derive(Debug)]
pub struct SpkSegmentType13 {
    pub(crate) array: SpkArray,
    window_size: usize,
    n_records: usize,
}

impl SpkSegmentType13 {
    /// Create a Type 13 (Hermite interpolation, unequal time steps) SPK array.
    ///
    /// `object_id` is the NAIF ID of the body, and `center_id` is the NAIF ID
    /// of the center body. `frame_id` is the NAIF frame ID. Each entry of
    /// `states` is `(epoch, [x, y, z], [vx, vy, vz])`. The epoch is in TDB
    /// seconds from J2000, the position is in km, and the velocity is in km/s.
    /// The segment covers the first epoch to the last epoch. `degree` is the
    /// Hermite polynomial degree. The window holds `(degree + 1) / 2` states.
    /// `segment_name` is the name stored in the DAF name record, which holds at
    /// most 40 characters.
    ///
    /// # Errors
    /// Returns [`Error::ValueError`] in these cases:
    /// - `states` is empty.
    /// - `degree` is even or outside `[1, 27]`.
    /// - `states` holds fewer than `(degree + 1) / 2` entries.
    /// - The epochs are not strictly increasing.
    pub fn new_array(
        object_id: i32,
        center_id: i32,
        frame_id: i32,
        states: &[(f64, [f64; 3], [f64; 3])],
        degree: u32,
        segment_name: &str,
    ) -> KeteResult<SpkArray> {
        let (Some(first), Some(last)) = (states.first(), states.last()) else {
            return Err(Error::ValueError(
                "Type 13: need at least one state.".into(),
            ));
        };
        let (jds_start, jds_end) = (first.0, last.0);

        let n = states.len();
        if !(1..=27).contains(&degree) || degree.is_multiple_of(2) {
            return Err(Error::ValueError(
                "Type 13: degree must be odd and in [1, 27].".into(),
            ));
        }
        let winsiz = degree.div_ceil(2);
        if n < winsiz as usize {
            return Err(Error::ValueError(
                "Type 13: need at least (degree+1)/2 states.".into(),
            ));
        }
        for w in states.windows(2) {
            if w[1].0 <= w[0].0 {
                return Err(Error::ValueError(
                    "Type 13: epochs must be strictly increasing.".into(),
                ));
            }
        }

        // Layout: [6*n states] [n epochs] [directory] [winsiz-1] [n]
        let n_dir = if n > 100 { (n - 1) / 100 } else { 0 };
        let mut data = Vec::with_capacity(7 * n + n_dir + 2);
        for &(_, pos, vel) in states {
            data.extend_from_slice(&pos);
            data.extend_from_slice(&vel);
        }
        for &(epoch, _, _) in states {
            data.push(epoch);
        }
        for i in 1..=n_dir {
            data.push(states[i * 100 - 1].0);
        }
        // CRITICAL: store winsiz - 1, NOT degree
        data.push(f64::from(winsiz - 1));
        data.push(n as f64);

        Ok(SpkArray::new(
            object_id,
            center_id,
            frame_id,
            13,
            jds_start,
            jds_end,
            data,
            segment_name.to_string(),
        ))
    }

    /// Create a Type 13 SPK array from [`State`] objects.
    ///
    /// The function converts epochs to TDB seconds from J2000, positions from
    /// AU to km, and velocities from AU/day to km/s. It takes the object ID and
    /// the center ID from the first state. [`Self::new_array`] describes
    /// `frame_id`, `degree`, and `segment_name`.
    ///
    /// # Errors
    /// Returns [`Error::ValueError`] in these cases:
    /// - `states` is empty.
    /// - The designation of the first state is not a NAIF integer ID.
    /// - `degree` is even or outside `[1, 27]`.
    /// - `states` holds fewer than `(degree + 1) / 2` entries.
    /// - The epochs are not strictly increasing.
    pub fn from_states<T: InertialFrame>(
        states: &[State<T>],
        frame_id: i32,
        degree: u32,
        segment_name: &str,
    ) -> KeteResult<SpkArray> {
        let Some(first) = states.first() else {
            return Err(Error::ValueError(
                "Type 13: need at least one state.".into(),
            ));
        };
        #[allow(
            clippy::wildcard_enum_match_arm,
            reason = "Only NAIF IDs are valid here."
        )]
        let object_id = match &first.desig {
            Desig::Naif(id) => *id,
            _ => {
                return Err(Error::ValueError(
                    "Type 13: states must have NAIF integer designations.".into(),
                ));
            }
        };
        let center_id = first.center_id();
        let raw_states: Vec<(f64, [f64; 3], [f64; 3])> = states
            .iter()
            .map(|s| {
                let pos: [f64; 3] = s.pos.into();
                let vel: [f64; 3] = s.vel.into();
                (
                    jd_to_spice_jd(s.epoch),
                    [pos[0] * AU_KM, pos[1] * AU_KM, pos[2] * AU_KM],
                    [
                        vel[0] * AU_KM / 86400.0,
                        vel[1] * AU_KM / 86400.0,
                        vel[2] * AU_KM / 86400.0,
                    ],
                )
            })
            .collect();
        Self::new_array(
            object_id,
            center_id,
            frame_id,
            &raw_states,
            degree,
            segment_name,
        )
    }

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
    pub(crate) fn try_get_pos_vel(&self, jds: f64) -> ([f64; 3], [f64; 3]) {
        let times = self.get_times();
        let start_idx = window_start(times, jds, self.window_size);

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

/// Type 13 Record View
/// A view into a record of type 13, provided mainly for clarity to the underlying
/// data structure.
struct Type13RecordView<'a> {
    pos: &'a [f64; 3],
    vel: &'a [f64; 3],
}

impl TryFrom<SpkArray> for SpkSegmentType13 {
    type Error = Error;

    fn try_from(array: SpkArray) -> KeteResult<Self> {
        // CSPICE stores (winsiz - 1) at data[len-2], where winsiz = (degree+1)/2.
        // The CSPICE reader (spkr09.c) adds 1 to recover the true window size.
        let (stored, n_records) = read_control(&array, "13")?;
        let window_size = stored + 1;

        if window_size > n_records {
            return Err(Error::IOError(format!(
                "SPK Type 13: window size ({window_size}) exceeds number of records ({n_records})"
            )));
        }

        Ok(Self {
            array,
            window_size,
            n_records,
        })
    }
}
