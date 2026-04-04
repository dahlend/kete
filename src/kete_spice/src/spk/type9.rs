//! SPK Segment Type 9 - Lagrange Interpolation (Unequal Time Steps).
//!
//! <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/req/spk.html#Type%209:%20Lagrange%20Interpolation%20---%20Unequal%20Time%20Steps>

use super::SpkArray;
use crate::interpolation::lagrange_interpolation;
use crate::jd_to_spice_jd;
use kete_core::constants::AU_KM;
use kete_core::errors::Error;
use kete_core::frames::InertialFrame;
use kete_core::prelude::{Desig, KeteResult, State};
use kete_core::time::{TDB, Time};

/// Lagrange Interpolation (Uneven Time Steps)
///
/// This uses a collection of individual positions/velocities and interpolates between
/// them using Lagrange interpolation.
/// <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/req/spk.html#Type%209:%20Lagrange%20Interpolation%20---%20Unequal%20Time%20Steps>
#[derive(Debug)]
pub struct SpkSegmentType9 {
    pub(crate) array: SpkArray,
    poly_degree: usize,
    n_records: usize,
}

impl SpkSegmentType9 {
    /// Create a Type 9 (Lagrange interpolation, unequal time steps) SPK array.
    ///
    /// # Arguments
    /// * `object_id`    - NAIF ID of the body.
    /// * `center_id`    - NAIF ID of the center body.
    /// * `frame_id`     - NAIF frame ID.
    /// * `states`       - `(epoch, [x,y,z] km, [vx,vy,vz] km/s)`, strictly increasing epochs.
    /// * `degree`       - Lagrange polynomial degree, in `[1, 27]`.
    /// * `segment_name` - Name stored in the DAF name record (max 40 chars).
    ///
    /// # Panics
    /// Panics if `states` is empty.
    ///
    /// # Errors
    /// Returns an error if `degree` is outside `[1, 27]`.
    pub fn new_array(
        object_id: i32,
        center_id: i32,
        frame_id: i32,
        states: &[(Time<TDB>, [f64; 3], [f64; 3])],
        degree: u32,
        segment_name: &str,
    ) -> KeteResult<SpkArray> {
        let jd_start = states.first().unwrap().0;
        let jd_end = states.last().unwrap().0;

        let states: Vec<(f64, [f64; 3], [f64; 3])> = states
            .iter()
            .map(|&(t, p, v)| (jd_to_spice_jd(t), p, v))
            .collect();

        let n = states.len();
        if !(1..=27).contains(&degree) {
            return Err(Error::ValueError(
                "Type 9: degree must be in [1, 27].".into(),
            ));
        }
        if n <= degree as usize {
            return Err(Error::ValueError(
                "Type 9: need at least degree + 1 states.".into(),
            ));
        }
        for w in states.windows(2) {
            if w[1].0 <= w[0].0 {
                return Err(Error::ValueError(
                    "Type 9: epochs must be strictly increasing.".into(),
                ));
            }
        }

        // Layout: [6*n states] [n epochs] [directory] [degree] [n]
        let n_dir = if n > 100 { (n - 1) / 100 } else { 0 };
        let mut data = Vec::with_capacity(7 * n + n_dir + 2);
        for &(_, pos, vel) in &states {
            data.extend_from_slice(&pos);
            data.extend_from_slice(&vel);
        }
        for &(epoch, _, _) in &states {
            data.push(epoch);
        }
        for i in 1..=n_dir {
            data.push(states[i * 100 - 1].0);
        }
        data.push(f64::from(degree));
        data.push(n as f64);

        Ok(SpkArray::new(
            object_id,
            center_id,
            frame_id,
            9,
            jd_start,
            jd_end,
            data,
            segment_name.to_string(),
        ))
    }

    /// Create a Type 9 SPK array from [`State`] objects.
    ///
    /// Positions and velocities are converted from AU / AU/day to km / km/s.
    /// The `object_id` and `center_id` are taken from the first state.
    ///
    /// # Panics
    /// Panics if `states` is empty.
    ///
    /// # Errors
    /// Returns an error if `degree` is outside `[1, 27]` or the first state's
    /// designation is not a NAIF integer ID.
    pub fn from_states<T: InertialFrame>(
        states: &[State<T>],
        frame_id: i32,
        degree: u32,
        segment_name: &str,
    ) -> KeteResult<SpkArray> {
        let first = states.first().unwrap();
        #[allow(
            clippy::wildcard_enum_match_arm,
            reason = "Only NAIF IDs are valid here."
        )]
        let object_id = match &first.desig {
            Desig::Naif(id) => *id,
            _ => {
                return Err(Error::ValueError(
                    "Type 9: states must have NAIF integer designations.".into(),
                ));
            }
        };
        let center_id = first.center_id;
        let raw_states: Vec<(Time<TDB>, [f64; 3], [f64; 3])> = states
            .iter()
            .map(|s| {
                let pos: [f64; 3] = s.pos.into();
                let vel: [f64; 3] = s.vel.into();
                (
                    s.epoch,
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
    #[allow(
        clippy::cast_possible_wrap,
        reason = "This is correct as long as the file is correct."
    )]
    pub(crate) fn try_get_pos_vel(&self, jds: f64) -> ([f64; 3], [f64; 3]) {
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

        #[allow(
            clippy::cast_sign_loss,
            reason = "This is correct as long as the file is correct."
        )]
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

/// Type 9 Record View
/// A view into a record of type 9, provided mainly for clarity to the underlying
/// data structure.
struct Type9RecordView<'a> {
    pos: &'a [f64; 3],
    vel: &'a [f64; 3],
}

impl TryFrom<SpkArray> for SpkSegmentType9 {
    type Error = Error;

    #[allow(
        clippy::cast_sign_loss,
        reason = "This is correct as long as the file is correct."
    )]
    fn try_from(array: SpkArray) -> KeteResult<Self> {
        let n_records = array.daf[array.daf.len() - 1] as usize;
        let poly_degree = array.daf[array.daf.len() - 2] as usize;

        if poly_degree + 1 > n_records {
            return Err(Error::IOError(format!(
                "SPK Type 9: polynomial degree ({poly_degree}) requires at least {} records, but only {n_records} present",
                poly_degree + 1
            )));
        }

        Ok(Self {
            array,
            poly_degree,
            n_records,
        })
    }
}
