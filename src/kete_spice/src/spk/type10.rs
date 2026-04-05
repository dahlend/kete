//! SPK Segment Type 10 - Space Command Two-Line Elements.
//!
//! <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/spk.html#Type%2010:%20Space%20Command%20Two-Line%20Elements>

use super::SpkArray;
use crate::spice_jd_to_jd;
use itertools::Itertools;
use kete_core::constants::AU_KM;
use kete_core::errors::Error;
use kete_core::prelude::KeteResult;
use sgp4::{
    Constants, Geopotential, MinutesSinceEpoch, Orbit,
    julian_years_since_j2000_afspc_compatibility_mode,
};

/// Space Command two-line elements
///
/// <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/spk.html#Type%2010:%20Space%20Command%20Two-Line%20Elements>
///
#[derive(Debug)]
pub struct SpkSegmentType10 {
    /// Generic Segments are a collection of a few different directories:
    /// `Packets` are where Type 10 stores the TLE values.
    /// `Packet Directory` is unused.
    /// `Reference Directory` is where the 100 step JDs are stored.
    /// `Reference Items` is a list of all JDs
    pub(in crate::spk) array: GenericSegment,

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

        let epoch = julian_years_since_j2000_afspc_compatibility_mode(
            &spice_jd_to_jd(epoch)
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
            sgp4::afspc_epoch_to_sidereal_time,
            epoch,
            b_star,
            orbit_0,
        )
        .expect("Failed to load orbit values")
    }

    /// Build the data array for a Type 10 (TLE) generic segment.
    ///
    /// The data layout follows the cSPICE generic segment format for
    /// explicitly-indexed fixed-size packets:
    /// `[constants][interleaved (ref + packet)...][contiguous refs][ref_dir][metadata]`
    ///
    /// Nutation values (packet indices 10-13) are set to zero; the kete reader
    /// does not use them.
    #[allow(dead_code, reason = "Writer for external use, no internal callers yet")]
    fn build_data(consts: &[f64; 8], elements: &[f64], epochs: &[f64]) -> KeteResult<Vec<f64>> {
        let n = epochs.len();
        if n == 0 {
            return Err(Error::ValueError(
                "Type 10: need at least one element set.".into(),
            ));
        }
        if elements.len() != n * 10 {
            return Err(Error::ValueError(format!(
                "Type 10: elements length ({}) must be n ({}) * 10",
                elements.len(),
                n
            )));
        }
        for w in epochs.windows(2) {
            if w[1] <= w[0] {
                return Err(Error::ValueError(
                    "Type 10: epochs must be strictly increasing.".into(),
                ));
            }
        }

        let n_ref_dir = if n > 100 { (n - 1) / 100 } else { 0 };
        let ref_items_addr = 8 + 15 * n;
        let ref_dir_addr = ref_items_addr + n + n_ref_dir;
        let total_len = 8 + 15 * n + n + n_ref_dir + 17;
        let mut data = Vec::with_capacity(total_len);

        // 8 geophysical constants
        data.extend_from_slice(consts);

        // Interleaved ref + packet slots (15 values each)
        for i in 0..n {
            data.push(epochs[i]); // reference value (epoch)
            data.extend_from_slice(&elements[i * 10..(i + 1) * 10]); // 10 TLE elements
            data.extend_from_slice(&[0.0; 4]); // 4 nutation values (zero)
        }

        // Contiguous reference items
        data.extend_from_slice(epochs);

        // Reference directory (every 100th epoch)
        for i in 1..=n_ref_dir {
            data.push(epochs[(i * 100 - 1).min(n - 1)]);
        }

        // 17 generic segment metadata values
        data.push(0.0); //  [0] const_addr
        data.push(8.0); //  [1] n_consts
        data.push(ref_dir_addr as f64); //  [2] ref_dir_addr
        data.push(n_ref_dir as f64); //  [3] n_item_ref_dir
        data.push(4.0); //  [4] ref_dir_type (explicit index)
        data.push(ref_items_addr as f64); //  [5] ref_items_addr
        data.push(n as f64); //  [6] n_ref_items
        data.push(0.0); //  [7] packet_dir_addr (none)
        data.push(0.0); //  [8] n_dir_packets  (none)
        data.push(0.0); //  [9] packet_dir_type (none)
        data.push(8.0); // [10] packet_addr
        data.push(n as f64); // [11] n_packets
        data.push(0.0); // [12] res_addr
        data.push(0.0); // [13] n_reserved
        data.push(14.0); // [14] max_packet_size
        data.push(1.0); // [15] offset (1 ref value per packet)
        data.push(17.0); // [16] n_meta

        debug_assert_eq!(data.len(), total_len, "Type 10 data length mismatch");
        Ok(data)
    }

    /// Create a Type 10 (TLE) SPK array.
    ///
    /// # Arguments
    /// * `object_id`    - NAIF body ID.
    /// * `center_id`    - NAIF center body ID (typically 399 for Earth).
    /// * `frame_id`     - NAIF reference frame ID.
    /// * `consts`       - 8 geophysical constants: `[j2, j3, j4, ke, qo, so, re, ae]`.
    /// * `elements`     - Flat TLE elements, 10 values per set in order:
    ///   `[ndt2o, ndd6o, bstar, incl, node0, ecc, omega, m0, n0, epoch]`.
    ///   Angles in radians, rates in radians/minute, epoch in seconds past J2000.
    /// * `epochs`       - n reference epochs (seconds past J2000), must equal element
    ///   epochs and be strictly increasing.
    /// * `segment_name` - Name for the DAF segment (max 40 chars).
    ///
    /// # Errors
    /// Returns an error if inputs are empty, dimensions mismatch, or epochs aren't
    /// strictly increasing.
    #[allow(
        clippy::missing_panics_doc,
        reason = "build_data validates non-empty slices before the unwrap is reached"
    )]
    pub fn new_array(
        object_id: i32,
        center_id: i32,
        frame_id: i32,
        consts: &[f64; 8],
        elements: &[f64],
        epochs: &[f64],
        segment_name: &str,
    ) -> KeteResult<SpkArray> {
        let data = Self::build_data(consts, elements, epochs)?;
        let jd_start = spice_jd_to_jd(epochs[0]);
        let jd_end = spice_jd_to_jd(*epochs.last().unwrap());
        Ok(SpkArray::new(
            object_id,
            center_id,
            frame_id,
            10,
            jd_start,
            jd_end,
            data,
            segment_name.to_string(),
        ))
    }

    #[inline(always)]
    pub(in crate::spk) fn try_get_pos_vel(&self, jds: f64) -> ([f64; 3], [f64; 3]) {
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

// ---------------------------------------------------------------------------
// Generic Segment (shared DAF structure used by Type 10 and 14)
// ---------------------------------------------------------------------------

// This segment type has poor documentation on the NAIF website.
/// Segments of type 10 and 14 use a "generic segment" definition.
/// The DAF Array is big flat vector of floats.
#[derive(Debug)]
#[allow(dead_code, reason = "Some fields are not used in this segment type")]
pub(in crate::spk) struct GenericSegment {
    /// Underlying Spk array
    pub(in crate::spk) array: SpkArray,

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
    pub(in crate::spk) fn constants(&self) -> &[f64] {
        unsafe {
            self.array
                .daf
                .data
                .get_unchecked(self.const_addr..self.const_addr + self.n_consts)
        }
    }

    /// Slice into the entire reference items array.
    pub(in crate::spk) fn get_reference_items(&self) -> &[f64] {
        unsafe {
            self.array
                .daf
                .data
                .get_unchecked(self.ref_items_addr..self.ref_items_addr + self.n_ref_items)
        }
    }

    pub(in crate::spk) fn get_packet<const T: usize>(&self, idx: usize) -> &[f64; T] {
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

    #[allow(
        clippy::cast_sign_loss,
        reason = "This is correct as long as the file is correct."
    )]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type10_round_trip() {
        let consts: [f64; 8] = [
            1.082_616e-3,
            -2.538_81e-6,
            -1.655_97e-6,
            7.436_691_61e-2,
            120.0,
            78.0,
            6378.135,
            1.0,
        ];
        let epoch1: f64 = 0.0; // J2000
        let epoch2: f64 = 86400.0; // J2000 + 1 day

        // Two element sets: [ndt2o, ndd6o, bstar, incl, node0, ecc, omega, m0, n0, epoch]
        let elements = vec![
            0.0, 0.0, 1e-4, 0.9, 1.5, 0.001, 2.0, 0.5, 0.06, epoch1, 0.0, 0.0, 1e-4, 0.9, 1.5,
            0.001, 2.0, 0.5, 0.06, epoch2,
        ];
        let epochs = vec![epoch1, epoch2];

        let array =
            SpkSegmentType10::new_array(-25544, 399, 1, &consts, &elements, &epochs, "test")
                .unwrap();

        // Round-trip through GenericSegment -> SpkSegmentType10
        let seg: SpkSegmentType10 = array.try_into().unwrap();

        // Reference items (epochs) round-trip correctly
        let times = seg.get_times();
        assert_eq!(times.len(), 2);
        assert_eq!(times[0], epoch1);
        assert_eq!(times[1], epoch2);

        // Geophysical constants round-trip correctly
        let c = seg.array.constants();
        assert_eq!(c.len(), 8);
        assert_eq!(c[0], consts[0]);
        assert_eq!(c[3], consts[3]);
        assert_eq!(c[6], consts[6]);

        // Packet data round-trip correctly
        let rec = seg.array.get_packet::<15>(0);
        assert_eq!(rec[0], epoch1); // ref epoch
        assert_eq!(rec[3], 1e-4); // bstar
        assert_eq!(rec[10], epoch1); // tle epoch
        assert_eq!(rec[11], 0.0); // nutation (zero)

        let rec1 = seg.array.get_packet::<15>(1);
        assert_eq!(rec1[0], epoch2);
        assert_eq!(rec1[10], epoch2);
    }

    #[test]
    fn type10_validation() {
        let consts = [0.0; 8];
        // Empty elements
        assert!(SpkSegmentType10::new_array(1, 399, 1, &consts, &[], &[], "t").is_err());
        // Mismatched lengths
        assert!(
            SpkSegmentType10::new_array(1, 399, 1, &consts, &[0.0; 10], &[0.0, 1.0], "t").is_err()
        );
        // Non-increasing epochs
        assert!(
            SpkSegmentType10::new_array(1, 399, 1, &consts, &[0.0; 20], &[1.0, 0.0], "t").is_err()
        );
    }
}
