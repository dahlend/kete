//! SPK Segment Type 10 - Space Command Two-Line Elements.
//!
//! <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/spk.html#Type%2010:%20Space%20Command%20Two-Line%20Elements>

use super::SpkArray;
use kete_core::constants::AU_KM;
use kete_core::errors::Error;
use kete_core::frames::teme_frame;
use kete_core::prelude::KeteResult;
use kete_core::time::{TDB, Time, UTC};
use nalgebra::Vector3;
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
    /// Create a Type 10 (TLE) SPK array.
    ///
    /// `object_id` is the NAIF ID of the body, and `center_id` is the NAIF ID
    /// of the center, usually 399 for Earth. `frame_id` is the NAIF ID of the
    /// reference frame. `consts` holds the 8 geophysical constants
    /// `[j2, j3, j4, ke, qo, so, re, ae]`.
    ///
    /// `elements` holds 10 values per element set, in the order
    /// `[ndt2o, ndd6o, bstar, incl, node0, ecc, omega, m0, n0, epoch]`. Angles
    /// are in radians, rates are in radians per minute, and the epoch is in TDB
    /// seconds from J2000. `epochs` holds one reference epoch per element set,
    /// in TDB seconds from J2000. Each reference epoch must equal the epoch of
    /// its element set, and the epochs must be strictly increasing.
    ///
    /// `first` and `last` are the coverage of the segment, in TDB seconds from
    /// J2000. The caller chooses them. The coverage can extend past the first
    /// and last element sets. The reader propagates the nearest element set to
    /// those times. `segment_name` is the name of the DAF segment, which holds
    /// at most 40 characters.
    ///
    /// # Errors
    /// Returns [`Error::ValueError`] if:
    /// - `epochs` is empty.
    /// - `elements` does not hold 10 values per epoch.
    /// - `epochs` is not strictly increasing.
    /// - A reference epoch differs from the epoch of its element set.
    /// - `first` or `last` is NaN, or `first` is after `last`.
    pub fn new_array(
        object_id: i32,
        center_id: i32,
        frame_id: i32,
        consts: &[f64; 8],
        elements: &[f64],
        epochs: &[f64],
        first: f64,
        last: f64,
        segment_name: &str,
    ) -> KeteResult<SpkArray> {
        let data = Self::build_data(consts, elements, epochs)?;
        if first.is_nan() || last.is_nan() || first > last {
            return Err(Error::ValueError(format!(
                "Type 10: coverage start {first} is after its end {last}."
            )));
        }
        Ok(SpkArray::new(
            object_id,
            center_id,
            frame_id,
            10,
            first,
            last,
            data,
            segment_name.to_string(),
        ))
    }

    /// Create a Type 10 SPK segment from the TLE element sets of one object.
    ///
    /// All element sets in `elements` must belong to the same satellite. The
    /// function sorts them by epoch before it writes them. `object_id`,
    /// `center_id`, `frame_id`, and `segment_name` are as in
    /// [`Self::new_array`].
    ///
    /// The Type 10 format requires these unit conversions:
    /// - Angles convert from degrees to radians.
    /// - Mean motion converts from revolutions per day to radians per minute.
    /// - Epochs convert from UTC to TDB seconds from J2000.
    ///
    /// The segment uses the WGS72 geophysical constants, which are the
    /// constants of the TLE fits. The coverage of the segment is the span of
    /// the element set epochs, extended by `pad_days` before the first epoch
    /// and after the last epoch.
    ///
    /// # Errors
    /// Returns [`Error::ValueError`] if `elements` is empty, or if `pad_days`
    /// is NaN or negative. Also returns [`Error::ValueError`] if the TDB epochs
    /// are not strictly increasing after the sort, for example when two element
    /// sets have the same epoch.
    pub fn from_tle_records(
        elements: &[sgp4::Elements],
        object_id: i32,
        center_id: i32,
        frame_id: i32,
        pad_days: f64,
        segment_name: &str,
    ) -> KeteResult<SpkArray> {
        // Unit conversion constants - placed before any let statements to
        // satisfy clippy::items_after_statements.
        const DEG2RAD: f64 = std::f64::consts::PI / 180.0;
        // rev/day  -> rad/min   = 2*pi / 1440
        const MM_TO_RAD_PER_MIN: f64 = std::f64::consts::PI / 720.0;
        // rev/day^2 -> rad/min^2  = 2*pi / 1440^2
        const MMDT_TO_RAD_PER_MIN2: f64 = 2.0 * std::f64::consts::PI / (1440.0 * 1440.0);
        // rev/day^3 -> rad/min^3  = 2*pi / 1440^3
        const MMDT2_TO_RAD_PER_MIN3: f64 = 2.0 * std::f64::consts::PI / (1440.0 * 1440.0 * 1440.0);

        if elements.is_empty() {
            return Err(Error::ValueError(
                "Type 10: need at least one TLE element set.".into(),
            ));
        }
        if pad_days.is_nan() || pad_days < 0.0 {
            return Err(Error::ValueError(format!(
                "Type 10: pad_days must be non-negative, found {pad_days}."
            )));
        }

        // The 8 geophysical constants of the Type 10 format, in the order
        // [j2, j3, j4, ke, qo, so, re, ae], from the sgp4 WGS72 model.
        // qo (120.0) and so (78.0) are the standard drag-layer heights in km.
        // The final 1.0 is ae, the number of distance units per Earth radius.
        let geop_consts: [f64; 8] = [
            sgp4::WGS72.j2,
            sgp4::WGS72.j3,
            sgp4::WGS72.j4,
            sgp4::WGS72.ke,
            120.0,
            78.0,
            sgp4::WGS72.ae,
            1.0,
        ];
        // Sort a local copy by UTC datetime.
        let mut sorted: Vec<&sgp4::Elements> = elements.iter().collect();
        sorted.sort_by_key(|e| e.datetime);

        let mut flat_elements: Vec<f64> = Vec::with_capacity(sorted.len() * 10);
        let mut epochs: Vec<f64> = Vec::with_capacity(sorted.len());

        for elem in &sorted {
            // TLE epochs are UTC. The Type 10 format stores epochs as TDB
            // seconds from J2000.
            let unix = elem.datetime.and_utc();
            // The cast from i64 to f64 can lose precision. Unix seconds in the
            // TLE era are far below 2^53, so the cast is exact.
            #[allow(
                clippy::cast_precision_loss,
                reason = "Unix seconds in the TLE era are exact in f64."
            )]
            let unix_days = (unix.timestamp() as f64
                + f64::from(unix.timestamp_subsec_nanos()) * 1e-9)
                / 86400.0;
            let et = Time::<UTC>::from_parts(unix_days, 2_440_587.5)
                .tdb()
                .j2000_seconds();
            flat_elements.push(elem.mean_motion_dot * MMDT_TO_RAD_PER_MIN2);
            flat_elements.push(elem.mean_motion_ddot * MMDT2_TO_RAD_PER_MIN3);
            flat_elements.push(elem.drag_term);
            flat_elements.push(elem.inclination * DEG2RAD);
            flat_elements.push(elem.right_ascension * DEG2RAD);
            flat_elements.push(elem.eccentricity);
            flat_elements.push(elem.argument_of_perigee * DEG2RAD);
            flat_elements.push(elem.mean_anomaly * DEG2RAD);
            flat_elements.push(elem.mean_motion * MM_TO_RAD_PER_MIN);
            flat_elements.push(et);
            epochs.push(et);
        }

        let pad = pad_days * 86400.0;
        Self::new_array(
            object_id,
            center_id,
            frame_id,
            &geop_consts,
            &flat_elements,
            &epochs,
            epochs[0] - pad,
            epochs[epochs.len() - 1] + pad,
            segment_name,
        )
    }

    /// Parse a block of TLE text and return one `SpkArray` per NORAD catalog
    /// number.
    ///
    /// The text is in 3-line or 2-line TLE format. The NAIF ID of each object
    /// is `-(norad_id as i32)`. All segments use `center_id` and `frame_id`.
    /// The coverage of each segment is the span of its element set epochs,
    /// extended by `pad_days`, as in [`Self::from_tle_records`].
    ///
    /// # Errors
    /// Returns [`Error::ValueError`] if the text cannot be parsed, or if the
    /// text has no TLE. Also returns each error of [`Self::from_tle_records`],
    /// for example when `pad_days` is negative.
    pub fn arrays_from_tle_text(
        text: &str,
        center_id: i32,
        frame_id: i32,
        pad_days: f64,
    ) -> KeteResult<Vec<SpkArray>> {
        let groups = parse_tle_text(text)?;
        if groups.is_empty() {
            return Err(Error::ValueError(
                "No valid TLE records found in the provided text.".into(),
            ));
        }
        let mut arrays = Vec::with_capacity(groups.len());
        for (norad_id, elements) in &groups {
            let object_id = -(*norad_id as i32);
            let name = elements
                .first()
                .and_then(|e| e.object_name.as_deref())
                .unwrap_or("")
                .to_string();
            arrays.push(Self::from_tle_records(
                elements, object_id, center_id, frame_id, pad_days, &name,
            )?);
        }
        Ok(arrays)
    }

    /// Return the position in AU and velocity in AU/day, in Equatorial J2000.
    ///
    /// `jds` is the time in TDB seconds from J2000. The function propagates the
    /// element set with the epoch nearest to `jds`, with SGP4 in improved mode
    /// and the geophysical constants of the segment. At the midpoint between
    /// two epochs, it uses the later element set. It does not blend the
    /// predictions of neighboring element sets.
    ///
    /// SGP4 gives the state in the TEME frame of date, from
    /// [`teme_frame`]. The function rotates the position and the velocity to
    /// Equatorial J2000. The velocity does not include the rotation rate of the
    /// TEME frame.
    ///
    /// # Errors
    /// Returns [`Error::ValueError`] if the epoch of the element set cannot be
    /// converted to a UTC date, if the elements are invalid, or if SGP4 fails
    /// to propagate.
    pub(in crate::spk) fn try_get_pos_vel(
        &self,
        time: Time<TDB>,
    ) -> KeteResult<([f64; 3], [f64; 3])> {
        let jds = time.j2000_seconds();
        let times = self.get_times();
        let n_before = times.partition_point(|&t| t < jds);
        let idx = if n_before == 0 {
            0
        } else if n_before == times.len()
            || (jds - times[n_before - 1]).abs() < (times[n_before] - jds).abs()
        {
            n_before - 1
        } else {
            n_before
        };

        let record = self.get_record(idx)?;
        let prediction = record
            .propagate(MinutesSinceEpoch(
                time.j2000_seconds_minus(self.array.get_packet::<15>(idx)[10]) / 60.0,
            ))
            .map_err(|e| Error::ValueError(format!("SGP4 propagation failed: {e}")))?;
        let pos = Vector3::from(prediction.position);
        let vel = Vector3::from(prediction.velocity);

        let rot = teme_frame(Time::<TDB>::from_j2000_seconds(jds)).rotation;
        let pos = rot * pos / AU_KM;
        let vel = rot * vel / AU_KM * 86400.0;
        Ok((pos.into(), vel.into()))
    }

    /// Build the data array for a Type 10 (TLE) generic segment.
    ///
    /// The data layout follows the cSPICE generic segment format for
    /// explicitly-indexed fixed-size packets:
    /// `[constants][interleaved (ref + packet)...][contiguous refs][ref_dir][metadata]`
    ///
    /// Nutation values (packet indices 10-13) are set to zero; the kete reader
    /// does not use them.
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
        if elements
            .chunks(10)
            .zip(epochs)
            .any(|(set, epoch)| set[9] != *epoch)
        {
            return Err(Error::ValueError(
                "Type 10: epochs must equal the element set epochs.".into(),
            ));
        }

        // In the generic segment layout, the reference directory follows the
        // reference items.
        let n_ref_dir = if n > 100 { (n - 1) / 100 } else { 0 };
        let ref_items_addr = 8 + 15 * n;
        let ref_dir_addr = ref_items_addr + n;
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

    #[inline(always)]
    fn get_times(&self) -> &[f64] {
        self.array.get_reference_items()
    }

    /// Build the SGP4 constants from the element set at `idx`.
    ///
    /// # Errors
    /// Returns [`Error::ValueError`] if the stored epoch does not convert to a
    /// UTC date, or if SGP4 rejects the elements.
    fn get_record(&self, idx: usize) -> KeteResult<Constants> {
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

        // The stored epoch is TDB. SGP4 expects the TLE epoch in UTC, so the
        // epoch converts back to UTC.
        let epoch = julian_years_since_j2000_afspc_compatibility_mode(
            &Time::<TDB>::from_j2000_seconds(epoch)
                .utc()
                .to_datetime()?
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
        .map_err(|e| Error::ValueError(format!("Invalid TLE elements: {e}")))?;
        Constants::new(
            self.geopotential,
            sgp4::afspc_epoch_to_sidereal_time,
            epoch,
            b_star,
            orbit_0,
        )
        .map_err(|e| Error::ValueError(format!("Invalid TLE elements: {e}")))
    }
}

impl TryFrom<SpkArray> for SpkSegmentType10 {
    type Error = Error;
    fn try_from(array: SpkArray) -> KeteResult<Self> {
        let array: GenericSegment = array.try_into()?;
        let constants = array.constants();
        if constants.len() < 8 {
            return Err(Error::IOError(format!(
                "SPK Type 10 needs 8 geophysical constants, found {}.",
                constants.len()
            )));
        }
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

    // The metadata values are f64 in the file, and the casts to usize can lose
    // the sign or truncate. Each metadata value is checked to be a finite,
    // non-negative whole number before its cast. The count of metadata values
    // is checked to be finite and in range, and its cast drops any fraction.
    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        reason = "Metadata values are checked to be non-negative whole numbers, and the count to be in range."
    )]
    fn try_from(array: SpkArray) -> KeteResult<Self> {
        let malformed = || Error::IOError("SPK generic segment is not correctly formatted.".into());
        let len = array.daf.len();

        // The last value of the array is the number of metadata values, stored
        // as an f64. There are at least 15 metadata values.
        let n_meta = *array.daf.data.last().ok_or_else(malformed)?;
        if !(n_meta.is_finite() && n_meta >= 15.0 && n_meta <= len as f64) {
            return Err(malformed());
        }
        let meta = &array.daf.data[len - n_meta as usize..len - 1];
        if meta
            .iter()
            .any(|x| !(x.is_finite() && *x >= 0.0 && x.fract() == 0.0))
        {
            return Err(malformed());
        }
        let meta: Vec<usize> = meta.iter().map(|x| *x as usize).collect();
        let [
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
        ] = meta[..14]
        else {
            return Err(malformed());
        };

        // Later reads of these regions do not check bounds, so each region must
        // lie inside the array.
        let fits = |addr: usize, count: usize| addr.checked_add(count).is_some_and(|e| e <= len);
        if !fits(const_addr, n_consts)
            || !fits(ref_items_addr, n_ref_items)
            || n_packets
                .checked_mul(15)
                .is_none_or(|n| !fits(packet_addr, n))
            || n_ref_items == 0
            || n_packets != n_ref_items
        {
            return Err(malformed());
        }

        Ok(Self {
            array,
            n_meta: n_meta as usize,
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

/// Parse a TLE text block (3-line or 2-line format) into groups keyed by
/// NORAD catalog number.
///
/// Tries `sgp4::parse_3les` first, then falls back to `sgp4::parse_2les`.
/// Each group is sorted by epoch (ascending).
///
/// # Errors
/// Returns an error if the text cannot be parsed as either 3-line or 2-line TLEs.
pub fn parse_tle_text(text: &str) -> KeteResult<Vec<(u64, Vec<sgp4::Elements>)>> {
    let elements = sgp4::parse_3les(text)
        .or_else(|_| sgp4::parse_2les(text))
        .map_err(|e| Error::ValueError(format!("Failed to parse TLE text: {e}")))?;

    let mut groups: std::collections::HashMap<u64, Vec<sgp4::Elements>> =
        std::collections::HashMap::new();
    for elem in elements {
        groups.entry(elem.norad_id).or_default().push(elem);
    }

    let mut result: Vec<(u64, Vec<sgp4::Elements>)> = groups.into_iter().collect();
    for (_, elems) in &mut result {
        elems.sort_by_key(|e| e.datetime);
    }
    result.sort_by_key(|(id, _)| *id);
    Ok(result)
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

        let array = SpkSegmentType10::new_array(
            -25544, 399, 1, &consts, &elements, &epochs, epoch1, epoch2, "test",
        )
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
        let make = |elements: &[f64], epochs: &[f64], first: f64, last: f64| {
            SpkSegmentType10::new_array(1, 399, 1, &consts, elements, epochs, first, last, "t")
        };
        let mut two = [0.0; 20];
        two[19] = 1.0;
        assert!(make(&two, &[0.0, 1.0], 0.0, 1.0).is_ok());
        // Empty elements
        assert!(make(&[], &[], 0.0, 1.0).is_err());
        // Mismatched lengths
        assert!(make(&[0.0; 10], &[0.0, 1.0], 0.0, 1.0).is_err());
        // Non-increasing epochs
        assert!(make(&[0.0; 20], &[1.0, 0.0], 0.0, 1.0).is_err());
        // Reference epochs which differ from the element epochs
        assert!(make(&two, &[0.0, 2.0], 0.0, 2.0).is_err());
        // Coverage which ends before it starts
        assert!(make(&two, &[0.0, 1.0], 1.0, 0.0).is_err());
    }

    #[test]
    fn type10_from_tle_text() {
        // Two ISS TLEs at different epochs; both have valid checksums and are
        // sourced from the sgp4 crate documentation.
        let tle_text = "\
ISS (ZARYA)
1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927
2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537
ISS (ZARYA)
1 25544U 98067A   20194.88612269 -.00002218  00000-0 -31515-4 0  9992
2 25544  51.6461 221.2784 0001413  89.1723 280.4612 15.49507896236008
";
        let groups = parse_tle_text(tle_text).unwrap();
        assert_eq!(groups.len(), 1);
        let (norad_id, elems) = &groups[0];
        assert_eq!(*norad_id, 25544_u64);
        assert_eq!(elems.len(), 2);
        assert!(elems[0].datetime <= elems[1].datetime);

        let array = SpkSegmentType10::from_tle_records(elems, -25544, 399, 1, 0.5, "ISS").unwrap();
        // The coverage is padded by half a day on either side.
        let start = array.jds_start;
        let end = array.jds_end;
        let seg: SpkSegmentType10 = array.try_into().unwrap();
        let times = seg.get_times();
        assert_eq!(times.len(), 2);
        // The 2008 epoch, 2008-264T12:25:40.104192 UTC, in TDB seconds from
        // J2000. The reference value is from SPICE STR2ET.
        assert!(
            (times[0] - 275_185_605.286_586).abs() < 1e-3,
            "{}",
            times[0]
        );
        assert!((start - (times[0] - 43200.0)).abs() < 1e-3);
        assert!((end - (times[1] + 43200.0)).abs() < 1e-3);
    }

    /// Each request uses the element set with the nearest epoch. At the epoch
    /// of a set, the state is the prediction of that set. Past the midpoint
    /// between two sets, the later set is used.
    #[test]
    fn type10_uses_the_nearest_element_set() {
        let tle_text = "\
ISS (ZARYA)
1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927
2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537
ISS (ZARYA)
1 25544U 98067A   08265.51782528 -.00002182  00000-0 -11606-4 0  2928
2 25544  51.6416 242.4627 0006703 130.5360 325.0288 15.72125391563532
";
        let arrays = SpkSegmentType10::arrays_from_tle_text(tle_text, 399, 1, 0.0).unwrap();
        let seg: SpkSegmentType10 = arrays.into_iter().next().unwrap().try_into().unwrap();
        let t1 = seg.array.get_packet::<15>(0)[10];
        let t2 = seg.array.get_packet::<15>(1)[10];

        // The prediction of one element set, rotated to J2000, in km.
        let predict = |idx: usize, jds: f64| {
            let record = seg.get_record(idx).unwrap();
            let epoch = seg.array.get_packet::<15>(idx)[10];
            let p = record
                .propagate(MinutesSinceEpoch((jds - epoch) / 60.0))
                .unwrap()
                .position;
            *teme_frame(Time::<TDB>::from_j2000_seconds(jds))
                .rotation
                .matrix()
                * Vector3::from(p)
        };
        let mid = f64::midpoint(t1, t2);
        for (jds, idx) in [(t1, 0), (mid - 60.0, 0), (mid + 60.0, 1), (t2, 1)] {
            let (pos, _) = seg.try_get_pos_vel(Time::from_j2000_seconds(jds)).unwrap();
            let err = (Vector3::from(pos) * AU_KM - predict(idx, jds)).norm();
            assert!(err < 1e-6, "{jds}: {err} km");
        }
    }

    #[test]
    fn type10_from_tle_text_two_objects() {
        // Two different objects using verified TLE pairs from the sgp4 test suite.
        let tle_text = "\
VANGUARD 1
1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753
2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667
ISS (ZARYA)
1 25544U 98067A   20194.88612269 -.00002218  00000-0 -31515-4 0  9992
2 25544  51.6461 221.2784 0001413  89.1723 280.4612 15.49507896236008
";
        let groups = parse_tle_text(tle_text).unwrap();
        assert_eq!(groups.len(), 2);
        let arrays = SpkSegmentType10::arrays_from_tle_text(tle_text, 399, 1, 0.5).unwrap();
        assert_eq!(arrays.len(), 2);
    }
}
