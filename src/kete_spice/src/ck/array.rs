use crate::daf::{DAFType, DafArray};
use kete_core::errors::{Error, KeteResult};

/// DAF Array of CK data.
/// These are segments of data.
/// This is a wrapper around the [`DafArray`] which is specific to CK data.
#[derive(Debug)]
pub struct CkArray {
    /// The internal representation of the DAF array.
    pub daf: DafArray,

    /// Start SCLK tick time of the spacecraft.
    pub tick_start: f64,

    /// End SCLK tick time of the spacecraft.
    pub tick_end: f64,

    /// Instrument ID
    pub instrument_id: i32,

    /// NAIF ID of the spacecraft.
    pub naif_id: i32,

    /// The spice frame ID of the array.
    /// Called the `Reference` in SPICE documentation.
    pub reference_frame_id: i32,

    /// The spice segment type.
    pub segment_type: i32,

    /// Does this segment produce angular rates.
    pub produces_angular_rates: bool,
}

impl CkArray {
    /// Is the specified SCLK tick within the range of this array.
    #[must_use]
    pub fn contains(&self, tick: f64) -> bool {
        (tick >= self.tick_start) && (tick <= self.tick_end)
    }

    /// Construct a new CK array from high-level parameters and data.
    #[must_use]
    pub fn new(
        instrument_id: i32,
        reference_frame_id: i32,
        segment_type: i32,
        produces_angular_rates: bool,
        tick_start: f64,
        tick_end: f64,
        data: Vec<f64>,
        name: String,
    ) -> Self {
        let avflag = i32::from(produces_angular_rates);
        let naif_id = instrument_id / 1000;
        let summary_floats: Box<[f64]> = vec![tick_start, tick_end].into();
        let summary_ints: Box<[i32]> = vec![
            instrument_id,
            reference_frame_id,
            segment_type,
            avflag,
            0,
            0,
        ]
        .into();
        let daf = DafArray::new(summary_floats, summary_ints, data.into(), DAFType::Ck, name);
        Self {
            daf,
            tick_start,
            tick_end,
            instrument_id,
            naif_id,
            reference_frame_id,
            segment_type,
            produces_angular_rates,
        }
    }
}

/// Convert a count stored as a float in a CK segment to `usize`.
///
/// `value` is the stored float. `max` is the largest valid count. `what` names
/// the count in the error message.
///
/// # Errors
/// [`Error::IOError`] if `value` is not a finite whole number in `[0, max]`.
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "The value is checked to be a whole number in range first."
)]
pub(in crate::ck) fn stored_count(value: f64, max: usize, what: &str) -> KeteResult<usize> {
    if value.is_finite() && value >= 0.0 && value.fract() == 0.0 && value <= max as f64 {
        Ok(value as usize)
    } else {
        Err(Error::IOError(format!(
            "CK segment has an invalid {what}: {value}."
        )))
    }
}

impl TryFrom<DafArray> for CkArray {
    type Error = Error;

    fn try_from(array: DafArray) -> Result<Self, Self::Error> {
        if array.daf_type != DAFType::Ck {
            return Err(Error::IOError("DAF Array is not a CK array.".into()));
        }

        if array.summary_floats.len() != 2 {
            return Err(Error::IOError(
                "DAF Array is not a CK array. Summary of array is incorrectly formatted, incorrect number of floats.".into(),
            ));
        }

        if array.summary_ints.len() != 6 {
            return Err(Error::IOError("DAF Array is not a CK array. Summary of array is incorrectly formatted, incorrect number of ints.".into()));
        }

        let tick_start = array.summary_floats[0];
        let tick_end = array.summary_floats[1];

        // The last two integers in the summary are the start and end of the array.
        // Those two values are already contained within the DafArray stored in this
        // object.
        let instrument_id = array.summary_ints[0];
        let naif_id = array.summary_ints[0] / 1000;
        let frame_id = array.summary_ints[1];
        let segment_type = array.summary_ints[2];
        let produces_angular_rates = array.summary_ints[3] == 1;

        Ok(Self {
            daf: array,
            tick_start,
            tick_end,
            instrument_id,
            naif_id,
            reference_frame_id: frame_id,
            segment_type,
            produces_angular_rates,
        })
    }
}
