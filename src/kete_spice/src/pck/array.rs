use crate::daf::{DAFType, DafArray};
use kete_core::errors::Error;
use kete_core::time::{TDB, Time};

#[derive(Debug)]

/// DAF Array of PCK data.
/// This is a wrapper around the [`DafArray`] which is specific to PCK data.
pub struct PckArray {
    /// The internal representation of the DAF array.
    pub daf: DafArray,

    /// Start of the coverage, in TDB seconds from J2000.
    pub jds_start: f64,

    /// End of the coverage, in TDB seconds from J2000.
    pub jds_end: f64,

    /// The ID which identifies this frame.
    pub frame_id: i32,

    /// The inertial reference frame this PCK is defined against.
    pub reference_frame_id: i32,

    /// The spice segment type.
    pub segment_type: i32,
}

impl PckArray {
    /// Is the specified JD within the range of this array.
    #[must_use]
    pub fn contains(&self, jd: Time<TDB>) -> bool {
        let jds = jd.j2000_seconds();
        (jds >= self.jds_start) && (jds <= self.jds_end)
    }

    /// Construct a new PCK array from high-level parameters and data.
    ///
    /// `frame_id` is the ID of the body-fixed frame. `reference_frame_id` is
    /// the ID of the inertial reference frame. `segment_type` is the PCK
    /// segment type. `jds_start` and `jds_end` are the coverage start and end,
    /// in TDB seconds from J2000, as the file stores them. `data` is the raw
    /// array content for the segment type. `name` is the name stored in the DAF
    /// name record.
    #[must_use]
    pub fn new(
        frame_id: i32,
        reference_frame_id: i32,
        segment_type: i32,
        jds_start: f64,
        jds_end: f64,
        data: Vec<f64>,
        name: String,
    ) -> Self {
        let summary_floats: Box<[f64]> = vec![jds_start, jds_end].into();
        let summary_ints: Box<[i32]> =
            vec![frame_id, reference_frame_id, segment_type, 0, 0].into();
        let daf = DafArray::new(
            summary_floats,
            summary_ints,
            data.into(),
            DAFType::Pck,
            name,
        );
        Self {
            daf,
            jds_start,
            jds_end,
            frame_id,
            reference_frame_id,
            segment_type,
        }
    }
}

impl TryFrom<DafArray> for PckArray {
    type Error = Error;

    fn try_from(array: DafArray) -> Result<Self, Self::Error> {
        if array.daf_type != DAFType::Pck {
            Err(Error::IOError("DAF Array is not a PCK array.".into()))?;
        }

        if array.summary_floats.len() != 2 {
            Err(Error::IOError(
                "DAF Array is not a PCK array. Summary of array is incorrectly formatted, incorrect number of floats.".into(),
            ))?;
        }

        if array.summary_ints.len() != 5 {
            Err(Error::IOError("DAF Array is not a PCK array. Summary of array is incorrectly formatted, incorrect number of ints.".into()))?;
        }

        let jds_start = array.summary_floats[0];
        let jds_end = array.summary_floats[1];

        // The last two integers in the summary are the start and end of the array.
        let frame_id = array.summary_ints[0];
        let reference_frame_id = array.summary_ints[1];
        let segment_type = array.summary_ints[2];

        Ok(Self {
            daf: array,
            jds_start,
            jds_end,
            frame_id,
            reference_frame_id,
            segment_type,
        })
    }
}
