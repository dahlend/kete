//! Fields of view of instruments at a time, from the loaded kernels.
//!
//! An instrument kernel gives the field of view in the frame of the instrument
//! (see [`crate::text::ik`]). [`instrument_fov_at`] rotates it to the
//! equatorial frame with [`try_frame_at`], and places the observer at the
//! spacecraft with the loaded SPK files. The pointing is geometric: no light
//! time or aberration correction is applied to it.

use crate::ck::{CkCollection, LOADED_CK};
use crate::ephemeris::SpiceEphemeris;
use crate::frames::{FrameDef, MAX_FRAME_CHAIN, try_frame_at};
use crate::text::ik::FovShape;
use crate::text::{LOADED_TEXT_KERNELS, TextKernels};
use kete_core::ephemeris::Ephemeris;
use kete_core::errors::{Error, KeteResult};
use kete_core::fov::{FOV, FovLike, GenericCone, GenericPolygon, GenericRectangle};
use kete_core::frames::{Equatorial, FrameId, Vector};
use kete_core::time::{TDB, Time};

/// The field of view of instrument `id` at `time`.
///
/// The observer is the spacecraft of the instrument, relative to the Sun. It
/// is the spacecraft of the first CK frame in the chain of TK frames from the
/// frame of the field of view (see [`TextKernels::ck_spk_id`]). Without a CK
/// frame in that chain, it is the instrument ID divided by 1000, the NAIF
/// convention for instrument IDs.
/// A rectangle gives a [`GenericRectangle`], a circle a
/// [`GenericCone`], and a polygon a [`GenericPolygon`].
///
/// # Errors
/// - [`Error::LockFailed`] if a kernel singleton cannot be read.
/// - [`Error::ValueError`] if the field of view definition is missing or
///   malformed, if the field of view is an ellipse, if a polygon is not valid
///   (see [`GenericPolygon::new`]), or if the instrument gives no spacecraft.
/// - The errors of [`try_frame_at`] for the frame of the field of view.
/// - [`Error::Bounds`] if the loaded SPK files have no state of the spacecraft
///   at `time`.
pub fn instrument_fov_at(id: i32, time: Time<TDB>) -> KeteResult<FOV> {
    let (fov, spacecraft) = {
        let text = LOADED_TEXT_KERNELS.try_read()?;
        let fov = text.instrument_fov(id)?;
        let ck = LOADED_CK.try_read()?;
        let spacecraft = spacecraft_id(&text, &ck, id, fov.frame)?;
        (fov, spacecraft)
    };
    if fov.shape == FovShape::Ellipse {
        return Err(Error::ValueError(format!(
            "Instrument {id} has an elliptical field of view, which kete does not \
             support."
        )));
    }
    let rotation = try_frame_at(fov.frame, time)?.rotation_to_equatorial()?;
    let to_equatorial = |v: &nalgebra::Vector3<f64>| Vector::<Equatorial>::from(rotation * v);
    let observer = SpiceEphemeris::loaded()?.try_get_state_with_center(spacecraft, time, 10)?;
    let bounds: Vec<Vector<Equatorial>> = fov.bounds.iter().map(to_equatorial).collect();

    Ok(match fov.shape {
        FovShape::Rectangle => {
            let corners = [bounds[0], bounds[1], bounds[2], bounds[3]];
            GenericRectangle::from_corners(corners, observer, 0.0).into_fov()
        }
        FovShape::Circle => {
            let boresight = to_equatorial(&fov.boresight);
            let angle = boresight.angle(&bounds[0]);
            GenericCone::new(boresight.normalize(), angle, observer).into_fov()
        }
        FovShape::Polygon => GenericPolygon::new(&bounds, observer)?.into_fov(),
        FovShape::Ellipse => unreachable!("an ellipse returns an error above"),
    })
}

/// The SPK ID of the spacecraft that carries instrument `id`, whose field of
/// view is in frame `frame`.
///
/// See [`instrument_fov_at`]. For example, the spacecraft of instrument
/// -226111 is -226.
///
/// # Errors
/// [`Error::ValueError`] if a frame in the chain is malformed, if the TK
/// chain loops or holds more than [`MAX_FRAME_CHAIN`] frames, if the CK frame
/// gives no SPK ID, or if the chain has no CK frame and the instrument ID is
/// above -1000 and below 1000.
fn spacecraft_id(
    text: &TextKernels,
    ck: &CkCollection,
    id: i32,
    frame: FrameId,
) -> KeteResult<i32> {
    let mut frame_id = frame;
    for _ in 0..MAX_FRAME_CHAIN {
        match text.frame_definition(frame_id)? {
            Some(FrameDef::Tk { relative, .. }) => frame_id = relative,
            Some(FrameDef::Ck { class_id, .. }) => return text.ck_spk_id(class_id),
            None if ck.has_instrument(frame_id.0) => return text.ck_spk_id(frame_id.0),
            Some(FrameDef::Inertial | FrameDef::Pck { .. }) | None => {
                return match id / 1000 {
                    0 => Err(Error::ValueError(format!(
                        "Instrument {id} gives no spacecraft ID."
                    ))),
                    spacecraft => Ok(spacecraft),
                };
            }
        }
    }
    Err(Error::ValueError(format!(
        "The frame of instrument {id} is in a cycle, or a chain of more than \
         {MAX_FRAME_CHAIN} TK frames."
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The spacecraft is that of the first CK frame in the TK chain from the
    /// frame of the field of view, else the instrument ID divided by 1000.
    #[test]
    fn spacecraft_ids() {
        let mut text = TextKernels::default();
        text.load_text(
            "\\begindata\n\
             FRAME_SC = -399000\nFRAME_-399000_NAME = 'SC'\n\
             FRAME_-399000_CLASS = 3\nFRAME_-399000_CLASS_ID = -399000\n\
             CK_-399000_SPK = 399\n\
             FRAME_BUS = -226000\nFRAME_-226000_NAME = 'BUS'\n\
             FRAME_-226000_CLASS = 3\nFRAME_-226000_CLASS_ID = -226000\n\
             FRAME_CAM = -399100\nFRAME_-399100_NAME = 'CAM'\n\
             FRAME_-399100_CLASS = 4\nFRAME_-399100_CLASS_ID = -399100\n\
             TKFRAME_-399100_RELATIVE = 'SC'\nTKFRAME_-399100_SPEC = 'MATRIX'\n\
             TKFRAME_-399100_MATRIX = ( 1 0 0 0 1 0 0 0 1 )\n\
             FRAME_LOOP = -5100\nFRAME_-5100_NAME = 'LOOP'\n\
             FRAME_-5100_CLASS = 4\nFRAME_-5100_CLASS_ID = -5100\n\
             TKFRAME_-5100_RELATIVE = 'LOOP'\nTKFRAME_-5100_SPEC = 'MATRIX'\n\
             TKFRAME_-5100_MATRIX = ( 1 0 0 0 1 0 0 0 1 )\n",
        )
        .unwrap();
        let ck = CkCollection::default();
        let spacecraft = |id, frame| spacecraft_id(&text, &ck, id, FrameId(frame));
        assert_eq!(spacecraft(-399_101, -399_100).unwrap(), 399);
        assert_eq!(spacecraft(-1_101, -226_000).unwrap(), -226);
        assert_eq!(spacecraft(-226_111, 1).unwrap(), -226);
        assert_eq!(spacecraft(399_001, 1).unwrap(), 399);
        assert!(spacecraft(-226, 1).is_err());
        assert!(spacecraft(-5_101, -5_100).is_err());
    }

    /// A field of view in J2000 on an instrument of the Earth: the FOV holds
    /// the directions of the boundary, and the Earth relative to the Sun as the
    /// observer. An ellipse is an error.
    #[test]
    fn instrument_fov_from_kernels() {
        crate::test_data::ensure_test_spk();
        LOADED_TEXT_KERNELS
            .write()
            .unwrap()
            .load_text(
                "\\begindata\nINS399901_FOV_FRAME = 'J2000'\nINS399901_FOV_SHAPE = 'POLYGON'\n\
                 INS399901_BORESIGHT = ( 1 0 0 )\n\
                 INS399901_FOV_BOUNDARY_CORNERS = ( 1 -0.1 -0.1  1 0.1 -0.1  1 0.1 0.1\n\
                 1 0.02 0.0  1 -0.1 0.1 )\n\
                 INS399902_FOV_FRAME = 'J2000'\nINS399902_FOV_SHAPE = 'ELLIPSE'\n\
                 INS399902_BORESIGHT = ( 1 0 0 )\n\
                 INS399902_FOV_BOUNDARY_CORNERS = ( 1 0.1 0  1 0 0.05 )\n\
                 NAIF_BODY_NAME += 'TEST CAMERA'\nNAIF_BODY_CODE += 399901\n",
            )
            .unwrap();
        let time = Time::<TDB>::new(2_451_545.0);
        let fov = instrument_fov_at(399_901, time).unwrap();
        let FOV::GenericPolygon(polygon) = &fov else {
            panic!("not a polygon");
        };
        assert!(!polygon.patch.is_convex());
        let corner = fov.corners().unwrap()[3];
        assert!((corner - Vector::new([1.0, 0.02, 0.0]).normalize()).norm() < 1e-15);
        let earth = SpiceEphemeris::loaded()
            .unwrap()
            .try_get_state_with_center(399, time, 10)
            .unwrap();
        assert_eq!(fov.observer().pos, earth.pos);
        assert!(instrument_fov_at(399_902, time).is_err());
        assert_eq!(
            LOADED_TEXT_KERNELS.read().unwrap().body_id("test   camera"),
            Some(399_901)
        );
    }
}
