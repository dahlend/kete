use kete_core::frames::ecef_to_geodetic_lat_lon;
use kete_core::spice::{LOADED_PCK, LOADED_SPK};
use kete_core::{constants, prelude::*};
use pyo3::{PyResult, pyfunction};

use crate::frame::PyFrames;
use crate::state::PyState;
use crate::time::PyTime;

/// Load all specified files into the PCK shared memory singleton.
#[pyfunction]
#[pyo3(name = "pck_load")]
pub fn pck_load_py(filenames: Vec<String>) -> PyResult<()> {
    let mut singleton = LOADED_PCK.write().unwrap();
    for filename in filenames.iter() {
        let load = (*singleton).load_file(filename);
        if let Err(err) = load {
            eprintln!("{filename} failed to load. {err}");
        }
    }
    Ok(())
}

/// Convert a position vector which is geocentered in the earth frame to a position
/// vector in the sun centered ecliptic frame.
///
/// This uses PCK files produces by the NAIF team at JPL. New files may be downloaded
/// from the NAIF website if so desired, but the one provided is a highest accuracy
/// copy available as of Jan 2025.
///
/// parameters
/// ----------
/// pos : list[float]
///     Position of the object in the geocentric reference frame in AU.
/// jd : float
///     Time in JD using TDB scaling.
/// new_center: int
///     NAIF ID of the new center point.
/// name : String
///     Optional name of the state.
#[pyfunction]
#[pyo3(name = "pck_earth_frame_to_ecliptic", signature = (pos, jd, new_center, name=None))]
pub fn pck_earth_frame_py(
    pos: [f64; 3],
    jd: PyTime,
    new_center: i32,
    name: Option<String>,
) -> PyResult<PyState> {
    let jd = jd.jd();
    let desig = {
        match name {
            Some(d) => Desig::Name(d),
            None => Desig::Empty,
        }
    };
    let pcks = LOADED_PCK.try_read().unwrap();
    let frame = pcks.try_get_orientation(3000, jd)?;

    let (pos, vel) = frame.to_equatorial(pos.into(), [0.0, 0.0, 0.0].into())?;

    let mut state: State<Equatorial> = State::new(desig, jd, pos.into(), vel.into(), 399);

    let spks = &LOADED_SPK.try_read().unwrap();
    spks.try_change_center(&mut state, new_center)?;

    Ok(PyState {
        raw: state,
        frame: PyFrames::Ecliptic,
        elements: None,
    })
}

/// Convert a [`State`] to the Earth's surface lat/lon/height on the WGS84 reference.
///
/// This requires the `earth_000101_*.pck` file to be loaded which contains the
/// instantaneous earth frame information. The one provided by kete has dates from
/// around 2000 to early 2024, along with predicts into the future. New files may be
/// downloaded from the NAIF website for additional precision or years of epoch.
///
/// Parameters
/// ----------
/// state: State
///     Convert the given state to latitude, longitude, and height in km on the WGS84
///     reference.
#[pyfunction]
#[pyo3(name = "state_to_earth_pos")]
pub fn pck_state_to_earth(state: PyState) -> PyResult<(f64, f64, f64)> {
    let pcks = LOADED_PCK.try_read().unwrap();
    let state = state
        .change_center(crate::desigs::NaifIDLike::Int(399))?
        .raw;
    let frame = pcks.try_get_orientation(3000, state.jd)?;

    let (pos, _) = frame.from_equatorial(state.pos.into(), state.vel.into())?;
    let [x, y, z] = pos.into();
    let (lat, lon, height) = ecef_to_geodetic_lat_lon(
        x * constants::AU_KM,
        y * constants::AU_KM,
        z * constants::AU_KM,
    );

    Ok((lat.to_degrees(), lon.to_degrees(), height))
}

/// Reset the contents of the PCK shared memory to the default set of PCK kernels.
#[pyfunction]
#[pyo3(name = "pck_reset")]
pub fn pck_reset_py() {
    LOADED_PCK.write().unwrap().reset()
}

/// Return a list of all loaded objects in the PCK singleton.
/// This is a list of the center NAIF IDs of the segments.
#[pyfunction]
#[pyo3(name = "pck_loaded")]
pub fn pck_loaded_objects_py() -> Vec<i32> {
    let loaded = LOADED_PCK.read().unwrap();
    loaded.loaded_objects()
}
