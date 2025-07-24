use kete_core::constants::AU_KM;
use kete_core::desigs::Desig;
use kete_core::frames::geodetic_lat_lon_to_ecef;
use kete_core::spice::{LOADED_PCK, LOADED_SPK};
use pyo3::{PyResult, Python, pyfunction};

use crate::desigs::NaifIDLike;
use crate::frame::PyFrames;
use crate::spice::{find_obs_code_py, pck_earth_frame_py};
use crate::state::PyState;
use crate::time::PyTime;

/// Load all specified files into the SPK shared memory singleton.
#[pyfunction]
#[pyo3(name = "spk_load")]
pub fn spk_load_py(py: Python<'_>, filenames: Vec<String>) -> PyResult<()> {
    let mut singleton = LOADED_SPK.write().unwrap();
    if filenames.len() > 100 {
        eprintln!("Loading {} spk files...", filenames.len());
    }
    for filename in filenames.iter() {
        py.check_signals()?;
        let load = (*singleton).load_file(filename);
        if let Err(err) = load {
            eprintln!("{filename} failed to load. {err}");
        }
    }
    Ok(())
}

/// Return all loaded SPK info on the specified NAIF ID.
/// Loaded info contains:
/// (name, JD_start, JD_end, Center Naif ID, Frame ID, SPK Segment type ID)
#[pyfunction]
#[pyo3(name = "_loaded_object_info")]
pub fn spk_available_info_py(naif_id: NaifIDLike) -> Vec<(String, f64, f64, i32, i32, i32)> {
    let (name, naif_id) = naif_id.try_into().unwrap();
    let singleton = &LOADED_SPK.try_read().unwrap();
    singleton
        .available_info(naif_id)
        .into_iter()
        .map(|(jd_start, jd_end, center_id, frame_id, segment_id)| {
            (
                name.clone(),
                jd_start,
                jd_end,
                center_id,
                frame_id,
                segment_id,
            )
        })
        .collect()
}

/// Return a list of all NAIF objects currently loaded in the SPICE shared memory singleton.
///
#[pyfunction]
#[pyo3(name = "loaded_objects")]
pub fn spk_loaded_objects_py() -> Vec<String> {
    let spk = &LOADED_SPK.try_read().unwrap();
    let loaded = spk.loaded_objects(false);
    let mut loaded: Vec<_> = loaded.into_iter().collect();
    loaded.sort();
    loaded
        .into_iter()
        .map(|spkid| Desig::Naif(spkid).try_naif_id_to_name().to_string())
        .collect()
}

/// Reset the contents of the SPK shared memory.
#[pyfunction]
#[pyo3(name = "spk_reset")]
pub fn spk_reset_py() {
    LOADED_SPK.write().unwrap().reset()
}

/// Reload the core SPK files.
#[pyfunction]
#[pyo3(name = "spk_load_core")]
pub fn spk_load_core_py() {
    LOADED_SPK.write().unwrap().load_core().unwrap()
}

/// Reload the core PCK files.
#[pyfunction]
#[pyo3(name = "pck_load_core")]
pub fn pck_load_core_py() {
    LOADED_PCK.write().unwrap().load_core().unwrap()
}

/// Reload the cache SPK files.
#[pyfunction]
#[pyo3(name = "spk_load_cache")]
pub fn spk_load_cache_py() {
    LOADED_SPK.write().unwrap().load_cache().unwrap()
}

/// Calculates the :class:`~kete.State` of the target object at the
/// specified time `jd`.
///
/// This defaults to the ecliptic heliocentric state, though other centers may be
/// chosen.
///
/// Parameters
/// ----------
/// target:
///     The names of the target object, this can include any object name listed in
///     :meth:`~kete.spice.loaded_objects`
/// jd:
///     Julian time (TDB) of the desired record.
/// center:
///     The center point, this defaults to being heliocentric.
/// frame:
///     Coordinate frame of the state, defaults to ecliptic.
///
/// Returns
/// -------
/// State
///     Returns the ecliptic state of the target in AU and AU/days.
///
/// Raises
/// ------
/// ValueError
///     If the desired time is outside of the range of the source binary file.
#[pyfunction]
#[pyo3(name = "get_state", signature = (id, jd, center=NaifIDLike::Int(10), frame=PyFrames::Ecliptic))]
pub fn spk_state_py(
    id: NaifIDLike,
    jd: PyTime,
    center: NaifIDLike,
    frame: PyFrames,
) -> PyResult<PyState> {
    let jd = jd.jd();
    let (_, center) = center.try_into()?;
    match id.clone().try_into() {
        Ok((_, id)) => {
            let spk = &LOADED_SPK.try_read().unwrap();
            let mut state = spk.try_get_state_with_center(id, jd, center)?;
            state.try_naif_id_to_name();
            Ok(PyState {
                raw: state,
                frame,
                elements: None,
            })
        }
        Err(e) => {
            if let NaifIDLike::String(name) = id {
                let (lat, lon, h, name, _) = find_obs_code_py(&name)?;
                let mut ecef = geodetic_lat_lon_to_ecef(lat.to_radians(), lon.to_radians(), h);
                ecef.iter_mut().for_each(|x| *x /= AU_KM);

                return pck_earth_frame_py(ecef, jd.into(), center, Some(name));
            }
            Err(e.clone().into())
        }
    }
}

/// Return the raw state of an object as encoded in the SPK Kernels.
///
/// This does not change center point, but all states are returned in
/// the Equatorial frame.
///
/// Parameters
/// ----------
/// id : int
///     NAIF ID of the object.
/// jd : float
///     Time (JD) in TDB scaled time.
#[pyfunction]
#[pyo3(name = "spk_raw_state")]
pub fn spk_raw_state_py(id: NaifIDLike, jd: PyTime) -> PyResult<PyState> {
    let (_, id) = id.try_into()?;
    let jd = jd.jd();
    let spk = &LOADED_SPK.try_read().unwrap();
    Ok(PyState {
        raw: spk.try_get_state(id, jd)?,
        frame: PyFrames::Equatorial,
        elements: None,
    })
}
