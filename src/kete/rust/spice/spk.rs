use kete_core::constants::AU_KM;
use kete_core::desigs::Desig;
use kete_core::frames::geodetic_lat_lon_to_ecef;
use kete_spice::prelude::{DafFile, LOADED_SPK};
use kete_spice::spk::repack_to_type2;
use kete_spice::spk::repack_to_type13;
use kete_spice::spk::type10::{SpkSegmentType10, parse_tle_text};
use pyo3::{PyResult, Python, pyclass, pyfunction, pymethods};
use std::collections::{HashMap, HashSet};

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
    singleton.build_mapping();
    Ok(())
}

/// Return all loaded SPK info on the specified NAIF ID.
/// Loaded info contains:
/// (name, JD_start, JD_end, Center Naif ID, Frame ID, SPK Segment type ID)
#[pyfunction]
#[pyo3(name = "_loaded_object_info")]
pub fn spk_available_info_py(naif_id: NaifIDLike) -> Vec<(String, PyTime, PyTime, i32, i32, i32)> {
    let (name, naif_id) = naif_id.try_into().unwrap();
    let singleton = &LOADED_SPK.try_read().unwrap();
    singleton
        .available_info(naif_id)
        .into_iter()
        .map(|(jd_start, jd_end, center_id, frame_id, segment_id)| {
            (
                name.clone(),
                jd_start.into(),
                jd_end.into(),
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
pub fn spk_reset_py() -> PyResult<()> {
    LOADED_SPK
        .write()
        .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("SPK lock poisoned"))?
        .reset();
    Ok(())
}

/// Reload the core SPK files.
#[pyfunction]
#[pyo3(name = "spk_load_core")]
pub fn spk_load_core_py() -> PyResult<()> {
    LOADED_SPK
        .write()
        .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("SPK lock poisoned"))?
        .load_core()?;
    Ok(())
}

/// Reload the cache SPK files.
#[pyfunction]
#[pyo3(name = "spk_load_cache")]
pub fn spk_load_cache_py() -> PyResult<()> {
    LOADED_SPK
        .write()
        .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("SPK lock poisoned"))?
        .load_cache()?;
    Ok(())
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
    let jd = jd.into();
    let (_, center) = center.try_into()?;
    match id.clone().try_into() {
        Ok((name, id)) => {
            let spk = &LOADED_SPK.try_read().unwrap();
            let mut state = spk.try_get_state_with_center(id, jd, center)?;
            state.desig = Desig::Name(name);
            Ok(PyState {
                raw: state,
                frame,
                elements: None,
            })
        }
        Err(e) => {
            if let NaifIDLike::String(name) = id {
                let (lat, lon, h, name, _) = find_obs_code_py(&name).map_err(|_| {
                    kete_core::errors::Error::ValueError(format!(
                        "Failed to resolve the specified object: {name}"
                    ))
                })?;
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
    let jd = jd.into();
    let spk = &LOADED_SPK.try_read().unwrap();
    Ok(PyState {
        raw: spk.try_get_state(id, jd)?,
        frame: PyFrames::Equatorial,
        elements: None,
    })
}

/// Builder for creating multi-segment SPK binary kernel files.
///
/// Segments of different types can be added incrementally before writing the
/// completed file to disk.  This is the primary entry point for creating new
/// SPK files from Python.
///
/// Examples
/// --------
/// >>> builder = kete.spice.SpkBuilder()
/// >>> builder.add_tle_segment("iss_tles.txt", -25544, 399)
/// >>> builder.write("iss.bsp")
///
/// Parameters
/// ----------
/// internal_desc :
///     Short internal description embedded in the DAF header (max 60 chars).
/// comment :
///     Free-text comment block written into the DAF file.
#[pyclass(name = "SpkBuilder")]
#[derive(Debug)]
pub struct PySpkBuilder {
    daf: DafFile,
}

#[pymethods]
impl PySpkBuilder {
    /// Create a new :class:`SpkBuilder`.
    ///
    /// Parameters
    /// ----------
    /// internal_desc :
    ///     Short description embedded in the DAF header.
    /// comment :
    ///     Free-text comment written into the file.
    #[new]
    #[pyo3(signature = (internal_desc = "", comment = ""))]
    pub fn new(internal_desc: &str, comment: &str) -> Self {
        Self {
            daf: DafFile::new_spk(internal_desc, comment),
        }
    }

    /// Add a Type 10 (TLE) segment from a TLE text file.
    ///
    /// All TLE entries in the file that share the same NORAD catalog number are
    /// combined into a single SPK segment.  NAIF object IDs are set to
    /// ``-(norad_id)``.
    ///
    /// Parameters
    /// ----------
    /// tle_file :
    ///     Path to a text file containing TLEs (2-line or 3-line format).
    /// center_id :
    ///     NAIF ID of the central body (399 = Earth).
    /// frame_id :
    ///     NAIF frame ID (1 = J2000 equatorial, the standard for TLE data).
    pub fn add_tle_segment(
        &mut self,
        tle_file: &str,
        center_id: i32,
        frame_id: i32,
    ) -> PyResult<()> {
        let text = std::fs::read_to_string(tle_file).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!(
                "Failed to read TLE file '{}': {}",
                tle_file, e
            ))
        })?;
        let arrays = SpkSegmentType10::arrays_from_tle_text(&text, center_id, frame_id)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        for array in arrays {
            self.daf.arrays.push(array.daf);
        }
        Ok(())
    }

    /// Repack segments for an object from the loaded SPK kernels and add them
    /// to this builder.
    ///
    /// Reads from the currently loaded SPK singleton, fits the positions in
    /// the Equatorial J2000 frame, and validates that the fit error is within
    /// the specified threshold.
    ///
    /// Parameters
    /// ----------
    /// object_id :
    ///     NAIF ID of the body to repack.
    /// center_id :
    ///     NAIF ID of the reference center body (default 10 = Sun).
    /// threshold_km :
    ///     Maximum allowable position error in km (default 0.5).
    /// degree :
    ///     Polynomial degree (default 15). For Type 2 Chebyshev: 1 to 27.
    ///     For Type 13 Hermite: must be odd and in 1 to 27 (default 7).
    /// output_type :
    ///     SPK segment type for the output: 2 (Chebyshev) or 13 (Hermite).
    ///     Type 2 compresses best for slow orbits (asteroids, planets).
    ///     Type 13 is suited for fast orbiters (LEO, MEO). Default 2.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If no coverage exists, the degree is out of range, or the threshold
    ///     cannot be met.
    #[pyo3(signature = (object_id, center_id=10, threshold_km=0.5, degree=None, output_type=2))]
    pub fn add_repacked_segment(
        &mut self,
        object_id: i32,
        center_id: i32,
        threshold_km: f64,
        degree: Option<usize>,
        output_type: i32,
    ) -> PyResult<()> {
        let spk = LOADED_SPK
            .try_read()
            .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("SPK lock poisoned"))?;
        let arrays = match output_type {
            2 => {
                let deg = degree.unwrap_or(15);
                repack_to_type2(&spk, object_id, center_id, threshold_km, deg, None)
            }
            13 => {
                let deg = degree.unwrap_or(7);
                repack_to_type13(&spk, object_id, center_id, threshold_km, deg, None)
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "output_type must be 2 or 13, got {output_type}"
                )));
            }
        }
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        for array in arrays {
            self.daf.arrays.push(array.daf);
        }
        Ok(())
    }

    /// Number of segments currently held by this builder.
    #[getter]
    pub fn n_segments(&self) -> usize {
        self.daf.arrays.len()
    }

    /// Write the completed SPK file to *filename*.
    ///
    /// The file must not already exist.
    ///
    /// Parameters
    /// ----------
    /// filename :
    ///     Destination path for the ``.bsp`` file.
    ///
    /// Raises
    /// ------
    /// FileExistsError
    ///     If the output file already exists.
    pub fn write(&self, filename: &str) -> PyResult<()> {
        if std::path::Path::new(filename).exists() {
            return Err(pyo3::exceptions::PyFileExistsError::new_err(format!(
                "Output file '{}' already exists. Specify a new filename.",
                filename
            )));
        }
        self.daf.write_file(filename).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!(
                "Failed to write SPK file '{}': {}",
                filename, e
            ))
        })
    }

    fn __repr__(&self) -> String {
        format!("SpkBuilder(n_segments={})", self.daf.arrays.len())
    }
}

/// Return the NORAD catalog IDs and UTC epoch ranges found in a TLE file,
/// without creating an SPK file.
///
/// Useful for inspecting a TLE file's contents before conversion.
///
/// Parameters
/// ----------
/// tle_file :
///     Path to a TLE text file.
///
/// Returns
/// -------
/// list of (norad_id, object_name, n_records)
#[pyfunction]
#[pyo3(name = "tle_file_info")]
pub fn tle_file_info_py(tle_file: &str) -> PyResult<Vec<(u64, String, usize)>> {
    let text = std::fs::read_to_string(tle_file).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!(
            "Failed to read TLE file '{}': {}",
            tle_file, e
        ))
    })?;
    Ok(parse_tle_text(&text)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        .into_iter()
        .map(|(id, elems)| {
            let name = elems
                .first()
                .and_then(|e| e.object_name.clone())
                .unwrap_or_default();
            let n = elems.len();
            (id, name, n)
        })
        .collect())
}

/// Repack an SPK file into a compact output file.
///
/// Loads the input file, fits the positions in the Equatorial J2000 frame
/// to each object, and writes the result to ``output_filename``.
///
/// The input file is loaded into the SPK singleton so that center-body chain
/// lookups work correctly.  It remains loaded after this call.
///
/// Parameters
/// ----------
/// input_filename :
///     Path to the source ``.bsp`` file to repack.
/// output_filename :
///     Destination path for the output ``.bsp`` file. Must not already exist.
/// object_ids :
///     NAIF IDs to repack. If ``None``, all objects found in the input file are
///     repacked.
/// center_id :
///     NAIF ID of the reference center body (default 10 = Sun).
/// threshold_km :
///     Maximum allowable position error in km (default 0.5).
/// degree :
///     Polynomial degree. For Type 2: 1 to 27 (default 15). For Type 13: must
///     be odd and in 1 to 27 (default 7).
/// output_type :
///     SPK segment type: 2 (Chebyshev) or 13 (Hermite). Default 2.
///
/// Returns
/// -------
/// list of (object_id, n_segments, n_records_total, max_error_km)
///     Summary information for each repacked object.
///
/// Raises
/// ------
/// FileExistsError
///     If the output file already exists.
/// ValueError
///     If no objects are found, or a repack fails for any object.
/// IOError
///     If reading the input file or writing the output file fails.
#[pyfunction]
#[pyo3(name = "repack_spk", signature = (input_filename, output_filename, object_ids=None, center_id=10, threshold_km=0.5, degree=None, output_type=2))]
pub fn repack_spk_py(
    py: Python<'_>,
    input_filename: &str,
    output_filename: &str,
    object_ids: Option<Vec<i32>>,
    center_id: i32,
    threshold_km: f64,
    degree: Option<usize>,
    output_type: i32,
) -> PyResult<Vec<(i32, usize, usize, f64)>> {
    if std::path::Path::new(output_filename).exists() {
        return Err(pyo3::exceptions::PyFileExistsError::new_err(format!(
            "Output file '{}' already exists. Specify a new filename.",
            output_filename
        )));
    }

    // Parse the input file to discover object IDs and their time ranges.
    let input_daf = DafFile::from_file(input_filename).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!(
            "Failed to read SPK file '{}': {}",
            input_filename, e
        ))
    })?;
    let mut file_ids = HashSet::new();
    let mut segment_types = HashSet::new();
    let mut object_ranges: HashMap<i32, Vec<(f64, f64)>> = HashMap::new();
    for daf_array in &input_daf.arrays {
        if !daf_array.summary_ints.is_empty() && daf_array.summary_floats.len() >= 2 {
            let oid = daf_array.summary_ints[0];
            let _ = file_ids.insert(oid);
            if daf_array.summary_ints.len() > 3 {
                let _ = segment_types.insert(daf_array.summary_ints[3]);
            }
            object_ranges
                .entry(oid)
                .or_default()
                .push((daf_array.summary_floats[0], daf_array.summary_floats[1]));
        }
    }

    // Build output comment: repack notice + original comments.
    let mut types_sorted: Vec<i32> = segment_types.into_iter().collect();
    types_sorted.sort();
    let types_str = types_sorted
        .iter()
        .map(|t| t.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let mut repack_comment = format!(
        "This kernel was repacked from the original file of type {} using Kete.",
        types_str
    );
    let orig = input_daf.comments.trim();
    if !orig.is_empty() {
        repack_comment.push_str("\nThe original un-altered comments are included below:\n\n");
        repack_comment.push_str(orig);
    }

    // Load the input file and core planetary kernels into the singleton so
    // center-chain lookups work (e.g. WISE center=399 needs de440s for
    // 399->3->0->10).  load_core() calls build_mapping() internally.
    {
        let mut singleton = LOADED_SPK
            .write()
            .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("SPK lock poisoned"))?;
        singleton.load_file(input_filename).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!(
                "Failed to load SPK file '{}': {}",
                input_filename, e
            ))
        })?;
        singleton.load_core().map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to load core kernels: {e}"))
        })?;
    }

    let ids: Vec<i32> = match object_ids {
        Some(ids) => ids,
        None => {
            let mut ids: Vec<i32> = file_ids.into_iter().collect();
            ids.sort();
            ids
        }
    };

    if ids.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "No objects found to repack.",
        ));
    }

    let spk = LOADED_SPK
        .try_read()
        .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("SPK lock poisoned"))?;

    let mut daf = DafFile::new_spk("kete repack", &repack_comment);
    let mut summary = Vec::with_capacity(ids.len());

    for &oid in &ids {
        py.check_signals()?;
        let ranges = object_ranges.get(&oid).map(|v| v.as_slice());
        let arrays = match output_type {
            2 => {
                let deg = degree.unwrap_or(15);
                repack_to_type2(&spk, oid, center_id, threshold_km, deg, ranges)
            }
            13 => {
                let deg = degree.unwrap_or(7);
                repack_to_type13(&spk, oid, center_id, threshold_km, deg, ranges)
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "output_type must be 2 or 13, got {output_type}"
                )));
            }
        }
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let n_seg = arrays.len();
        let n_rec: usize = arrays
            .iter()
            .map(|a| {
                #[allow(clippy::cast_sign_loss)]
                let n = a.daf.data[a.daf.data.len() - 1] as usize;
                n
            })
            .sum();
        summary.push((oid, n_seg, n_rec, threshold_km));
        for array in arrays {
            daf.arrays.push(array.daf);
        }
    }

    drop(spk);
    daf.write_file(output_filename).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!(
            "Failed to write SPK file '{}': {}",
            output_filename, e
        ))
    })?;

    Ok(summary)
}
