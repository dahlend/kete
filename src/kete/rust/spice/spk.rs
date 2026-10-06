// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-FileCopyrightText: 2025 California Institute of Technology
// SPDX-License-Identifier: BSD-3-Clause

use kete_core::constants::AU_KM;
use kete_core::desigs::Desig;
use kete_core::frames::geodetic_lat_lon_to_ecef;
use kete_spice::prelude::{DafFile, LOADED_SPK};
use kete_spice::spk::repack_to_type2;
use kete_spice::spk::repack_to_type13;
use kete_spice::spk::type10::SpkSegmentType10;
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
/// Add segments of different types one at a time, then write the completed
/// file to disk. Use this class to create new SPK files from Python.
///
/// Parameters
/// ----------
/// internal_desc : str, optional
///   Short internal description in the DAF header, at most 60 characters.
///   Default is an empty string.
/// comment : str, optional
///   Free-text comment block in the DAF file. Default is an empty string.
///
/// Examples
/// --------
/// .. code-block:: python
///
///     builder = kete.spice.SpkBuilder()
///     builder.add_tle_segment("iss_tles.txt", 399, 1)
///     builder.write("iss.bsp")
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

    /// Add Type 10 (TLE) segments from a TLE text file.
    ///
    /// The method combines all TLE entries with the same NORAD catalog number
    /// into one SPK segment. The NAIF ID of each object is ``-norad_id``.
    ///
    /// Parameters
    /// ----------
    /// tle_file : str
    ///   Path to a text file of TLEs, in 2-line or 3-line format.
    /// center_id : int
    ///   NAIF ID of the central body, for example 399 for Earth.
    /// frame_id : int
    ///   NAIF frame ID stored in each segment. kete evaluates Type 10 states in
    ///   the J2000 equatorial frame, which is frame ID 1.
    /// pad_days : float, optional
    ///   Days of coverage added before the first and after the last element set
    ///   of each segment. In the padding, the nearest element set is
    ///   propagated. An object with one element set covers only this padding.
    ///   Must be non-negative. Default is 0.5.
    ///
    /// Raises
    /// ------
    /// OSError
    ///   If the TLE file cannot be read.
    /// ValueError
    ///   If the text holds no valid TLE, or if ``pad_days`` is negative or NaN.
    #[pyo3(signature = (tle_file, center_id, frame_id, pad_days=0.5))]
    pub fn add_tle_segment(
        &mut self,
        tle_file: &str,
        center_id: i32,
        frame_id: i32,
        pad_days: f64,
    ) -> PyResult<()> {
        let text = std::fs::read_to_string(tle_file).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!(
                "Failed to read TLE file '{}': {}",
                tle_file, e
            ))
        })?;
        let arrays = SpkSegmentType10::arrays_from_tle_text(&text, center_id, frame_id, pad_days)
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

/// Repack an SPK file into a compact output file.
///
/// The function fits the positions of each object in the Equatorial J2000
/// frame. It writes the result to ``output_filename``. The coverage of each
/// object is the coverage of its segments in the input file.
///
/// The function loads the input file into the SPK singleton, where it stays
/// after the call. The fit reads the loaded kernels, which provide the
/// center-body chains. The input file loads last, so where another kernel also
/// covers an object, the fit uses the input file.
///
/// Parameters
/// ----------
/// input_filename : str
///   Path to the source ``.bsp`` file to repack.
/// output_filename : str
///   Destination path for the output ``.bsp`` file. The file must not exist.
/// object_ids : list of int, optional
///   NAIF IDs to repack. If ``None`` (default), the function repacks all
///   objects in the input file.
/// center_id : int, optional
///   NAIF ID of the center body of the output. If ``None`` (default), each
///   object uses the center of its first segment in the input file. An object
///   that is not in the input file uses 10 (Sun).
/// threshold_km : float, optional
///   Maximum position error of the output, in km. Default is 0.5.
/// degree : int, optional
///   Polynomial degree. For Type 2, 1 to 27, and ``None`` gives 15. For
///   Type 13, an odd number from 1 to 27, and ``None`` gives 7. Default is
///   ``None``.
/// output_type : int, optional
///   SPK segment type: 2 (Chebyshev) or 13 (Hermite). Default is 2.
///
/// Returns
/// -------
/// list[tuple[int, int, int, float]]
///   One tuple of (object_id, n_segments, n_records_total, threshold_km) for
///   each repacked object.
///
/// Raises
/// ------
/// FileExistsError
///   If the output file already exists.
/// ValueError
///   If no objects are found, if ``output_type`` is not 2 or 13, or if the
///   repack fails for any object.
/// OSError
///   If the input file cannot be loaded, or if the output file cannot be
///   written.
#[pyfunction]
#[pyo3(name = "repack_spk", signature = (input_filename, output_filename, object_ids=None, center_id=None, threshold_km=0.5, degree=None, output_type=2))]
#[allow(clippy::too_many_arguments)]
pub fn repack_spk_py(
    py: Python<'_>,
    input_filename: &str,
    output_filename: &str,
    object_ids: Option<Vec<i32>>,
    center_id: Option<i32>,
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
    let mut object_centers: HashMap<i32, i32> = HashMap::new();
    for daf_array in &input_daf.arrays {
        if !daf_array.summary_ints.is_empty() && daf_array.summary_floats.len() >= 2 {
            let oid = daf_array.summary_ints[0];
            let _ = file_ids.insert(oid);
            if daf_array.summary_ints.len() > 1 {
                // Use the center from the first segment seen for each object.
                let _ = object_centers
                    .entry(oid)
                    .or_insert(daf_array.summary_ints[1]);
            }
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

    // The fit reads the loaded kernels, so center chains can use any kernel the
    // user loaded. The input file loads last, so its data takes precedence over
    // every other kernel that covers the object.
    LOADED_SPK
        .write()
        .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("SPK lock poisoned"))?
        .load_file(input_filename)
        .map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!(
                "Failed to load SPK file '{}': {}",
                input_filename, e
            ))
        })?;
    let spk = LOADED_SPK
        .read()
        .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("SPK lock poisoned"))?;

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

    let mut daf = DafFile::new_spk("kete repack", &repack_comment);
    let mut summary = Vec::with_capacity(ids.len());

    for &oid in &ids {
        py.check_signals()?;
        let oid_center = center_id.unwrap_or_else(|| *object_centers.get(&oid).unwrap_or(&10));
        let ranges = object_ranges.get(&oid).map(|v| v.as_slice());
        let arrays = match output_type {
            2 => {
                let deg = degree.unwrap_or(15);
                repack_to_type2(&spk, oid, oid_center, threshold_km, deg, ranges)
            }
            13 => {
                let deg = degree.unwrap_or(7);
                repack_to_type13(&spk, oid, oid_center, threshold_km, deg, ranges)
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

    daf.write_file(output_filename).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!(
            "Failed to write SPK file '{}': {}",
            output_filename, e
        ))
    })?;

    Ok(summary)
}
