use super::*;
use kete_core::prelude::*;
use pyo3::exceptions;
use pyo3::prelude::*;

use bincode::serde::{decode_from_std_read, encode_into_std_write};
use std::fs::File;
use std::io::{BufReader, BufWriter};

/// Polymorphic support
#[derive(FromPyObject, Debug)]
pub enum FOVListLike {
    /// Vector of allowed FOVs
    Vec(Vec<AllowedFOV>),

    /// An FOVList which is equivalent.
    FOVList(FOVList),
}

impl FOVListLike {
    /// Convert to a vector of kete core fovs.
    /// Frames will always be Equatorial
    pub fn into_sorted_vec_fov(self) -> Vec<kete_core::fov::FOV> {
        let mut fovs = match self {
            FOVListLike::Vec(v) => v,
            FOVListLike::FOVList(list) => list.0,
        };
        fovs.sort_by(|a, b| a.jd().total_cmp(&b.jd()));
        fovs.into_iter().map(|fov| fov.unwrap()).collect()
    }
}

/// A list of FOVs, which can be saved and loaded.
#[pyclass(module = "kete", sequence)]
#[derive(Debug, Clone)]
pub struct FOVList(pub Vec<AllowedFOV>);

#[pymethods]
impl FOVList {
    #[new]
    #[allow(missing_docs)]
    pub fn new(list: Vec<AllowedFOV>) -> Self {
        FOVList(list)
    }

    /// Sort the list of FOVs by their JD.
    pub fn sort(&mut self) {
        self.0.sort_by(|a, b| a.jd().total_cmp(&b.jd()))
    }

    fn __len__(&self) -> usize {
        self.0.len()
    }

    #[allow(missing_docs)]
    pub fn __getitem__(&self, mut idx: isize) -> PyResult<AllowedFOV> {
        if idx < 0 {
            idx += self.0.len() as isize;
        }
        if (idx < 0) || (idx as usize >= self.__len__()) {
            return Err(PyErr::new::<exceptions::PyIndexError, _>(
                "index out of range",
            ));
        }
        Ok(self.0[idx as usize].clone())
    }

    #[allow(missing_docs)]
    pub fn __repr__(&self) -> String {
        format!("FOVList(<{} FOVs>)", self.0.len())
    }

    /// Save the fov list into a file.
    fn save(&self, filename: String) -> PyResult<()> {
        let mut f = BufWriter::new(File::create(filename)?);
        let fovs: Vec<kete_core::fov::FOV> =
            self.0.iter().map(|fov| fov.clone().unwrap()).collect();

        let _ = encode_into_std_write(fovs, &mut f, bincode::config::legacy())
            .map_err(|_| Error::IOError("Failed to write to file".into()))?;
        Ok(())
    }

    /// Load the fov list from a file.
    #[staticmethod]
    fn load(filename: String) -> PyResult<Self> {
        let mut f = BufReader::new(File::open(filename)?);

        let fovs: Vec<kete_core::fov::FOV> =
            decode_from_std_read(&mut f, bincode::config::legacy())
                .map_err(|_| Error::IOError("Failed to read from file".into()))?;
        let fovs: Vec<AllowedFOV> = fovs.into_iter().map(|x| x.into()).collect();
        Ok(FOVList(fovs))
    }
}
