use kete_core::errors::Error;
use kete_core::frames::FrameId;
use kete_spice::prelude::LOADED_TEXT_KERNELS;
use pyo3::{FromPyObject, PyResult, pyfunction};

/// A frame given from Python as a SPICE frame ID or a frame name.
#[derive(Debug, Clone, FromPyObject)]
pub enum FrameLike {
    /// SPICE frame ID.
    Id(i32),

    /// Frame name, from the built-in frames or a loaded frames kernel.
    Name(String),
}

impl FrameLike {
    /// The SPICE frame ID, with a name looked up in the built-in frames and the
    /// loaded frames kernels.
    ///
    /// # Errors
    /// `ValueError` if no frame has the name, or the text kernels cannot be
    /// read.
    pub fn frame_id(&self) -> PyResult<FrameId> {
        match self {
            Self::Id(id) => Ok(FrameId(*id)),
            Self::Name(name) => Ok(LOADED_TEXT_KERNELS
                .try_read()
                .map_err(|_| Error::LockFailed)?
                .frame_id(name)?),
        }
    }
}

/// Remove all loaded text kernels: SCLK, frames and text PCK kernels.
#[pyfunction]
#[pyo3(name = "text_kernels_reset")]
pub fn text_kernels_reset_py() {
    LOADED_TEXT_KERNELS.write().unwrap().reset()
}
