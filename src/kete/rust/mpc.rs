//! Interface for Minor Planet Center (MPC) utilities
//!
//!
use pyo3::prelude::*;

/// Pack an unpacked MPC designation into a packed format.
#[pyfunction]
#[pyo3(name = "pack_designation")]
pub fn pack_designation_py(desig: String) -> PyResult<String> {
    let packed = kete_core::desigs::Desig::parse_mpc_designation(&desig)?;
    dbg!(&packed);
    Ok(packed.try_pack()?)
}

/// Pack an packed MPC designation into a unpacked format.
#[pyfunction]
#[pyo3(name = "unpack_designation")]
pub fn unpack_designation_py(desig: String) -> PyResult<String> {
    let packed = kete_core::desigs::Desig::parse_mpc_packed_designation(&desig)?;
    Ok(packed.to_string())
}
