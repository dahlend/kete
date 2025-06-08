//! Interface for Minor Planet Center (MPC) utilities
//!
//!
use pyo3::prelude::*;

/// Accepts either a unpacked provisional designation or permanent designation and
/// returns the packed representation.
///
/// >>> kete.mpc.pack_designation("1998 SQ108")
/// 'J98SA8Q'
///
/// >>> kete.mpc.pack_designation("3140113")
/// '~AZaz'
///
/// Parameters
/// ----------
/// unpacked :
///     An unpacked designation to be packed into either a permanent or provisional
///     designation.
#[pyfunction]
#[pyo3(name = "pack_designation")]
pub fn pack_designation_py(desig: String) -> PyResult<String> {
    let packed = kete_core::desigs::Desig::parse_mpc_designation(desig.trim())?;
    Ok(packed.try_pack()?)
}

/// Accepts either a packed provisional designation or permanent designation and returns
/// the unpacked representation.
///
/// >>> kete.mpc.unpack_designation("J98SA8Q")
/// '1998 SQ108'
///
/// >>> kete.mpc.unpack_designation("~AZaz")
/// '3140113'
///
/// Parameters
/// ----------
/// packed :
///     A packed 5, 7, or 8 character MPC designation of an object.
#[pyfunction]
#[pyo3(name = "unpack_designation")]
pub fn unpack_designation_py(desig: String) -> PyResult<String> {
    let packed = kete_core::desigs::Desig::parse_mpc_packed_designation(desig.trim())?;
    Ok(packed.to_string())
}
