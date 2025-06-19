//! Interface for Minor Planet Center (MPC) utilities
//!
//!
use kete_core::desigs::Desig;
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
    let packed = Desig::parse_mpc_designation(desig.trim())?;
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
    let packed = Desig::parse_mpc_packed_designation(desig.trim())?;
    Ok(packed.to_string())
}

/// Given the provided partial name or integer, find the full name contained within
/// the loaded SPICE kernels.
///
/// >>> kete.spice.name_lookup("jupi")
/// ('jupiter barycenter', 5)
///
/// >>> kete.spice.name_lookup(10)
/// ('sun', 10)
///
/// If there are multiple names, but an exact match, the exact match is returned. In
/// the case of ``Earth``, there is also ``Earth Barycenter``, but asking for Earth
/// will return the exact match. Putting ``eart`` will raise an exception as there
/// are 2 partial matches.
///
/// >>> kete.spice.name_lookup("Earth")
/// ('earth', 399)
///
/// >>> kete.spice.name_lookup("Earth b")
/// ('earth barycenter', 3)
///
/// Parameters
/// ----------
/// name :
///     Name, partial name, or integer id value of the object.
///
/// Returns
/// -------
/// tuple :
///     Two elements in the tuple, the full name and the integer id value.
#[pyfunction]
#[pyo3(name = "name_lookup")]
pub fn naif_name_lookup_py(name: NaifIDLike) -> PyResult<(String, i32)> {
    Ok(name.try_into()?)
}

/// A type that can be used to represent a NAIF ID.
/// This is for polymorphic use in functions that can accept either a string or an
/// integer as a NAIF ID.
#[derive(Debug, Clone, PartialEq, Eq, FromPyObject, IntoPyObject)]
pub enum NaifIDLike {
    /// A string that can be converted to a NAIF ID.
    String(String),

    /// An integer that is a NAIF ID.
    Int(i32),
}

impl From<NaifIDLike> for Desig {
    fn from(value: NaifIDLike) -> Self {
        match value {
            NaifIDLike::String(s) => Desig::Name(s),
            NaifIDLike::Int(i) => Desig::Naif(i).try_naif_id_to_name(),
        }
    }
}

impl TryInto<(String, i32)> for NaifIDLike {
    type Error = kete_core::prelude::Error;

    #[inline(always)]
    fn try_into(self) -> Result<(String, i32), Self::Error> {
        match self {
            NaifIDLike::String(s) => {
                if s.chars().all(|c| c.is_ascii_digit()) {
                    // If the string is all digits, convert it directly to an integer.
                    if let Ok(id) = s.parse::<i32>() {
                        return Ok((kete_core::spice::try_name_from_id(id).unwrap_or(s), id));
                    }
                }
                // try the spk cache
                let mut spk = kete_core::spice::LOADED_SPK.write().unwrap();
                let id = spk.try_id_from_name(&s)?;
                Ok((id.name, id.id))
            }
            NaifIDLike::Int(i) => Ok((
                kete_core::spice::try_name_from_id(i).unwrap_or(i.to_string()),
                i,
            )),
        }
    }
}
