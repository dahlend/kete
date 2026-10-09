// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-FileCopyrightText: 2025 California Institute of Technology
// SPDX-License-Identifier: BSD-3-Clause

//! # Representation and parsing of the designations of objects.
//!
//! The Minor Planet Center (MPC) designations are used to identify
//! Comets and Asteroids. They have specific text formats that are
//! used to represent the names of these objects.
//!
//! Typically there are two broad types of designations:
//!
//!   - Permanent Designations - The orbits are very well known.
//!   - Provisional Designations - The orbits are not as well known.
//!
//! Asteroids and Comets each have their own representations of each
//! of these types of designations.
//!
//! Additionally, some asteroids are later found to be active, and are
//! reclassified as comets. In these cases they will still retain
//! their original provisional asteroid designation, but will have
//! an additional C/ or P/ etc prepended to the designation.
//!
//! The MPC also "packs" these designations into a reduced character
//! length string. For the Permanent Designations, this is 5 characters
//! for the Provisional Designations, this is 7 or 8 characters.
//!
//! The tools in this modules allow for parsing, packing, and unpacking
//! of these designations.

use std::fmt::Display;
use std::str;
use std::str::FromStr;

use crate::errors::{Error, KeteResult};
use crate::frames::{EARTH_A, ecef_to_geodetic_lat_lon};
use crate::util::partial_str_match;
use nalgebra::{Rotation3, UnitVector3, Vector3};

static MPC_HEX: &str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/// Order letters of a provisional designation, I is omitted.
static ORDER_CHARS: &str = "ABCDEFGHJKLMNOPQRSTUVWXYZ";

/// Half-month letters of a provisional designation, I is omitted and Z is unused.
static HALF_MONTH_CHARS: &str = "ABCDEFGHJKLMNOPQRSTUVWXY";

/// Largest permanent number the packed format can hold, `~zzzz`.
const MAX_PERM: u32 = 620_000 + 62 * 62 * 62 * 62 - 1;

/// Planets with permanent satellite designations: NAIF id, name, and the letter
/// used in the packed designation.
const PLANETS: [(i32, &str, char); 6] = [
    (399, "Earth", 'E'),
    (499, "Mars", 'M'),
    (599, "Jupiter", 'J'),
    (699, "Saturn", 'S'),
    (799, "Uranus", 'U'),
    (899, "Neptune", 'N'),
];

/// Designations for an object.
///
/// This enum represents all of the different types of designations
/// which kete can represent.
#[derive(Debug, Clone, PartialEq, Hash, Eq)]
#[must_use]
pub enum Desig {
    /// No id assigned.
    Empty,

    /// Asteroid Permanent ID, an integer.
    Perm(u32),

    /// Asteroid Provisional Designation
    Prov(String),

    /// Comet Permanent Designation
    /// First element is the orbit type `CPAXD`.
    /// Second is the integer designation.
    /// Third is if the comet is fragmented or not, if [`Some`] then the character
    /// is the fragment letter, if [`None`] then it is not a fragment.
    CometPerm(char, u32, Option<char>),

    /// Comet Provisional Designation
    /// First element is the orbit type `CPAXD` if available.
    /// Second is the string designation.
    /// Third is if the comet is fragmented or not, if [`Some`] then the character
    /// is the fragment letter, if [`None`] then it is not a fragment.
    CometProv(Option<char>, String, Option<char>),

    /// Planetary Satellite
    /// First element is the NAIF id of the planet,
    /// Second is the number of the satellite.
    PlanetSat(i32, u32),

    /// Text name
    Name(String),

    /// NAIF id for the object.
    /// These are used by SPICE kernels for identification.
    Naif(i32),

    /// MPC Observatory Code
    ObservatoryCode(String),
}

impl Desig {
    /// Return a full string representation of the designation, including the type.
    #[must_use]
    pub fn full_string(&self) -> String {
        format!("{self:?}")
    }

    /// Try to convert a [`Desig::Naif`] into a [`Desig::Name`] by looking it up.
    ///
    /// If unsuccessful this returns the original designation unchanged.
    pub fn try_naif_id_to_name(self) -> Self {
        if let Self::Naif(id) = &self {
            if let Some(name) = try_name_from_id(*id) {
                Self::Name(name)
            } else {
                self
            }
        } else {
            self
        }
    }

    /// Convert a [`Desig::Name`] into a [`Desig::Naif`] if possible.
    ///
    /// This will look up a NAIF id from the name if it exists.
    pub fn try_name_to_naif_id(self) -> Self {
        if let Self::Name(name) = &self {
            if let Ok(id) = name.parse::<i32>() {
                return Self::Naif(id);
            }

            let naif_ids = naif_ids_from_name(name);
            match naif_ids.as_slice() {
                [i] => return Self::Naif(i.id),
                _ => {
                    for id in &naif_ids {
                        if id.name.to_lowercase() == name.to_lowercase() {
                            return Self::Naif(id.id);
                        }
                    }
                }
            }
        }
        self
    }

    /// Convert a [`Desig::Name`] into a [`Desig::ObservatoryCode`] if possible.
    pub fn try_name_to_obs_code(self) -> Self {
        if let Self::Name(name) = &self {
            let obs_codes = try_obs_code_from_name(name);

            match obs_codes.as_slice() {
                [i] => return i.code.clone(),
                _ => {
                    for id in &obs_codes {
                        if id.name.to_lowercase() == name.to_lowercase() {
                            return id.code.clone();
                        }
                    }
                }
            }
        }
        self
    }

    /// parse an MPC unpacked designation string into a [`Desig`].
    ///
    /// ```
    ///     use kete_core::desigs::Desig;
    ///
    ///     // Asteroid permanent designations
    ///     let desig = Desig::parse_mpc_designation("123456");
    ///     assert_eq!(desig, Ok(Desig::Perm(123456)));
    ///
    ///     let desig = Desig::parse_mpc_designation("1");
    ///     assert_eq!(desig, Ok(Desig::Perm(1)));
    ///
    ///     // Comet permanent designations
    ///     let desig = Desig::parse_mpc_designation("2I");
    ///     assert_eq!(desig, Ok(Desig::CometPerm('I', 2, None)));
    ///
    ///     let desig = Desig::parse_mpc_designation("212P");
    ///     assert_eq!(desig, Ok(Desig::CometPerm('P', 212, None)));
    ///
    ///     // Asteroid provisional designations
    ///     let desig = Desig::parse_mpc_designation("2008 AA360");
    ///     assert_eq!(desig, Ok(Desig::Prov("2008 AA360".to_string())));
    ///
    ///     let desig = Desig::parse_mpc_designation("1995 XA");
    ///     assert_eq!(desig, Ok(Desig::Prov("1995 XA".to_string())));
    ///
    ///     let desig = Desig::parse_mpc_designation("4101 T-3");
    ///     assert_eq!(desig, Ok(Desig::Prov("4101 T-3".to_string())));
    ///
    ///     let desig = Desig::parse_mpc_designation("A801 AA");
    ///     assert_eq!(desig, Ok(Desig::Prov("A801 AA".to_string())));
    ///
    ///     let desig = Desig::parse_mpc_designation("2026 CZ619");
    ///     assert_eq!(desig, Ok(Desig::Prov("2026 CZ619".to_string())));
    ///
    ///     // Extended Provisional (2025-2035)
    ///     let desig = Desig::parse_mpc_designation("2026 CA620");
    ///     assert_eq!(desig, Ok(Desig::Prov("2026 CA620".to_string())));
    ///
    ///     let desig = Desig::parse_mpc_designation("2028 EA339749");
    ///     assert_eq!(desig, Ok(Desig::Prov("2028 EA339749".to_string())));
    ///
    ///     let desig = Desig::parse_mpc_designation("2026 CL591673");
    ///     assert_eq!(desig, Ok(Desig::Prov("2026 CL591673".to_string())));
    ///
    ///     let desig = Desig::parse_mpc_designation("2029 FL591673");
    ///     assert_eq!(desig, Ok(Desig::Prov("2029 FL591673".to_string())));
    ///
    ///     // Comet provisional designations
    ///     let desig = Desig::parse_mpc_designation("1996 N2");
    ///     assert_eq!(desig, Ok(Desig::CometProv(None, "1996 N2".to_string(), None)));
    ///
    ///     let desig = Desig::parse_mpc_designation("C/2020 F3");
    ///     assert_eq!(desig, Ok(Desig::CometProv(Some('C'), "2020 F3".to_string(), None)));
    ///
    ///     let desig = Desig::parse_mpc_designation("p/2005 SB216");
    ///     assert_eq!(desig, Ok(Desig::CometProv(Some('P'), "2005 SB216".to_string(), None)));
    ///
    ///     // Numbered comets, including defunct comets and fragments
    ///     let desig = Desig::parse_mpc_designation("3D");
    ///     assert_eq!(desig, Ok(Desig::CometPerm('D', 3, None)));
    ///
    ///     let desig = Desig::parse_mpc_designation("73P-B");
    ///     assert_eq!(desig, Ok(Desig::CometPerm('P', 73, Some('B'))));
    ///
    ///     let desig = Desig::parse_mpc_designation("C/2016 J1-B");
    ///     assert_eq!(desig, Ok(Desig::CometProv(Some('C'), "2016 J1".to_string(), Some('B'))));
    ///
    ///     // Planetary satellites
    ///     let desig = Desig::parse_mpc_designation("Jupiter V");
    ///     assert_eq!(desig, Ok(Desig::PlanetSat(599, 5)));
    ///
    ///     let desig = Desig::parse_mpc_designation("Neptune XI");
    ///     assert_eq!(desig, Ok(Desig::PlanetSat(899, 11)));
    ///
    ///     let desig = Desig::parse_mpc_designation("Uranus IV");
    ///     assert_eq!(desig, Ok(Desig::PlanetSat(799, 4)));
    ///
    /// ```
    ///
    /// # Errors
    ///
    /// This may fail for the following reasons:
    /// - Empty designation provided.
    /// - Parsing of the designation fails.
    pub fn parse_mpc_designation(designation: &str) -> KeteResult<Self> {
        let desig = Self::classify_mpc_designation(designation)?;

        // Check the designation against the packed format, which enforces the MPC
        // rules. A fragment of a numbered comet is valid but has no packed field,
        // so it is checked without the fragment.
        let check = if let Self::CometPerm(orbit_type, id, Some(_)) = desig {
            Self::CometPerm(orbit_type, id, None)
        } else {
            desig.clone()
        };
        let _ = check.try_pack()?;
        Ok(desig)
    }

    /// Sort an unpacked MPC designation into its type, without checking it
    /// against the MPC rules.
    fn classify_mpc_designation(designation: &str) -> KeteResult<Self> {
        let err = || Error::ValueError(format!("Invalid MPC Designation: {designation}"));
        if designation.is_empty() {
            return Err(Error::ValueError("Designation cannot be empty".to_string()));
        }
        if !designation.is_ascii() {
            return Err(err());
        }

        // All digits is a numbered minor planet.
        if designation.bytes().all(|b| b.is_ascii_digit()) {
            return Ok(Self::Perm(designation.parse().map_err(|_| err())?));
        }

        let Some((header, tail)) = designation.split_once(' ') else {
            // No space is a numbered comet, such as 1P, 3D, or the fragment 73P-B.
            let (number, fragment) = match designation.split_once('-') {
                Some((number, fragment)) => {
                    (number, Some(comet_fragment(fragment).ok_or_else(err)?))
                }
                None => (designation, None),
            };
            let orbit_type = number.chars().last().ok_or_else(err)?;
            let id = digits(&number[..number.len() - 1]).ok_or_else(err)?;
            return Ok(Self::CometPerm(orbit_type, id, fragment));
        };

        if let Some(&(naif_id, _, _)) = PLANETS.iter().find(|p| p.1 == header) {
            return Ok(Self::PlanetSat(naif_id, roman_to_int(tail)?));
        }

        // A comet type prefix, such as C/ or P/, marks a comet.
        let (orbit_type, body) = match designation.split_once('/') {
            Some((prefix, body)) => {
                let mut chars = prefix.chars();
                let (Some(orbit_type), None) = (chars.next(), chars.next()) else {
                    return Err(err());
                };
                (Some(orbit_type.to_ascii_uppercase()), body)
            }
            None => (None, designation),
        };
        if orbit_type == Some('S') {
            return Err(Error::ValueError(format!(
                "Provisional natural satellite designations are not supported: {designation}"
            )));
        }

        let (year, tail) = body.split_once(' ').ok_or_else(err)?;
        if tail.as_bytes().get(1).is_some_and(u8::is_ascii_digit) {
            // A comet designation, such as 1995 A1 or 1994 P1-B.
            let (tail, fragment) = match tail.split_once('-') {
                Some((tail, fragment)) => (tail, Some(comet_fragment(fragment).ok_or_else(err)?)),
                None => (tail, None),
            };
            Ok(Self::CometProv(
                orbit_type,
                format!("{year} {tail}"),
                fragment,
            ))
        } else {
            // A minor planet designation, which a comet keeps when it was first
            // designated as a minor planet.
            Ok(with_orbit_type(orbit_type, body.to_string()))
        }
    }

    /// Pack the designation into the MPC Packed format.
    /// ```
    ///    use kete_core::desigs::Desig;
    ///
    ///    // Asteroid permanent designations
    ///    let packed = Desig::Perm(123456).try_pack();
    ///    assert_eq!(packed, Ok("C3456".to_string()));
    ///
    ///    let packed = Desig::Perm(619999).try_pack();
    ///    assert_eq!(packed, Ok("z9999".to_string()));
    ///
    ///    let packed = Desig::Perm(15396335).try_pack();
    ///    assert_eq!(packed, Ok("~zzzz".to_string()));
    ///
    ///    let packed = Desig::Perm(620028).try_pack();
    ///    assert_eq!(packed, Ok("~000S".to_string()));
    ///
    ///    // Asteroid Provisional Designations
    ///    let packed = Desig::Prov("1995 XA".to_string()).try_pack();
    ///    assert_eq!(packed, Ok("J95X00A".to_string()));
    ///
    ///    let packed = Desig::Prov("1995 XL1".to_string()).try_pack();
    ///    assert_eq!(packed, Ok("J95X01L".to_string()));
    ///
    ///    let packed = Desig::Prov("1998 SS162".to_string()).try_pack();
    ///    assert_eq!(packed, Ok("J98SG2S".to_string()));
    ///
    ///    let packed = Desig::Prov("2099 AZ193".to_string()).try_pack();
    ///    assert_eq!(packed, Ok("K99AJ3Z".to_string()));
    ///
    ///    let packed = Desig::Prov("2016 JB1".to_string()).try_pack();
    ///    assert_eq!(packed, Ok("K16J01B".to_string()));
    ///
    ///    let packed = Desig::Prov("2016 JB1".to_string()).try_pack();
    ///    assert_eq!(packed, Ok("K16J01B".to_string()));
    ///
    ///    let packed = Desig::Prov("2026 CZ619".to_string()).try_pack();
    ///    assert_eq!(packed, Ok("K26Cz9Z".to_string()));
    ///
    ///    // Extended Provisional (2025-2035)
    ///    let packed = Desig::Prov("2026 CA620".to_string()).try_pack();
    ///    assert_eq!(packed, Ok("_QC0000".to_string()));
    ///
    ///    let packed = Desig::Prov("2028 EA339749".to_string()).try_pack();
    ///    assert_eq!(packed, Ok("_SEZZZZ".to_string()));
    ///
    ///    let packed = Desig::Prov("2026 CL591673".to_string()).try_pack();
    ///    assert_eq!(packed, Ok("_QCzzzz".to_string()));
    ///
    ///    let packed = Desig::Prov("2029 FL591673".to_string()).try_pack();
    ///    assert_eq!(packed, Ok("_TFzzzz".to_string()));
    ///
    ///    // Surveys
    ///    let packed = Desig::Prov("2040 P-L".to_string()).try_pack();
    ///    assert_eq!(packed, Ok("PLS2040".to_string()));
    ///
    ///    let packed = Desig::Prov("1010 T-2".to_string()).try_pack();
    ///    assert_eq!(packed, Ok("T2S1010".to_string()));
    ///
    ///    // Pre-1925 designations
    ///    let packed = Desig::Prov("A801 AA".to_string()).try_pack();
    ///    assert_eq!(packed, Ok("I01A00A".to_string()));
    ///
    ///    // Comet permanent designations
    ///    let packed = Desig::CometPerm('I', 2, None).try_pack();
    ///    assert_eq!(packed, Ok("0002I".to_string()));
    ///
    ///    let packed = Desig::CometPerm('P', 212, None).try_pack();
    ///    assert_eq!(packed, Ok("0212P".to_string()));
    ///
    ///    // Comet provisional designations
    ///    let packed = Desig::CometProv(Some('D'), "1918 W1".to_string(), None).try_pack();
    ///    assert_eq!(packed, Ok("DJ18W010".to_string()));
    ///
    ///    let packed = Desig::CometProv(Some('P'), "2005 SB216".to_string(), None).try_pack();
    ///    assert_eq!(packed, Ok("PK05SL6B".to_string()));
    ///
    ///    let packed = Desig::CometProv(None, "2005 SB216".to_string(), None).try_pack();
    ///    assert_eq!(packed, Ok("K05SL6B".to_string()));
    ///
    ///    let packed = Desig::CometProv(None, "2016 J1".to_string(), Some('B')).try_pack();
    ///    assert_eq!(packed, Ok("K16J01b".to_string()));
    ///
    ///    // Planetary satellites
    ///    let packed = Desig::PlanetSat(599, 5).try_pack();
    ///    assert_eq!(packed, Ok("J005S".to_string()));
    ///
    ///    let packed = Desig::PlanetSat(699, 19).try_pack();
    ///    assert_eq!(packed, Ok("S019S".to_string()));
    ///
    ///    let packed = Desig::PlanetSat(799, 4).try_pack();
    ///    assert_eq!(packed, Ok("U004S".to_string()));
    /// ```
    ///
    /// # Errors
    ///
    /// This may fail for the following reasons:
    /// - Empty designation provided.
    /// - Parsing of the designation fails.
    pub fn try_pack(&self) -> KeteResult<String> {
        let err = || {
            Error::ValueError(format!(
                "Not a valid MPC designation: {}",
                self.full_string()
            ))
        };
        match self {
            Self::Empty | Self::Name(_) | Self::Naif(_) => Err(Error::ValueError(format!(
                "Only MPC designations can be packed: {}",
                self.full_string()
            ))),
            Self::ObservatoryCode(code) => {
                if code.len() == 3 && code.is_ascii() {
                    Ok(code.clone())
                } else {
                    Err(err())
                }
            }
            Self::Perm(num) => match num {
                1..620_000 => Ok(format!("{}{:04}", mpc_hex_char(num / 10_000), num % 10_000)),
                620_000..=MAX_PERM => Ok(format!("~{:0>4}", num_to_mpc_hex(num - 620_000))),
                _ => Err(Error::ValueError(format!(
                    "MPC Permanent Designation out of range: {num}"
                ))),
            },
            Self::CometPerm(orbit_type, id, fragment) => {
                if fragment.is_some() {
                    return Err(Error::ValueError(format!(
                        "The packed MPC Comet Permanent Designation has no field for a \
                         fragment: {self}"
                    )));
                }
                if "PDI".contains(*orbit_type) && (1..=9999).contains(id) {
                    Ok(format!("{id:04}{orbit_type}"))
                } else {
                    Err(err())
                }
            }
            Self::PlanetSat(naif_id, num) => {
                let letter = PLANETS.iter().find(|p| p.0 == *naif_id).map(|p| p.2);
                match letter {
                    Some(letter) if (1..=999).contains(num) => Ok(format!("{letter}{num:03}S")),
                    _ => Err(err()),
                }
            }
            Self::Prov(des) => pack_prov(des),
            Self::CometProv(orbit_type, des, fragment) => {
                let prefix = match orbit_type.map(|o| o.to_ascii_uppercase()) {
                    None => String::new(),
                    Some(o) if "PCDXAI".contains(o) => o.to_string(),
                    Some(_) => return Err(err()),
                };
                let (_, tail) = des.split_once(' ').ok_or_else(err)?;
                if tail.as_bytes().get(1).is_some_and(u8::is_ascii_digit) {
                    Ok(format!("{prefix}{}", pack_comet_prov(des, *fragment)?))
                } else if fragment.is_none() {
                    // A minor planet designation of an object redesignated as a
                    // comet, which has no field for a fragment.
                    Ok(format!("{prefix}{}", pack_prov(des)?))
                } else {
                    Err(err())
                }
            }
        }
    }

    /// Unpacked a MPC packed designation into a [`Desig`] enum.
    ///
    /// # Errors
    /// Returns error if parsing fails.
    pub fn parse_mpc_packed_designation(packed: &str) -> KeteResult<Self> {
        if !packed.is_ascii() {
            return Err(Error::ValueError(format!(
                "Invalid MPC packed designation: {packed}"
            )));
        }
        if packed.len() == 5 {
            unpack_perm_designation(packed)
        } else if packed.len() >= 7 && packed.len() <= 8 {
            unpack_prov_designation(packed)
        } else {
            Err(Error::ValueError(format!(
                "Invalid MPC designation length: {}",
                packed.len()
            )))
        }
    }

    /// Try to extract a NAIF ID from this designation.
    ///
    /// Returns `Some(id)` for [`Desig::Naif`], or for [`Desig::Name`] if the name
    /// resolves to exactly one NAIF ID. Returns `None` for all other variants.
    #[must_use]
    pub fn naif_id(self) -> Option<i32> {
        match self {
            Self::Naif(id) => Some(id),
            Self::Name(name) => {
                let resolved = Self::Name(name).try_name_to_naif_id();
                if let Self::Naif(id) = resolved {
                    Some(id)
                } else {
                    None
                }
            }
            Self::Empty
            | Self::Perm(_)
            | Self::Prov(_)
            | Self::CometPerm(..)
            | Self::CometProv(..)
            | Self::PlanetSat(..)
            | Self::ObservatoryCode(_) => None,
        }
    }
}

impl From<Option<i32>> for Desig {
    fn from(value: Option<i32>) -> Self {
        match value {
            Some(id) => Self::Naif(id),
            None => Self::Empty,
        }
    }
}

impl From<i32> for Desig {
    fn from(value: i32) -> Self {
        Self::Naif(value)
    }
}

impl Display for Desig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&match self {
            Self::Empty => "None".to_string(),
            Self::Prov(s) | Self::Name(s) | Self::ObservatoryCode(s) => s.clone(),
            Self::Perm(i) => i.to_string(),
            Self::Naif(i) => i.to_string(),
            Self::CometPerm(orbit_type, id, fragment) => {
                let frag_str = fragment.map_or(String::new(), |x| format!("-{x}"));
                format!("{id}{orbit_type}{frag_str}")
            }
            Self::CometProv(orbit_type, id, fragment) => {
                let orbit_str = orbit_type.map_or(String::new(), |o| o.to_string() + "/");
                let frag_str = fragment.map_or(String::new(), |x| "-".to_string() + &x.to_string());
                format!("{orbit_str}{id}{frag_str}")
            }
            Self::PlanetSat(naif_id, sat_num) => {
                let roman = int_to_roman(*sat_num).unwrap_or_else(|_| sat_num.to_string());
                let planet = PLANETS
                    .iter()
                    .find(|p| p.0 == *naif_id)
                    .map_or_else(|| naif_id.to_string(), |p| p.1.to_string());
                format!("{planet} {roman}")
            }
        })
    }
}

/// Base-62 character of a value below 62.
fn mpc_hex_char(value: u32) -> char {
    char::from(MPC_HEX.as_bytes()[value as usize])
}

/// Value of a base-62 character.
fn mpc_hex_value(c: char) -> Option<u32> {
    MPC_HEX.find(c).and_then(|x| u32::try_from(x).ok())
}

/// Value of a non-empty string of ASCII digits.
fn digits(text: &str) -> Option<u32> {
    if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    text.parse().ok()
}

/// Value of a number within a designation, which has no leading zero.
fn designation_number(text: &str) -> Option<u32> {
    if text.starts_with('0') {
        return None;
    }
    digits(text)
}

/// A comet fragment, a single letter.
fn comet_fragment(text: &str) -> Option<char> {
    let mut chars = text.chars();
    match (chars.next(), chars.next()) {
        (Some(f), None) if f.is_ascii_alphabetic() => Some(f.to_ascii_uppercase()),
        _ => None,
    }
}

/// Year of a provisional designation, where a leading A replaces the 1 of the
/// years before 1925.
fn prov_year(year: &str) -> Option<u32> {
    if year.len() != 4 {
        return None;
    }
    match year.strip_prefix('A') {
        Some(rest) => digits(rest).map(|y| y + 1000).filter(|&y| y < 1925),
        None => digits(year),
    }
}

/// A minor planet provisional designation, as a comet if it has an orbit type.
fn with_orbit_type(orbit_type: Option<char>, designation: String) -> Desig {
    match orbit_type {
        Some(_) => Desig::CometProv(orbit_type, designation, None),
        None => Desig::Prov(designation),
    }
}

/// Pack a minor planet provisional designation, such as 1995 XL1.
fn pack_prov(des: &str) -> KeteResult<String> {
    let err = || Error::ValueError(format!("Invalid MPC Provisional Designation: {des}"));
    let (year, tail) = des.split_once(' ').ok_or_else(err)?;

    if matches!(tail, "P-L" | "T-1" | "T-2" | "T-3") {
        // Survey designations hold a 4 digit number within the survey.
        if year.len() != 4 || digits(year).is_none() {
            return Err(err());
        }
        return Ok(format!("{}{}S{year}", &tail[0..1], &tail[2..3]));
    }

    let year = prov_year(year).ok_or_else(err)?;
    let mut chars = tail.chars();
    let (Some(half_month), Some(order)) = (chars.next(), chars.next()) else {
        return Err(err());
    };
    if !HALF_MONTH_CHARS.contains(half_month) || !ORDER_CHARS.contains(order) {
        return Err(err());
    }
    let cycle = match chars.as_str() {
        "" => 0,
        cycle => designation_number(cycle).ok_or_else(err)?,
    };

    if cycle < 620 {
        if !(1800..2100).contains(&year) {
            return Err(err());
        }
        // Unlike the order letters, the cycle count letters include I.
        Ok(format!(
            "{}{:02}{half_month}{}{}{order}",
            mpc_hex_char(year / 100),
            year % 100,
            mpc_hex_char(cycle / 10),
            cycle % 10
        ))
    } else {
        // Extended format, for more than 15,500 designations in a half-month,
        // defined for the years 2010 through 2035.
        if !(2010..=2035).contains(&year) {
            return Err(Error::ValueError(format!(
                "The extended packed MPC Provisional Designation is only defined for \
                 2010 through 2035: {des}"
            )));
        }
        // ORDER_CHARS counts A = 0, B = 1 ... skipping I.
        let order = ORDER_CHARS
            .find(order)
            .and_then(|x| u32::try_from(x).ok())
            .ok_or_else(err)?;
        let count = cycle
            .checked_mul(25)
            .and_then(|x| x.checked_add(order))
            .map(|x| x - 15_500)
            .filter(|&x| x < 62_u32.pow(4))
            .ok_or_else(|| {
                Error::ValueError(format!(
                    "MPC Provisional Designation too large to pack: {des}"
                ))
            })?;
        Ok(format!(
            "_{}{half_month}{:0>4}",
            mpc_hex_char(year - 2000),
            num_to_mpc_hex(count)
        ))
    }
}

/// Pack a comet provisional designation, such as 2033 L89, without its orbit type.
fn pack_comet_prov(des: &str, fragment: Option<char>) -> KeteResult<String> {
    let err = || Error::ValueError(format!("Invalid MPC Comet Provisional Designation: {des}"));
    let (year, tail) = des.split_once(' ').ok_or_else(err)?;
    // Comets keep provisional designations from historical apparitions, so any
    // four-digit year before 2100 packs. Minor planet designations start at 1800.
    let year = digits(year)
        .filter(|y| year.len() == 4 && *y < 2100)
        .ok_or_else(err)?;
    let mut chars = tail.chars();
    let half_month = chars
        .next()
        .filter(|c| HALF_MONTH_CHARS.contains(*c))
        .ok_or_else(err)?;
    // There is no extended packed format for comets.
    let num = designation_number(chars.as_str())
        .filter(|n| (1..620).contains(n))
        .ok_or_else(err)?;
    let fragment = match fragment {
        None => '0',
        Some(f) if f.is_ascii_alphabetic() => f.to_ascii_lowercase(),
        Some(_) => return Err(err()),
    };
    Ok(format!(
        "{}{:02}{half_month}{}{}{fragment}",
        mpc_hex_char(year / 100),
        year % 100,
        mpc_hex_char(num / 10),
        num % 10
    ))
}

/// Convert a u64 number to a string representation in the MPC hexadecimal format.
/// ```
///     use kete_core::desigs::num_to_mpc_hex;
///
///     let hex_str = num_to_mpc_hex(63);
///     assert_eq!(hex_str, "11");
/// ```
#[must_use]
pub fn num_to_mpc_hex(mut num: u32) -> String {
    if num == 0 {
        return "0".to_string();
    }
    let mut result = Vec::new();
    while num > 0 {
        result.push(mpc_hex_char(num % 62));
        num /= 62;
    }
    result.iter().rev().collect()
}

/// Convert a string in the MPC hexadecimal format to a u64 number.
///
/// ```
///    use kete_core::desigs::mpc_hex_to_num;
///    let num = mpc_hex_to_num("00011");
///    assert_eq!(num, Ok(63));
///
///    let largest = mpc_hex_to_num("zzzzz");
///    assert_eq!(largest, Ok(916132831));
/// ```
///
/// # Errors
/// Fails when input string contains invalid characters, or the value is too large.
pub fn mpc_hex_to_num(hex: &str) -> KeteResult<u32> {
    let mut result = 0_u32;
    for c in hex.chars() {
        let value = mpc_hex_value(c).ok_or_else(|| {
            Error::ValueError(format!("Invalid character in MPC hexadecimal string: {c}"))
        })?;
        result = result
            .checked_mul(62)
            .and_then(|x| x.checked_add(value))
            .ok_or_else(|| {
                Error::ValueError(format!("MPC hexadecimal string is too large: {hex}"))
            })?;
    }
    Ok(result)
}

/// Unpack the 5 character MPC Permanent Designation.
///
/// ```
///     use kete_core::desigs::{unpack_perm_designation, Desig};
///     use kete_core::errors::KeteResult;
///
///     let desig = unpack_perm_designation("C3456").unwrap();
///     assert_eq!(desig, Desig::Perm(123456));
///
///     let desig = unpack_perm_designation("z9999");
///     assert_eq!(desig, Ok(Desig::Perm(619999)));
///
///     let desig = unpack_perm_designation("~zzzz");
///     assert_eq!(desig, Ok(Desig::Perm(15396335)));
///
///     let desig = unpack_perm_designation("~000S");
///     assert_eq!(desig, Ok(Desig::Perm(620028)));
///
///     let desig = unpack_perm_designation("0002I");
///     assert_eq!(desig, Ok(Desig::CometPerm('I', 2, None)));
///
///     let desig = unpack_perm_designation("0212P");
///     assert_eq!(desig, Ok(Desig::CometPerm('P', 212, None)));
///
///     let desig = unpack_perm_designation("J005S");
///     assert_eq!(desig, Ok(Desig::PlanetSat(599, 5)));
///
///     let desig = unpack_perm_designation("S019S");
///     assert_eq!(desig, Ok(Desig::PlanetSat(699, 19)));
///
///     let desig = unpack_perm_designation("U004S");
///     assert_eq!(desig, Ok(Desig::PlanetSat(799, 4)));
///
///     let desig = unpack_perm_designation("N011S");
///     assert_eq!(desig, Ok(Desig::PlanetSat(899, 11)));
/// ```
///
/// # Errors
/// Fails when input string contains invalid characters.
///
pub fn unpack_perm_designation(designation: &str) -> KeteResult<Desig> {
    let desig = decode_perm(designation).ok_or_else(|| {
        Error::ValueError(format!("Invalid MPC Permanent Designation: {designation}"))
    })?;
    // Packing checks the values against the MPC rules.
    let _ = desig.try_pack()?;
    Ok(desig)
}

/// Decode a 5 character packed permanent designation, without checking the
/// values against the MPC rules.
fn decode_perm(packed: &str) -> Option<Desig> {
    if !packed.is_ascii() || packed.len() != 5 {
        return None;
    }
    let first = packed.chars().next()?;
    let last = packed.chars().last()?;
    if first == '~' {
        // Numbered minor planets from 620,000.
        return Some(Desig::Perm(mpc_hex_to_num(&packed[1..]).ok()? + 620_000));
    }
    match last {
        'S' => {
            let naif_id = PLANETS.iter().find(|p| p.2 == first)?.0;
            Some(Desig::PlanetSat(naif_id, digits(&packed[1..4])?))
        }
        '0'..='9' => {
            let num = mpc_hex_value(first)? * 10_000 + digits(&packed[1..])?;
            Some(Desig::Perm(num))
        }
        _ => Some(Desig::CometPerm(last, digits(&packed[..4])?, None)),
    }
}

/// Unpack a provisional designation.
///
/// ```
///     use kete_core::desigs::{unpack_prov_designation, Desig};
///
///     // Comet Provisional Designations
///     let desig = unpack_prov_designation("CI70Q010").unwrap();
///     assert_eq!(desig, Desig::CometProv(Some('C'), "1870 Q1".to_string(), None));
///
///     let desig = unpack_prov_designation("pK05SL6B").unwrap();
///     assert_eq!(desig, Desig::CometProv(Some('P'), "2005 SB216".to_string(), None));
///
///     let desig = unpack_prov_designation("I70Q01a").unwrap();
///     assert_eq!(desig, Desig::CometProv(None, "1870 Q1".to_string(), Some('A')));
///
///     let desig = unpack_prov_designation("K16J01b").unwrap();
///     assert_eq!(desig, Desig::CometProv(None, "2016 J1".to_string(), Some('B')));
///
///     let desig = unpack_prov_designation("PK05SL6B").unwrap();
///     assert_eq!(desig, Desig::CometProv(Some('P'), "2005 SB216".to_string(), None));
///
///     let desig = unpack_prov_designation("K33L89c").unwrap();
///     assert_eq!(desig, Desig::CometProv(None, "2033 L89".to_string(), Some('C')));
///
///     // Asteroid Provisional Designations
///     let desig = unpack_prov_designation("J95X00A").unwrap();
///     assert_eq!(desig, Desig::Prov("1995 XA".to_string()));
///
///     let desig = unpack_prov_designation("J95X01L").unwrap();
///     assert_eq!(desig, Desig::Prov("1995 XL1".to_string()));
///
///     let desig = unpack_prov_designation("J98SG2S").unwrap();
///     assert_eq!(desig, Desig::Prov("1998 SS162".to_string()));
///
///     let desig = unpack_prov_designation("K99AJ3Z").unwrap();
///     assert_eq!(desig, Desig::Prov("2099 AZ193".to_string()));
///
///     let desig = unpack_prov_designation("K16J01B").unwrap();
///     assert_eq!(desig, Desig::Prov("2016 JB1".to_string()));
///
///     // Extended Provisional Designations
///     let desig = unpack_prov_designation("_SEZZZZ").unwrap();
///     assert_eq!(desig, Desig::Prov("2028 EA339749".to_string()));
///
///     let desig = unpack_prov_designation("_TFzzzz").unwrap();
///     assert_eq!(desig, Desig::Prov("2029 FL591673".to_string()));
///
///     let desig = unpack_prov_designation("_RD0aEM").unwrap();
///     assert_eq!(desig, Desig::Prov("2027 DZ6190".to_string()));
///
///     // pre 1925
///     let desig = unpack_prov_designation("I01A00A").unwrap();
///     assert_eq!(desig, Desig::Prov("A801 AA".to_string()));
///
///     // Survey designations
///     let desig = unpack_prov_designation("PLS2040").unwrap();
///     assert_eq!(desig, Desig::Prov("2040 P-L".to_string()));
///
///     let desig = unpack_prov_designation("T1S3138").unwrap();
///     assert_eq!(desig, Desig::Prov("3138 T-1".to_string()));
///
///     let desig = unpack_prov_designation("T2S1010").unwrap();
///     assert_eq!(desig, Desig::Prov("1010 T-2".to_string()));
///
///     let desig = unpack_prov_designation("T3S4101").unwrap();
///     assert_eq!(desig, Desig::Prov("4101 T-3".to_string()));
/// ```
///
/// # Errors
/// Fails when input string contains invalid characters.
///
pub fn unpack_prov_designation(designation: &str) -> KeteResult<Desig> {
    let desig = decode_prov(designation).ok_or_else(|| {
        Error::ValueError(format!(
            "Invalid MPC Provisional Designation: {designation}"
        ))
    })?;
    // Packing checks the values against the MPC rules.
    let _ = desig.try_pack()?;
    Ok(desig)
}

/// Decode a 7 or 8 character packed provisional designation, without checking
/// the values against the MPC rules.
///
/// Note that some comets were originally designated as minor planets, so they
/// may have a minor planet provisional designation with a comet orbit type in
/// front.
fn decode_prov(packed: &str) -> Option<Desig> {
    if !packed.is_ascii() || !(7..=8).contains(&packed.len()) {
        return None;
    }

    // Survey designations, such as T1S3138 for 3138 T-1. The third character of
    // any other designation is a digit of the year.
    if packed.len() == 7 && matches!(&packed[..3], "PLS" | "T1S" | "T2S" | "T3S") {
        return Some(Desig::Prov(format!(
            "{} {}-{}",
            &packed[3..],
            &packed[0..1],
            &packed[1..2]
        )));
    }

    let (orbit_type, body) = match packed.len() {
        8 => (
            Some(packed.chars().next()?.to_ascii_uppercase()),
            &packed[1..],
        ),
        _ => (None, packed),
    };
    let body: Vec<char> = body.chars().collect();

    if body[0] == '_' {
        // Extended format: a capital letter for the years 2010 through 2035, the
        // half-month, and 4 base-62 characters holding the order of designation
        // after the first 15,500.
        if !body[1].is_ascii_uppercase() {
            return None;
        }
        let year = 2000 + mpc_hex_value(body[1])?;
        let half_month = body[2];
        let total = mpc_hex_to_num(&body[3..].iter().collect::<String>()).ok()? + 15_500;
        let order = char::from(ORDER_CHARS.as_bytes()[(total % 25) as usize]);
        let des = format!("{year} {half_month}{order}{}", total / 25);
        return Some(with_orbit_type(orbit_type, des));
    }

    let century = mpc_hex_value(body[0]).filter(|c| *c <= 20)?;
    let year = century * 100 + digits(&body[1..3].iter().collect::<String>())?;
    let half_month = body[3];
    let number = mpc_hex_value(body[4])? * 10 + body[5].to_digit(10)?;
    let last = body[6];
    let comet = last == '0' || last.is_ascii_lowercase();
    // Comet designations reach back before 1800, minor planet designations do not.
    if !comet && century < 18 {
        return None;
    }

    if comet {
        // Comet format, the last character is '0' or the fragment.
        let fragment = (last != '0').then(|| last.to_ascii_uppercase());
        let des = format!("{year:04} {half_month}{number}");
        Some(Desig::CometProv(orbit_type, des, fragment))
    } else {
        // Minor planet format, the last character is the order letter.
        let year = if year < 1925 {
            format!("A{}", year - 1000)
        } else {
            year.to_string()
        };
        let cycle = if number == 0 {
            String::new()
        } else {
            number.to_string()
        };
        Some(with_orbit_type(
            orbit_type,
            format!("{year} {half_month}{last}{cycle}"),
        ))
    }
}

static ROMAN_PAIRS: [(u32, &str); 13] = [
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
];

/// Convert an integer to a roman numeral string.
///
/// ```
///     use kete_core::desigs::int_to_roman;
///     use kete_core::errors::Error;
///
///     let roman = int_to_roman(1994);
///     assert_eq!(roman, Ok("MCMXCIV".to_string()));
///
///     let roman = int_to_roman(3999);
///     assert_eq!(roman, Ok("MMMCMXCIX".to_string()));
///
///     let roman = int_to_roman(4);
///     assert_eq!(roman, Ok("IV".to_string()));
///
///     let roman = int_to_roman(42);
///     assert_eq!(roman, Ok("XLII".to_string()));
///
///     let roman = int_to_roman(42);
///     assert_eq!(roman, Ok("XLII".to_string()));
///
///     let roman = int_to_roman(0);
///     assert_eq!(roman, Err(Error::ValueError("Number must be between 1 and 3999".into())));
/// ```
///
/// # Errors
/// Fails when input is either 0 or greater than 3999.
///
pub fn int_to_roman(mut num: u32) -> KeteResult<String> {
    if num > 3999 || num == 0 {
        return Err(Error::ValueError(
            "Number must be between 1 and 3999".into(),
        ));
    }
    let mut result = String::new();
    for &(value, symbol) in &ROMAN_PAIRS {
        while num >= value {
            result.push_str(symbol);
            num -= value;
        }
    }
    Ok(result)
}

/// Convert a roman numeral string to an integer.
/// ```
///    use kete_core::desigs::roman_to_int;
///
///    let num = roman_to_int("IV");
///    assert_eq!(num, Ok(4));
///
///    let num = roman_to_int("MCMXCIV");
///    assert_eq!(num, Ok(1994));
///
///    let num = roman_to_int("MMMCMXCIX");
///    assert_eq!(num, Ok(3999));
///
///    let num = roman_to_int("XLII");
///    assert_eq!(num, Ok(42));
///
///    let num = roman_to_int("XXXXX");
///    assert!(num.is_err());
/// ```
/// # Errors
/// Fails when input contains invalid characters.
///
pub fn roman_to_int(roman: &str) -> KeteResult<u32> {
    let mut result = 0;
    let mut last_value = 4000;

    for character in roman.chars() {
        let character = character.to_string();
        let val = ROMAN_PAIRS
            .iter()
            .find(|&&(_, symbol)| symbol == character)
            .ok_or(Error::ValueError(format!(
                "Invalid character in roman numeral: {character}"
            )))?
            .0;
        if val > last_value {
            // Subtract the last value if the current is larger (e.g., IV)
            result += val - 2 * last_value;
        } else {
            result += val;
        }
        last_value = val;
    }

    // Validate the result is a valid roman numeral
    if int_to_roman(result) != Ok(roman.to_string()) {
        return Err(Error::ValueError(format!(
            "Invalid roman numeral: {roman} {result}"
        )));
    }
    Ok(result)
}

/// NAIF ID information
#[derive(Debug, Clone)]
pub struct NaifId {
    /// NAIF id
    pub id: i32,

    /// name of the object
    pub name: String,
}

impl FromStr for NaifId {
    type Err = Error;

    /// Load an [`NaifId`] from a single string.
    fn from_str(row: &str) -> KeteResult<Self> {
        let id = i32::from_str(row[0..10].trim()).unwrap();
        let name = row[11..].trim().to_string();
        Ok(Self { id, name })
    }
}

const PRELOAD_IDS: &[u8] = include_bytes!("../data/naif_ids.csv");

static NAIF_IDS: std::sync::LazyLock<Box<[NaifId]>> = std::sync::LazyLock::new(|| {
    let mut ids = Vec::new();
    let text = str::from_utf8(PRELOAD_IDS).unwrap().split('\n');
    for row in text.skip(1) {
        ids.push(NaifId::from_str(row).unwrap());
    }
    ids.into()
});

/// Return the string name of the desired ID if possible.
pub fn try_name_from_id(id: i32) -> Option<String> {
    for naif_id in NAIF_IDS.iter() {
        if naif_id.id == id {
            return Some(naif_id.name.clone());
        }
    }
    None
}

/// Try to find a NAIF id from a name.
///
/// This will return all matching IDs for the given name.
///
/// This does a partial string match, case insensitive.
pub fn naif_ids_from_name(name: &str) -> Vec<NaifId> {
    let desigs: Vec<&str> = NAIF_IDS.iter().map(|n| n.name.as_str()).collect();
    partial_str_match(name, &desigs)
        .into_iter()
        .map(|(i, _)| NAIF_IDS[i].clone())
        .collect()
}

/// Observatory information
#[derive(Debug, Clone)]
pub struct ObsCode {
    /// observatory code
    pub code: Desig,

    /// longitude in degrees
    pub lon: f64,

    /// latitude in degrees
    pub lat: f64,

    /// altitude above the WGS84 ellipsoid in km
    pub altitude: f64,

    /// name of the observatory
    pub name: String,
}

impl FromStr for ObsCode {
    type Err = Error;

    /// Load an [`ObsCode`] from a single string.
    fn from_str(row: &str) -> KeteResult<Self> {
        let code = row[0..3].to_string();
        // spacecraft have a code, but no location, so we allow for NaN values here
        let rec_lon = f64::from_str(row[3..13].trim()).unwrap_or(f64::NAN);
        let cos = f64::from_str(row[13..21].trim()).unwrap_or(f64::NAN);
        let sin = f64::from_str(row[21..30].trim()).unwrap_or(f64::NAN);
        let vec = Vector3::new(cos, 0.0, sin) * EARTH_A;

        let rotation = Rotation3::from_axis_angle(
            &UnitVector3::new_normalize([0.0, 0.0, 1.0].into()),
            rec_lon.to_radians(),
        );
        let vec = rotation.transform_vector(&vec);

        let (lat, lon, altitude) = ecef_to_geodetic_lat_lon(vec.x, vec.y, vec.z);

        let name = row[30..].trim().to_string();
        Ok(Self {
            code: Desig::ObservatoryCode(code),
            lon: lon.to_degrees(),
            lat: lat.to_degrees(),
            altitude,
            name,
        })
    }
}

const PRELOAD_OBS: &[u8] = include_bytes!("../data/mpc_obs.tsv");

/// Observatory Codes
pub static OBS_CODES: std::sync::LazyLock<Vec<ObsCode>> = std::sync::LazyLock::new(|| {
    let mut codes = Vec::new();
    let text = str::from_utf8(PRELOAD_OBS).unwrap().split('\n');
    for row in text.skip(1) {
        if let Ok(code) = ObsCode::from_str(row) {
            codes.push(code);
        }
    }
    codes
});

/// Return all possible observatory code matches for a given name.
///
/// This does a case insensitive partial match on the observatory names.
///
/// This first checks the names of the observatories, then checks the codes
/// for matches.
///
/// If multiple matches are found, all of them are returned.
pub fn try_obs_code_from_name(name: &str) -> Vec<ObsCode> {
    let desigs: Vec<&str> = OBS_CODES.iter().map(|n| n.name.as_str()).collect();
    let codes: Vec<String> = OBS_CODES.iter().map(|n| n.code.to_string()).collect();
    let mut matches: Vec<_> = partial_str_match(name, &desigs)
        .into_iter()
        .map(|(i, _)| OBS_CODES[i].clone())
        .collect();
    matches.extend(
        partial_str_match(name, &codes.iter().map(String::as_str).collect::<Vec<_>>())
            .into_iter()
            .map(|(i, _)| OBS_CODES[i].clone()),
    );
    matches
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn desig_strings() {
        assert_eq!(Desig::Empty.to_string(), "None");
        assert_eq!(Desig::Naif(100).to_string(), "100");
        assert_eq!(Desig::Name("Foo".into()).to_string(), "Foo");
        assert_eq!(Desig::Perm(123).to_string(), "123");
        assert_eq!(Desig::Prov("Prov".into()).to_string(), "Prov");
    }

    #[test]
    fn naif_name_resolution() {
        let desig = Desig::Naif(1).try_naif_id_to_name();
        assert_eq!(desig, Desig::Name("mercury barycenter".into()));
        assert_eq!(desig.full_string(), "Name(\"mercury barycenter\")");
        assert_eq!(desig.to_string(), "mercury barycenter");
    }

    #[test]
    fn obs_codes_loaded() {
        assert!(!OBS_CODES.is_empty());
    }

    /// Every (unpacked, packed) example given by the MPC, from
    /// <https://www.minorplanetcenter.net/iau/info/PackedDes.html> and
    /// <https://docs.minorplanetcenter.net/mpc-ops-docs/designations/provisional-designations/>.
    const MPC_EXAMPLES: [(&str, &str); 51] = [
        ("1995 XA", "J95X00A"),
        ("1995 XL1", "J95X01L"),
        ("1995 FB13", "J95F13B"),
        ("1998 SQ108", "J98SA8Q"),
        ("1998 SV127", "J98SC7V"),
        ("1998 SS162", "J98SG2S"),
        ("2099 AZ193", "K99AJ3Z"),
        ("2008 AA360", "K08Aa0A"),
        ("2007 TA418", "K07Tf8A"),
        ("2040 P-L", "PLS2040"),
        ("3138 T-1", "T1S3138"),
        ("1010 T-2", "T2S1010"),
        ("4101 T-3", "T3S4101"),
        ("1995 A1", "J95A010"),
        ("1994 P1-B", "J94P01b"),
        ("1994 P1", "J94P010"),
        ("2048 X13", "K48X130"),
        ("2033 L89-C", "K33L89c"),
        ("2088 A103", "K88AA30"),
        ("3202", "03202"),
        ("50000", "50000"),
        ("100345", "A0345"),
        ("360017", "a0017"),
        ("203289", "K3289"),
        ("620000", "~0000"),
        ("620061", "~000z"),
        ("3140113", "~AZaz"),
        ("15396335", "~zzzz"),
        ("Jupiter XIII", "J013S"),
        ("Saturn X", "S010S"),
        ("Neptune II", "N002S"),
        ("2023 BA", "K23B00A"),
        ("2024 CZ3", "K24C03Z"),
        ("2025 DZ619", "K25Dz9Z"),
        ("2025 DA620", "_PD0000"),
        ("2026 DY620", "_QD000N"),
        ("2027 DZ6190", "_RD0aEM"),
        ("2028 EA339749", "_SEZZZZ"),
        ("2029 FL591673", "_TFzzzz"),
        ("2026 CZ619", "K26Cz9Z"),
        ("2026 CA620", "_QC0000"),
        ("2026 CZ6190", "_QC0aEM"),
        ("2026 CL591673", "_QCzzzz"),
        ("P/2023 BA", "PK23B00A"),
        ("C/2024 CZ3", "CK24C03Z"),
        ("A/2025 DZ619", "AK25Dz9Z"),
        ("P/2025 DA620", "P_PD0000"),
        ("C/2026 DY620", "C_QD000N"),
        ("A/2027 DZ6190", "A_RD0aEM"),
        ("C/2028 EA339749", "C_SEZZZZ"),
        ("P/2029 FL591673", "P_TFzzzz"),
    ];

    #[test]
    fn mpc_examples_round_trip() {
        for (unpacked, packed) in MPC_EXAMPLES {
            let desig = Desig::parse_mpc_designation(unpacked).unwrap();
            assert_eq!(
                desig.try_pack().as_deref(),
                Ok(packed),
                "packing {unpacked}"
            );
            let desig = Desig::parse_mpc_packed_designation(packed).unwrap();
            assert_eq!(desig.to_string(), unpacked, "unpacking {packed}");
        }
    }

    /// Designations valid under the MPC rules, whose letters happen to form Roman
    /// numerals or which use the defunct comet type.
    #[test]
    fn valid_edge_designations_parse() {
        for (unpacked, packed) in [
            ("2000 CD", "K00C00D"),
            ("1998 XX", "J98X00X"),
            ("2004 MC", "K04M00C"),
            ("3D", "0003D"),
            ("A904 OA", "J04O00A"),
        ] {
            let desig = Desig::parse_mpc_designation(unpacked).unwrap();
            assert_eq!(
                desig.try_pack().as_deref(),
                Ok(packed),
                "packing {unpacked}"
            );
            let desig = Desig::parse_mpc_packed_designation(packed).unwrap();
            assert_eq!(desig.to_string(), unpacked, "unpacking {packed}");
        }
        let fragment = Desig::parse_mpc_designation("73P-B").unwrap();
        assert_eq!(fragment, Desig::CometPerm('P', 73, Some('B')));
        assert_eq!(fragment.to_string(), "73P-B");
    }

    /// Comet provisional designations from before 1800 pack and unpack.
    #[test]
    fn historical_comet_designations_round_trip() {
        for (unpacked, packed) in [
            ("C/1680 V1", "CG80V010"),
            ("C/1066 G1", "CA66G010"),
            ("C/0837 F1", "C837F010"),
            ("P/1772 E1", "PH72E010"),
        ] {
            let desig = Desig::parse_mpc_designation(unpacked).unwrap();
            assert_eq!(desig.try_pack().unwrap(), packed, "{unpacked}");
            let back = Desig::parse_mpc_packed_designation(packed).unwrap();
            assert_eq!(back.to_string(), unpacked, "{packed}");
        }
        // The minor planet form of a packed provisional designation still starts
        // in 1800.
        assert!(Desig::parse_mpc_packed_designation("H99A00A").is_err());
    }

    /// Input the MPC rules do not allow is an error, never a panic or a malformed
    /// packed string.
    #[test]
    fn invalid_designations_are_errors() {
        for unpacked in [
            "2020 IA",       // half-month I is omitted
            "2020 ZA",       // half-month Z is unused
            "2020 AI",       // order letter I is omitted
            "2005 AA620",    // extended format starts in 2010
            "2036 AA620",    // extended format ends in 2035
            "2026 CM591673", // past _zzzz
            "2026 CI620",    // order letter I is omitted
            "15396336",      // past ~zzzz
            "73P-B",         // no packed field for a numbered comet fragment
            "C/2026 A620",   // no extended format for comets
            "S/2003 J 2",    // satellite provisional designations
            "-B",            // no comet number
            "Jupiter M",     // satellite numbers have 3 digits
            "0",             // numbering starts at 1
            "2101 AA",       // packed years run from 1800 through 2099
            "1799 AA",       // minor planet designations start in 1800
            "C/2100 A1",     // comet years end in 2099
            "A925 AA",       // the A form is only for years before 1925
            "1995 XA0",      // no cycle count of zero
            "1995 XA01",     // no leading zero in the cycle count
            "C/2016 I1",     // half-month I is omitted
            "2020 A\u{e9}",
        ] {
            let packed = Desig::parse_mpc_designation(unpacked).and_then(|d| d.try_pack());
            assert!(packed.is_err(), "{unpacked} packed to {packed:?}");
        }
        for packed in [
            "_5A0000",
            "T1S31388",
            "SK03J020",
            "K16\u{e9}1",
            "K20I00A",
            "K20A00I",
            "L01A00A", // century past 20
            "H01A00A", // century before 18
            "00000",   // numbering starts at 1
            "0001X",   // numbered comets are P, D or I
            "K16J000", // comet order starts at 1
            "J000S",   // satellite numbering starts at 1
        ] {
            let unpacked = Desig::parse_mpc_packed_designation(packed);
            assert!(unpacked.is_err(), "{packed} unpacked to {unpacked:?}");
        }
    }
}
