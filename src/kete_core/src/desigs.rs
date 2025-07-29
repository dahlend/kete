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
//
// BSD 3-Clause License
//
// Copyright (c) 2025, Dar Dahlen
// Copyright (c) 2025, California Institute of Technology
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

use std::fmt::Display;

use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::{
    errors::{Error, KeteResult},
    spice::{naif_ids_from_name, try_name_from_id, try_obs_code_from_name},
};

static MPC_HEX: &str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
static ORDER_CHARS: &str = "ABCDEFGHJKLMNOPQRSTUVWXYZ";

/// Designations for an object.
///
/// This enum represents all of the different types of designations
/// which kete can represent.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Hash, Eq)]
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
    /// Third is if the comet is fragmented or not, if `Some` then the character
    /// is the fragment letter, if `None` then it is not a fragment.
    CometPerm(char, u32, Option<char>),

    /// Comet Provisional Designation
    /// First element is the orbit type `CPAXD` if available.
    /// Second is the string designation.
    /// Third is if the comet is fragmented or not, if `Some` then the character
    /// is the fragment letter, if `None` then it is not a fragment.
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
    pub fn full_string(&self) -> String {
        format!("{self:?}")
    }

    /// Try to convert a naif ID into a name.
    #[must_use]
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

    /// Convert the Desig as close to a NAIF id as is possible.
    /// This will lookup a NAIF id from the name if it exists.
    #[must_use]
    pub fn try_name_to_naif_id(self) -> Self {
        if let Self::Name(name) = &self {
            if let Ok(id) = name.parse::<i32>() {
                return Self::Naif(id);
            }

            let naif_ids = naif_ids_from_name(name);
            if naif_ids.len() == 1 {
                return Self::Naif(naif_ids[0].id);
            }
            // if there are multiple NAIF ids, or none, do not change the designation.
        }
        self
    }

    /// Convert the ``Name`` into an ``ObservatoryCode`` if possible.
    #[must_use]
    pub fn try_name_to_obs_code(self) -> Self {
        if let Self::Name(name) = &self {
            let obs_codes = try_obs_code_from_name(name);
            if obs_codes.len() == 1 {
                return Self::ObservatoryCode(obs_codes[0].name.clone());
            }
            // if there are multiple NAIF ids, or none, do not change the designation.
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
    ///    let desig = Desig::parse_mpc_designation("2026 CZ619");
    ///    assert_eq!(desig, Ok(Desig::Prov("2026 CZ619".to_string())));
    ///
    ///    // Extended Provisional (2025-2035)
    ///    let desig = Desig::parse_mpc_designation("2026 CA620");
    ///    assert_eq!(desig, Ok(Desig::Prov("2026 CA620".to_string())));
    ///
    ///    let desig = Desig::parse_mpc_designation("2028 EA339749");
    ///    assert_eq!(desig, Ok(Desig::Prov("2028 EA339749".to_string())));
    ///
    ///    let desig = Desig::parse_mpc_designation("2026 CL591673");
    ///    assert_eq!(desig, Ok(Desig::Prov("2026 CL591673".to_string())));
    ///
    ///    let desig = Desig::parse_mpc_designation("2029 FL591673");
    ///    assert_eq!(desig, Ok(Desig::Prov("2029 FL591673".to_string())));
    ///
    ///     // Comet provisional designations
    ///     let desig = Desig::parse_mpc_designation("1996 N2");
    ///     assert_eq!(desig, Ok(Desig::CometProv(None, "1996 N2".to_string(), None)));
    ///
    ///     let desig = Desig::parse_mpc_designation("C/2020 F3");
    ///     assert_eq!(desig, Ok(Desig::CometProv(Some('C'), "2020 F3".to_string(), None)));
    ///
    ///     let desig = Desig::parse_mpc_designation("p/2005 SB216");
    ///     assert_eq!(desig, Ok(Desig::CometProv(Some('p'), "2005 SB216".to_string(), None)));
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
    pub fn parse_mpc_designation(designation: &str) -> KeteResult<Self> {
        if designation.is_empty() {
            return Err(Error::ValueError("Designation cannot be empty".to_string()));
        }

        // if it has a slash, its a comet /
        // if it has a dash, its a fragmented comet
        // if it starts with a planet name, its a planetary satellite
        // if all digits are a number its a permanent designation

        if designation.chars().all(|x| x.is_ascii_digit()) {
            // its a permanent designation
            let num = designation.parse::<u32>().map_err(|_| {
                Error::ValueError(format!(
                    "Failed to parse MPC Permanent Designation: {designation}"
                ))
            })?;
            return Ok(Self::Perm(num));
        } else if !designation.contains(' ') {
            // there are no spaces, so it is not a provisional designation
            // it is probably a perm comet designation
            // check if the last character is in 'IP'
            let orbit_type = designation.chars().last().unwrap();
            if "IP".contains(orbit_type) {
                // it is a comet permanent designation
                let num = designation[..designation.len() - 1]
                    .parse::<u32>()
                    .map_err(|_| {
                        Error::ValueError(format!(
                            "Failed to parse MPC Comet Permanent Designation: {designation}"
                        ))
                    })?;
                return Ok(Self::CometPerm(orbit_type, num, None));
            }
            return Err(Error::ValueError(format!(
                "Invalid MPC Designation, sort of looks like a Comet: {designation}"
            )));
        }

        // It is some form of provisional designation or planetary satellite designation.
        let (header, tail) = designation
            .split_once(' ')
            .ok_or_else(|| Error::ValueError(format!("Invalid MPC Designation: {designation}")))?;

        if let Ok(num) = roman_to_int(tail) {
            match header {
                "Earth" => return Ok(Self::PlanetSat(399, num)),
                "Mars" => return Ok(Self::PlanetSat(499, num)),
                "Jupiter" => return Ok(Self::PlanetSat(599, num)),
                "Saturn" => return Ok(Self::PlanetSat(699, num)),
                "Uranus" => return Ok(Self::PlanetSat(799, num)),
                "Neptune" => return Ok(Self::PlanetSat(899, num)),
                _ => {
                    return Err(Error::ValueError(format!(
                        "Invalid planetary satellite designation: {designation}"
                    )));
                }
            }
        }

        if (header.len() < 4) || (tail.len() < 2) {
            return Err(Error::ValueError(format!(
                "Invalid MPC Designation header: {header}"
            )));
        }

        let indicator = tail
            .chars()
            .nth(1)
            .ok_or_else(|| Error::ValueError(format!("Invalid MPC Designation: {designation}")))?;
        let is_comet = header.contains('/') | indicator.is_ascii_digit();

        // if this indicator can be used to determine the type of designation
        // Letter or '-' makes it an asteroid designation
        //     If the asteroid designation also contains a '/' then it was redesigned
        //     as a comet.
        // Number makes it a comet

        if !is_comet & (indicator.is_ascii_alphabetic() || indicator == '-') {
            // It is an asteroid provisional designation
            Ok(Self::Prov(designation.to_string()))
        } else {
            // It is a comet provisional designation
            let (des, fragment) = designation.split_once('-').unwrap_or((designation, ""));

            let (orbit_type, des) = des.split_once('/').unwrap_or(("", des));
            if (fragment.len() > 1) || (orbit_type.len()) > 1 {
                return Err(Error::ValueError(format!(
                    "Invalid MPC Designation, looks like a comet provisional: {designation}"
                )));
            }
            let fragment = if fragment.is_empty() {
                None
            } else {
                Some(fragment.chars().next().unwrap())
            };
            let orbit_type = if orbit_type.is_empty() {
                None
            } else {
                Some(orbit_type.chars().next().unwrap())
            };
            Ok(Self::CometProv(orbit_type, des.to_string(), fragment))
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
    pub fn try_pack(&self) -> KeteResult<String> {
        match self {
            Self::Empty => Err(Error::ValueError(
                "Cannot pack an empty designation".to_string(),
            )),
            Self::Name(s) => Err(Error::ValueError(format!(
                "Cannot pack a name as a designation: {s}"
            ))),
            Self::Naif(i) => Err(Error::ValueError(format!(
                "Cannot pack a NAIF ID as a designation: {i}"
            ))),
            Self::ObservatoryCode(s) => {
                if s.len() != 3 {
                    return Err(Error::ValueError(format!(
                        "MPC Earth Station Designation must be 3 characters: {s}",
                    )));
                }
                Ok(format!("{s:0>3}"))
            }
            Self::Perm(num) => {
                if *num < 620_000 {
                    let idx = num / 10_000;
                    let rem = num % 10_000;
                    Ok(format!(
                        "{}{:0>4}",
                        MPC_HEX.chars().nth(idx as usize).unwrap(),
                        rem
                    ))
                } else {
                    Ok(format!("~{:0>4}", &num_to_mpc_hex(num - 620_000)))
                }
            }
            Self::CometPerm(orbit_type, id, _) => Ok(format!("{id:0>4}{orbit_type}")),
            Self::PlanetSat(planet_id, sat_num) => match planet_id {
                399 => Ok(format!("E{sat_num:0>3}S")),
                499 => Ok(format!("M{sat_num:0>3}S")),
                599 => Ok(format!("J{sat_num:0>3}S")),
                699 => Ok(format!("S{sat_num:0>3}S")),
                799 => Ok(format!("U{sat_num:0>3}S")),
                899 => Ok(format!("N{sat_num:0>3}S")),
                _ => Err(Error::ValueError(format!(
                    "Invalid planetary satellite center NAIF ID: {planet_id}"
                ))),
            },

            Self::Prov(des) => {
                if des.len() <= 6 {
                    return Err(Error::ValueError(format!(
                        "MPC Provisional Designation too short: {des}",
                    )));
                }
                let (year, des) = des
                    .split_ascii_whitespace()
                    .take(2)
                    .collect_tuple()
                    .unwrap();

                if des.starts_with("P-L")
                    || des.starts_with("T-1")
                    || des.starts_with("T-2")
                    || des.starts_with("T-3")
                {
                    Ok(format!("{}{}S{}", &des[0..1], &des[2..3], year))
                } else {
                    let order = des.chars().nth(1).unwrap();
                    let num: u32 = if des.len() == 2 { 0 } else { des[2..].parse()? };
                    match num {
                        num if (num >= 620) => {
                            // order counts A = 1, B = 2 ... skipping I
                            let single_count = ORDER_CHARS.find(order).unwrap() as u32;

                            let total_num = 25 * num + single_count;
                            let num_packed = num_to_mpc_hex(total_num - 15_500);
                            let decade: u32 = year[2..4].parse()?;
                            let decade_packed = num_to_mpc_hex(decade);

                            let half_month = &des[..1];
                            Ok(format!("_{decade_packed}{half_month}{num_packed:0>4}"))
                        }
                        _ => {
                            // unlike the `order`, this DOES include I...
                            let idx = MPC_HEX.chars().nth(num as usize / 10).unwrap();
                            let idy = num % 10;
                            static YEAR_LOOKUP: &[(&str, &str); 5] = &[
                                ("18", "I"),
                                ("19", "J"),
                                ("20", "K"),
                                ("A9", "J"),
                                ("A8", "I"),
                            ];
                            let century =
                                YEAR_LOOKUP.iter().find(|&&x| x.0 == &year[0..2]).map_or(
                                    Err(Error::ValueError(format!(
                                        "Invalid year in MPC Provisional Designation: {year}",
                                    ))),
                                    |&(_, c)| Ok(c),
                                )?;
                            let decade = &year[2..4];
                            let half_month = &des[..1];
                            Ok(format!("{century}{decade}{half_month}{idx}{idy}{order}"))
                        }
                    }
                }
            }
            Self::CometProv(orbit_type, unpacked, fragment) => {
                let (year, des) = unpacked
                    .split_ascii_whitespace()
                    .take(2)
                    .collect_tuple()
                    .ok_or_else(|| {
                        Error::ValueError(format!(
                            "Invalid MPC Comet Provisional Designation: {unpacked}"
                        ))
                    })?;

                if des.chars().nth(1).unwrap().is_ascii_digit() {
                    // Comet like designation
                    // 2033 L89-C
                    let num = des[1..].parse::<u32>().map_err(|_| {
                        Error::ValueError(format!(
                            "Invalid MPC Comet Provisional Designation: {des} {unpacked}",
                        ))
                    })?;
                    let outnum = if num > 99 {
                        format!(
                            "{}{}",
                            MPC_HEX.chars().nth(num as usize / 10).unwrap(),
                            num % 10
                        )
                    } else {
                        format!("{num:>02}")
                    };

                    Ok(format!(
                        "{}{}{}{}{}{}",
                        orbit_type.map(|o| o.to_string()).unwrap_or_default(),
                        MPC_HEX.chars().nth(year[0..2].parse::<usize>()?).unwrap(),
                        &year[2..4],
                        &des[0..1],
                        outnum,
                        fragment.map_or("0".to_string(), |f| f.to_ascii_lowercase().to_string())
                    ))
                } else {
                    // its an asteroid like designation
                    let packed = Self::Prov(unpacked.to_string()).try_pack()?;
                    Ok(format!(
                        "{}{}",
                        orbit_type.map_or("".to_string(), |o| o.to_string()),
                        packed
                    ))
                }
            }
        }
    }

    /// Unpacked a MPC packed designation into a [`Desig`] enum.
    pub fn parse_mpc_packed_designation(packed: &str) -> KeteResult<Self> {
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
}

impl Display for Desig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&match self {
            Self::Empty => "None".to_string(),
            Self::Prov(s) | Self::Name(s) | Self::ObservatoryCode(s) => s.clone(),
            Self::Perm(i) => i.to_string(),
            Self::Naif(i) => i.to_string(),
            Self::CometPerm(orbit_type, id, fragment) => {
                let frag_str = fragment.map_or(String::new(), |x| format!(" {x}"));
                format!("{id}{orbit_type}{frag_str}")
            }
            Self::CometProv(orbit_type, id, fragment) => {
                let orbit_str = orbit_type.map_or(String::new(), |o| o.to_string() + "/");
                let frag_str = fragment.map_or(String::new(), |x| "-".to_string() + &x.to_string());
                format!("{orbit_str}{id}{frag_str}")
            }
            Self::PlanetSat(planet_id, sat_num) => {
                let roman = int_to_roman(*sat_num).map_err(|_| std::fmt::Error)?;
                match planet_id {
                    399 => format!("Earth {roman}"),
                    499 => format!("Mars {roman}"),
                    599 => format!("Jupiter {roman}"),
                    699 => format!("Saturn {roman}"),
                    799 => format!("Uranus {roman}"),
                    899 => format!("Neptune {roman}"),
                    _ => return Err(std::fmt::Error),
                }
            }
        })
    }
}

/// Convert a u64 number to a string representation in the MPC hexadecimal format.
/// ```
///     use kete_core::desigs::num_to_mpc_hex;
///
///     let hex_str = num_to_mpc_hex(63);
///     assert_eq!(hex_str, "11");
/// ```
pub fn num_to_mpc_hex(mut num: u32) -> String {
    let mut result = String::new();

    if num == 0 {
        return "0".to_string();
    }
    while num > 0 {
        let digit = (num % 62) as usize;
        result.push(MPC_HEX.chars().nth(digit).unwrap());
        num /= 62;
    }
    result.chars().rev().collect()
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
pub fn mpc_hex_to_num(hex: &str) -> KeteResult<u32> {
    let mut result = 0_u32;
    for c in hex.chars() {
        if let Some(pos) = MPC_HEX.find(c) {
            result = result * 62 + pos as u32;
        } else {
            return Err(Error::IOError(format!(
                "Invalid character in MPC hexadecimal string: {c}"
            )));
        }
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
pub fn unpack_perm_designation(designation: &str) -> KeteResult<Desig> {
    if designation.len() != 5 {
        return Err(Error::ValueError(format!(
            "Invalid MPC Permanent Designation length, must be length 5: {}",
            designation.len()
        )));
    }
    // check if the first char is '~'
    if let Some(remaining) = designation.strip_prefix('~') {
        // high number packed asteroids, above 620_000
        let num = mpc_hex_to_num(remaining)?;
        Ok(Desig::Perm(num + 620_000))
    } else if designation.ends_with('S') {
        // planet satellite
        let (first_char, remaining) = designation.split_at(1);
        // take all but the last character from remaining and parse it to a u32
        let num = remaining[..3].parse::<u32>().map_err(|e| {
            Error::ValueError(format!(
                "Failed to parse planetary satellite designation: {e} {remaining}"
            ))
        })?;
        if !(1..=3999).contains(&num) {
            return Err(Error::ValueError(format!(
                "Planetary satellite number must be between 1 and 3999: {num}"
            )));
        }
        match first_char {
            "E" => Ok(Desig::PlanetSat(399, num)),
            "M" => Ok(Desig::PlanetSat(499, num)),
            "J" => Ok(Desig::PlanetSat(599, num)),
            "S" => Ok(Desig::PlanetSat(699, num)),
            "U" => Ok(Desig::PlanetSat(799, num)),
            "N" => Ok(Desig::PlanetSat(899, num)),
            _ => Err(Error::ValueError(format!(
                "Invalid first character for planetary satellite designation: {first_char}"
            ))),
        }
    } else if designation.ends_with('A')
        || designation.ends_with('P')
        || designation.ends_with('D')
        || designation.ends_with('X')
        || designation.ends_with('I')
    {
        // comets
        let num = designation[..4].parse::<u32>().map_err(|e| {
            Error::ValueError(format!("Failed to parse comet permanent designation: {e}"))
        })?;
        Ok(Desig::CometPerm(
            designation.chars().last().unwrap(),
            num,
            None,
        ))
    } else {
        // split string by first character and remaining
        let (first_char, remaining) = designation.split_at(1);
        if let Some(pos) = MPC_HEX.find(first_char) {
            let first_num = pos as u32 * 10_000;
            let rest_num = remaining.parse::<u32>().map_err(|e| {
                Error::ValueError(format!("Failed to parse rest of designation: {e}"))
            })?;
            Ok(Desig::Perm(first_num + rest_num))
        } else {
            Err(Error::ValueError(format!(
                "Invalid character in MPC Permanent Designation: {first_char}"
            )))
        }
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
pub fn unpack_prov_designation(designation: &str) -> KeteResult<Desig> {
    if designation.len() > 8 {
        return Err(Error::ValueError(format!(
            "Invalid MPC Provisional Designation length, must be 7 or 8: {}",
            designation.len()
        )));
    }
    let order = designation.chars().nth(6).unwrap();

    let (orbit_type, designation, is_comet) = match designation.len() {
        d if d == 7 && designation.starts_with("PLS")
            || designation.starts_with("T1S")
            || designation.starts_with("T2S")
            || designation.starts_with("T3S") =>
        {
            if designation[3..].chars().all(|c| !c.is_ascii_digit()) {
                return Err(Error::ValueError(format!(
                    "Provisional designation appears to be a survey but incorrectly formatted: {designation}"
                )));
            }
            return Ok(Desig::Prov(format!(
                "{} {}-{}",
                &designation[3..],
                &designation[0..1],
                &designation[1..2],
            )));
        }
        7 => (
            None,
            designation,
            !designation.starts_with('_') & (order.is_ascii_digit() || order.is_ascii_lowercase()),
        ),
        8 => {
            let (first, rest) = designation.split_at(1);
            let first = first.chars().next().unwrap().to_ascii_uppercase();
            if "APCDXI".contains(first) {
                (Some(first), rest, true)
            } else {
                return Err(Error::ValueError(format!(
                    "Invalid MPC Provisional Designation: {designation}"
                )));
            }
        }
        _ => {
            return Err(Error::ValueError(format!(
                "Invalid MPC Provisional Designation length, must be 7 or 8: {}",
                designation.len()
            )));
        }
    };

    let err = || {
        Error::ValueError(format!(
            "Invalid MPC Provisional Designation: {designation}"
        ))
    };
    if is_comet {
        let century = MPC_HEX.find(&designation[..1]).ok_or_else(err)? * 100;
        let year = designation[1..3].parse::<u32>().map_err(|_| err())? + century as u32;
        let comet_num = MPC_HEX.find(&designation[4..5]).ok_or_else(err)? as u32 * 10
            + designation[5..6].parse::<u32>().map_err(|_| err())?;
        let half_month = &designation[3..4];
        let frag = designation.chars().nth(6).unwrap();

        match frag {
            '0' => Ok(Desig::CometProv(
                orbit_type,
                format!("{year} {half_month}{comet_num}"),
                None,
            )),
            x if x.is_ascii_lowercase() => Ok(Desig::CometProv(
                orbit_type,
                format!("{year} {half_month}{comet_num}"),
                Some(x.to_ascii_uppercase()),
            )),
            _ => {
                // this has an asteroid prov designation, isn't that fun?
                let ast_desig = unpack_ast_prov_desig(designation)?;
                Ok(Desig::CometProv(orbit_type, ast_desig.to_string(), None))
            }
        }
    } else {
        unpack_ast_prov_desig(designation)
    }
}

/// unpack an MPC Provisional Designation that is in the asteroid format.
///
/// Note that some comets were originally labeled as asteroids, so they
/// may have an asteroid provisional designation, with a comet orbit type
/// in front.
fn unpack_ast_prov_desig(designation: &str) -> KeteResult<Desig> {
    if designation.starts_with('_') {
        return unpack_ast_extended_prov_des(designation);
    }
    let order = designation.chars().nth(6).unwrap();
    let err = || {
        Error::ValueError(format!(
            "Invalid MPC Provisional Designation: {designation}"
        ))
    };
    let century = MPC_HEX.find(&designation[..1]).ok_or_else(err)? as u32 * 100;
    let year = designation[1..3].parse::<u32>().map_err(|_| err())? + century;
    let year = if year < 1925 {
        format!("A{}", year % 1000)
    } else {
        year.to_string()
    };
    let obs_num = MPC_HEX.find(&designation[4..5]).ok_or_else(err)? as u32 * 10
        + designation[5..6].parse::<u32>().map_err(|_| err())?;
    let obs_str = if obs_num == 0 {
        String::new()
    } else {
        obs_num.to_string()
    };
    let half_month = designation.chars().nth(3).unwrap();
    Ok(Desig::Prov(format!("{year} {half_month}{order}{obs_str}")))
}

/// Unpack the extended asteroid provisional designation
///
///
fn unpack_ast_extended_prov_des(designation: &str) -> KeteResult<Desig> {
    let is_extended = designation.starts_with('_');
    let err = || {
        Error::ValueError(format!(
            "Invalid MPC Extended Provisional Designation: {designation}"
        ))
    };
    if !is_extended || designation.len() != 7 {
        return Err(err());
    }
    let year = 2000 + MPC_HEX.find(&designation[1..2]).ok_or_else(err)? as u32;
    let half_month = designation.chars().nth(2).unwrap();
    let total_count = mpc_hex_to_num(&designation[3..])? + 15_500;
    let count = total_count.div_euclid(25);
    let order = ORDER_CHARS
        .chars()
        .nth(total_count.rem_euclid(25) as usize)
        .unwrap();

    Ok(Desig::Prov(format!("{year} {half_month}{order}{count}")))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn desig_strings() {
        assert!(Desig::Empty.to_string() == "None");
        assert!(Desig::Naif(100).to_string() == "100");
        assert!(Desig::Name("Foo".into()).to_string() == "Foo");
        assert!(Desig::Perm(123).to_string() == "123");
        assert!(Desig::Prov("Prov".into()).to_string() == "Prov");
    }
}
