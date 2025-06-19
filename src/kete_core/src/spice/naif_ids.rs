//! List of NAIF ID values.
//! This list is not comprehensive, but is more complete than the C-SPICE
//! implementation.
use lazy_static::lazy_static;
use serde::Deserialize;

use crate::prelude::{Error, KeteResult};
use crate::util::partial_str_match;
use std::str;
use std::str::FromStr;

/// NAIF ID information
#[derive(Debug, Deserialize, Clone)]
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

const PRELOAD_IDS: &[u8] = include_bytes!("../../data/naif_ids.csv");

lazy_static! {
    /// NAIF Ids
    static ref NAIF_IDS: Box<[NaifId]> = {
        let mut ids = Vec::new();
        let text = str::from_utf8(PRELOAD_IDS).unwrap().split('\n');
        for row in text.skip(1) {
            ids.push(NaifId::from_str(row).unwrap());
        }
        ids.into()
    };
}

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
    // this should be re-written to be simpler
    let desigs: Vec<String> = NAIF_IDS.iter().map(|n| n.name.to_lowercase()).collect();
    let desigs: Vec<&str> = desigs.iter().map(|x| x.as_str()).collect();
    partial_str_match(&name.to_lowercase(), &desigs)
        .into_iter()
        .map(|(i, _)| NAIF_IDS[i].clone())
        .collect()
}
