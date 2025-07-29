//! List of NAIF ID values.
//! This list is not comprehensive, but is more complete than the C-SPICE
//! implementation.
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

/// Observatory Codes
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
    // this should be re-written to be simpler
    let desigs: Vec<String> = NAIF_IDS.iter().map(|n| n.name.to_lowercase()).collect();
    let desigs: Vec<&str> = desigs.iter().map(String::as_str).collect();
    partial_str_match(&name.to_lowercase(), &desigs)
        .into_iter()
        .map(|(i, _)| NAIF_IDS[i].clone())
        .collect()
}
