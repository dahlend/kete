//! Optional feature support for reading/writing parquet table
//!
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

use itertools::Itertools;
use std::fs::File;

use crate::errors::{Error, KeteResult};
use crate::frames::Equatorial;
use crate::state::State;

use polars::prelude::*;

/// Write a collection of states to a parquet table.
pub fn write_states_parquet(states: &[State<Equatorial>], filename: &str) -> KeteResult<()> {
    let desigs = Column::new(
        "desig".into(),
        states
            .iter()
            .map(|state| state.desig.to_string())
            .collect_vec(),
    );
    let jd = Column::new(
        "jd".into(),
        states.iter().map(|state| state.jd).collect_vec(),
    );
    let x = Column::new(
        "x".into(),
        states.iter().map(|state| state.pos[0]).collect_vec(),
    );
    let y = Column::new(
        "y".into(),
        states.iter().map(|state| state.pos[1]).collect_vec(),
    );
    let z = Column::new(
        "z".into(),
        states.iter().map(|state| state.pos[2]).collect_vec(),
    );
    let vx = Column::new(
        "vx".into(),
        states.iter().map(|state| state.vel[0]).collect_vec(),
    );
    let vy = Column::new(
        "vy".into(),
        states.iter().map(|state| state.vel[1]).collect_vec(),
    );
    let vz = Column::new(
        "vz".into(),
        states.iter().map(|state| state.vel[2]).collect_vec(),
    );
    let center = Column::new(
        "center".into(),
        states.iter().map(|state| state.center_id).collect_vec(),
    );
    let mut df = DataFrame::new(vec![desigs, jd, x, y, z, vx, vy, vz, center])
        .expect("Failed to construct dataframe");
    let file = File::create(filename).expect("could not create file");
    let _ = ParquetWriter::new(file)
        .finish(&mut df)
        .map_err(|_| Error::IOError("Failed to write to file".into()))?;
    Ok(())
}

/// Read a collection of states from a parquet table.
pub fn read_states_parquet(filename: &str) -> KeteResult<Vec<State<Equatorial>>> {
    // this reads the parquet table, then creates iterators over the contents, making
    // states by going through the iterators one at a time.
    let r = File::open(filename)?;
    let reader = ParquetReader::new(r);
    let mut dataframe = reader.finish().map_err(|_| {
        Error::IOError("Failed to read contents of file as a parquet table.".into())
    })?;
    let dataframe = dataframe.as_single_chunk_par();

    // create all the iterators, these are all type dependant, so they get special cased
    let mut desig_iter = dataframe
        .column("desig")
        .map_err(|_| Error::IOError("File doesn't contain the correct columns".into()))?
        .str()
        .expect("Designations are not all strings.")
        .into_no_null_iter();

    let mut center_iter = dataframe
        .column("center")
        .map_err(|_| Error::IOError("File doesn't contain the correct columns".into()))?
        .i32()
        .expect("Centers are not all ints.")
        .into_no_null_iter();

    // the remaining columns are all floats, so here we make a vector of iterators of
    // floats
    let mut state_iters = dataframe
        .columns(["jd", "x", "y", "z", "vx", "vy", "vz"])
        .map_err(|_| Error::IOError("File doesn't contain the correct columns".into()))?
        .iter()
        .map(|s| {
            s.f64()
                .expect("state information is not all floats.")
                .into_no_null_iter()
        })
        .collect::<Vec<_>>();

    Ok((0..dataframe.height())
        .map(|_| {
            let desig = desig_iter
                .next()
                .expect("should have as many iterations as rows");
            let center_id = center_iter.next().unwrap();

            let jd = state_iters[0].next().unwrap();
            let x = state_iters[1].next().unwrap();
            let y = state_iters[2].next().unwrap();
            let z = state_iters[3].next().unwrap();
            let vx = state_iters[4].next().unwrap();
            let vy = state_iters[5].next().unwrap();
            let vz = state_iters[6].next().unwrap();

            State::new(
                crate::desigs::Desig::Name(desig.to_string()),
                jd,
                [x, y, z].into(),
                [vx, vy, vz].into(),
                center_id,
            )
        })
        .collect())
}
