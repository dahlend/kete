// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Scenario tests on synthetic objects.
//!
//! The default tests assert bounds, and some print the table they measured,
//! shown by `cargo test -- --nocapture`. The ignored tests in `matrix.rs` are
//! the regression benchmark of speed and quality, see there.

use kete_core::frames::{Equatorial, SunCenter};
use kete_core::prelude::{KeteResult, State};
use kete_core::time::{TDB, Time};

use crate::hypothesis::solve;
use crate::{Hypothesis, Sight};

mod axes;
mod calibration;
mod confirm;
mod grid;
mod harness;
mod hypothesis;
mod linking;
mod matrix;
mod merge;
mod ranges;
mod spread;
pub(crate) mod synthetic;
mod tiles;
mod units;
mod windows;

/// The reference time of every scenario. Nothing depends on its value.
fn reference() -> Time<TDB> {
    Time::new(2460000.5)
}

/// The orbit of two sights under a hypothesis, as its state at the reference
/// time, see [`solve`].
fn orbit_from_hypothesis(
    a: &Sight,
    b: &Sight,
    hypothesis: &Hypothesis,
) -> KeteResult<State<Equatorial, SunCenter>> {
    Ok(solve(a, b, hypothesis)?.state)
}
