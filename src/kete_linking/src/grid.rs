// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! The grid of hypotheses a search runs over.

use kete_core::constants::GMS;
use kete_core::desigs::Desig;
use kete_core::frames::{Equatorial, SunCenter, Vector};
use kete_core::prelude::{Error, KeteResult, State};

use crate::Hypothesis;
use crate::hypothesis::axes;

/// Step of the grid in the natural log of the depth which the `grid_factor`
/// of a [`crate::SearchPass`] multiplies.
///
/// The cost of a search grows with the number of hypotheses, about the
/// inverse of the product of the two steps. With the default grid allowance
/// of [`crate::LinkSettings`], steps of this size find main belt and more
/// distant objects, though one seen once a night, found only with
/// `lone_pairs`, may be split into several candidates, each confirmed on its
/// own. Many near-Earth objects are lost at
/// these steps, see the test `matrix_grid_steps`. The default passes search
/// at multiples of these steps, a coarse pass and then a fine one, see
/// [`crate::SearchSettings::default`]. Close objects need steps chosen for
/// their distance, see `matrix_close_objects`.
pub const DEPTH_STEP: f64 = 0.4;

/// Step of the grid in depth rate over depth, per day, which the
/// `grid_factor` of a [`crate::SearchPass`] multiplies.
///
/// Unlike the depth rate alone, this ratio needs about the same step for
/// every population. Distant objects, whose bound orbits allow a narrower
/// range of the ratio, get a finer step from [`hypothesis_grid`]. See
/// [`DEPTH_STEP`].
pub const RATE_STEP: f64 = 4e-3;

/// Largest step of the grid in depth rate, as a fraction of the escape speed
/// at the plane of the hypothesis.
///
/// The solve refuses unbound orbits, so a hypothesis only finds an object if
/// the pairs of the object solve to bound orbits under it. Changing the depth
/// rate by `dv` changes the velocity of the solved orbit by `dv / c` along
/// the sight, where `c` is the cosine between sight and direction. For an
/// object at speed `v` where the escape speed is `v_e`, the depth rates which
/// keep its orbit bound therefore span at least `2 c sqrt(v_e^2 - v^2)`.
/// That is least for an object far from the Sun across the plane, where `v_e`
/// is less than at the plane. A quarter of the escape speed at the plane is
/// within that span for orbits of eccentricity up to about one half at any
/// alignment the solve accepts, see [`crate::MIN_ALIGNMENT`].
const ESCAPE_FRACTION: f64 = 0.25;

/// Radius of the Sun in AU. No bound orbit passes inside it.
const SUN_RADIUS: f64 = 0.00465;

/// Largest integer below which every integer is exactly a float, `2^53`.
const MAX_EXACT: f64 = 9_007_199_254_740_992.0;

/// Most hypotheses a grid may have. A grid is held in memory whole, as this
/// many times the size of a [`Hypothesis`] at most.
pub const MAX_HYPOTHESES: usize = 10_000_000;

/// Every hypothesis of a grid about one origin, along the six coordinate
/// axes of the Equatorial frame, both ways.
///
/// Depths run from `min_depth` to `max_depth` in steps of `depth_step` in
/// their natural log. At each depth the ratio of depth rate to depth takes
/// the whole multiples of a step which cover, to within half a step, the
/// rates a bound orbit allows: the escape speed at the distance of the plane
/// from the Sun, less the motion of the origin along the axis. The step is
/// `rate_step`, or a quarter of that escape speed over the depth where that
/// is less. The solve refuses unbound orbits, and for distant objects the
/// rates under which a pair stays bound span less than [`RATE_STEP`]. The
/// finer step keeps a hypothesis among them for every bound object.
///
/// On an axis across the line from the Sun to the origin, the plane of a
/// small depth passes close to the Sun, so the size of a grid grows as the
/// smallest depth to the power -1.5.
///
/// # Arguments
///
/// * `origin` - Point depths are measured from, and the reference time, as a
///   state with respect to the Sun in AU and AU per day.
/// * `min_depth` - Smallest depth, in AU.
/// * `max_depth` - Largest depth, in AU.
/// * `depth_step` - Step in the natural log of the depth, see [`DEPTH_STEP`].
/// * `rate_step` - Step in depth rate over depth, per day, see [`RATE_STEP`].
///
/// # Errors
///
/// [`Error::ValueError`] unless `0 < min_depth <= max_depth`, both steps are
/// positive, and all values are finite, if the grid would have more than
/// [`MAX_HYPOTHESES`] hypotheses, or if a rate would be more than `2^53`
/// steps from zero, as for a tiny `rate_step`, or for a depth so large that
/// the escape speed at its plane is lost to rounding.
pub fn hypothesis_grid(
    origin: &State<Equatorial, SunCenter>,
    min_depth: f64,
    max_depth: f64,
    depth_step: f64,
    rate_step: f64,
) -> KeteResult<Vec<Hypothesis>> {
    let layers = grid_layers(origin, min_depth, max_depth, depth_step, rate_step)?;
    // Every hypothesis holds a copy of the origin, without its designation,
    // whose text each copy would allocate again.
    let origin = State {
        desig: Desig::Empty,
        ..origin.clone()
    };
    let size = layers.iter().map(|&(_, _, _, low, high)| high - low + 1);
    let mut grid = Vec::with_capacity(size.sum::<i64>().try_into().unwrap_or(0));
    for (direction, depth, step, low, high) in layers {
        for k in low..=high {
            grid.push(Hypothesis {
                origin: origin.clone(),
                direction,
                depth,
                depth_rate: k as f64 * step * depth,
            });
        }
    }
    Ok(grid)
}

/// The errors of [`hypothesis_grid`] for its depths and steps, which need no
/// origin.
fn check_grid(min_depth: f64, max_depth: f64, depth_step: f64, rate_step: f64) -> KeteResult<()> {
    let finite = max_depth.is_finite() && depth_step.is_finite() && rate_step.is_finite();
    let depths = 0.0 < min_depth && min_depth <= max_depth;
    let steps = depth_step > 0.0 && rate_step > 0.0;
    if !(depths && steps && finite) {
        return Err(Error::ValueError(format!(
            "Hypothesis grid: need 0 < min_depth <= max_depth and positive steps, got depths \
             {min_depth} to {max_depth} and steps {depth_step}, {rate_step}"
        )));
    }
    Ok(())
}

/// The layers of [`hypothesis_grid`], each a direction, a depth, the step in
/// depth rate over depth there, and the lowest and highest multiples of that
/// step, with every check of [`hypothesis_grid`], without building the grid.
pub(crate) fn grid_layers(
    origin: &State<Equatorial, SunCenter>,
    min_depth: f64,
    max_depth: f64,
    depth_step: f64,
    rate_step: f64,
) -> KeteResult<Vec<(Vector<Equatorial>, f64, f64, i64, i64)>> {
    check_grid(min_depth, max_depth, depth_step, rate_step)?;
    if !origin.is_finite() {
        return Err(Error::ValueError(
            "Hypothesis grid: the origin must be finite".into(),
        ));
    }
    let too_large = || {
        Error::ValueError(format!(
            "Hypothesis grid: more than {MAX_HYPOTHESES} hypotheses, take a larger smallest \
             depth or larger steps"
        ))
    };
    let depths = ((max_depth / min_depth).ln() / depth_step + 1e-9).floor() + 1.0;
    let fits = 6.0 * depths <= MAX_HYPOTHESES as f64;
    if !fits {
        return Err(too_large());
    }
    // Counted in floating point before any cast, since a tiny step or depth
    // gives counts past the range of any integer.
    let mut layers = Vec::new();
    let mut total = 0.0;
    for direction in axes() {
        let along = direction.dot(&origin.vel);
        for k in 0..depths as i32 {
            let depth = min_depth * (depth_step * f64::from(k)).exp();
            let plane = (depth + direction.dot(&origin.pos)).abs().max(SUN_RADIUS);
            let fastest = (2.0 * GMS / plane).sqrt();
            let step = rate_step.min(ESCAPE_FRACTION * fastest / depth);
            // Past a depth where the escape speed over the depth is lost to
            // rounding there is no step.
            let deep = || {
                Error::ValueError(format!(
                    "Hypothesis grid: the depth {depth} AU is too large to step its rates, take \
                     a smaller max_depth"
                ))
            };
            if step <= 0.0 {
                return Err(deep());
            }
            // Half a step past each end, also when the range is under a step,
            // so that `high` is at least `low`.
            let low = ((-fastest - along) / depth / step - 0.5).ceil();
            let high = ((fastest - along) / depth / step + 0.5).floor();
            total += high - low + 1.0;
            let fits = total <= MAX_HYPOTHESES as f64;
            if !fits {
                return Err(too_large());
            }
            if !(low.abs() < MAX_EXACT && high.abs() < MAX_EXACT) {
                // A step finer than `rate_step` spans a few steps, so only
                // the motion of the origin can put them past the exact
                // integers: the escape speed is lost against it.
                if step < rate_step {
                    return Err(deep());
                }
                return Err(Error::ValueError(format!(
                    "Hypothesis grid: rates over depth reach {:e} steps, past the integers a \
                     float holds exactly, take a larger rate_step",
                    low.abs().max(high.abs())
                )));
            }
            layers.push((direction, depth, step, low as i64, high as i64));
        }
    }
    Ok(layers)
}
