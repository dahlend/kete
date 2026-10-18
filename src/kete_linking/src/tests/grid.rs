// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! The grid of hypotheses.

use std::collections::HashSet;

use kete_core::constants::GMS;
use kete_core::prelude::Error;

use crate::hypothesis::axes;
use crate::{DEPTH_STEP, RATE_STEP, hypothesis_grid};

use super::synthetic::{HOUR, Observer, Population, Rng, nearest_axis, sight, true_hypothesis};
use super::{orbit_from_hypothesis, reference};

#[test]
fn hypothesis_grid_shape() {
    let epoch = reference();
    let planet = Observer::planet(epoch);
    let origin = planet.state(epoch);
    let grid = hypothesis_grid(&origin, 0.5, 5.0, DEPTH_STEP, RATE_STEP).unwrap();
    assert!(grid.iter().all(|h| h.origin == origin));

    // Depths are a geometric series from the smallest up to the largest, and
    // every depth and axis has rates which are whole steps. The step is the
    // default, or a quarter of the escape speed at the plane over the depth
    // if less. The rates reach what a bound orbit on the plane allows, and no
    // more than half a step past it.
    let count = (0..1000)
        .take_while(|&k| 0.5 * (DEPTH_STEP * f64::from(k)).exp() <= 5.0)
        .count();
    let mut seen = HashSet::new();
    for h in &grid {
        let axis = axes().iter().position(|a| *a == h.direction).unwrap();
        let k = ((h.depth / 0.5).ln() / DEPTH_STEP).round();
        let depth = 0.5 * (DEPTH_STEP * k).exp();
        assert!((h.depth / depth - 1.0).abs() < 1e-12, "depth {}", h.depth);
        assert!(k >= 0.0 && h.depth <= 5.0, "depth {}", h.depth);
        let plane = (h.depth + h.direction.dot(&origin.pos)).abs().max(0.00465);
        let escape = (2.0 * GMS / plane).sqrt();
        let step = RATE_STEP.min(0.25 * escape / h.depth);
        let rate = h.depth_rate / h.depth / step;
        assert!((rate - rate.round()).abs() < 1e-6, "rate {}", h.depth_rate);
        let speed = h.depth_rate + h.direction.dot(&origin.vel);
        assert!(
            speed.abs() - escape <= 0.5 * RATE_STEP * h.depth + 1e-12,
            "rate {} at depth {}",
            h.depth_rate,
            h.depth
        );
        let _ = seen.insert((axis, depth.to_bits()));
    }
    assert_eq!(seen.len(), 6 * count, "depths and axes with rates");

    // Each random bound object of every population is within half a step of
    // some hypothesis, of a grid about its own depth, and its pair solves to
    // a bound orbit under one of those near hypotheses. The bound range of
    // a distant object is narrower than the default rate step. Sights
    // without noise leave no margin past the escape speed.
    let mut rng = Rng(41);
    let half = (0.5 * DEPTH_STEP).exp();
    for population in Population::BOUND.into_iter().chain([Population::Centaur]) {
        for _ in 0..40 {
            let object = population.draw(&mut rng, epoch);
            let axis = nearest_axis(&(object.pos - origin.pos));
            let truth = true_hypothesis(&object, &origin, axis);
            let local = hypothesis_grid(
                &origin,
                truth.depth / half,
                truth.depth * half,
                DEPTH_STEP,
                RATE_STEP,
            )
            .unwrap();
            let near: Vec<_> = local
                .iter()
                .filter(|h| {
                    h.direction == axis
                        && (h.depth / truth.depth).ln().abs() <= 0.5 * DEPTH_STEP + 1e-9
                        && (h.depth_rate / h.depth - truth.depth_rate / truth.depth).abs()
                            <= 0.5 * RATE_STEP + 1e-12
                })
                .collect();
            assert!(
                !near.is_empty(),
                "{population:?} at depth {} not covered",
                truth.depth
            );
            let pair = [-HOUR, HOUR].map(|t| sight(&planet, &object, epoch + t).0);
            let solves = near
                .iter()
                .any(|h| orbit_from_hypothesis(&pair[0], &pair[1], h).is_ok());
            assert!(
                solves,
                "{population:?} at depth {} has no bound orbit",
                truth.depth
            );
        }
    }
}

#[test]
fn hypothesis_grid_rejects() {
    let epoch = reference();
    let origin = Observer::planet(epoch).state(epoch);
    let refused = |near: f64, far: f64, depth_step: f64, rate_step: f64| {
        let result = hypothesis_grid(&origin, near, far, depth_step, rate_step);
        matches!(result, Err(Error::ValueError(_)))
    };

    // Limits which are not a range of positive depths and positive steps.
    for (near, far, depth_step, rate_step) in [
        (0.0, 1.0, 0.1, 1e-3),
        (2.0, 1.0, 0.1, 1e-3),
        (1.0, 2.0, 0.0, 1e-3),
        (1.0, 2.0, 0.1, -1.0),
        (1.0, f64::INFINITY, 0.1, 1e-3),
        (f64::NAN, 2.0, 0.1, 1e-3),
    ] {
        assert!(refused(near, far, depth_step, rate_step), "{near} {far}");
    }

    // A grid too large to hold is refused before it is built, as are steps
    // or depths so small that a count of steps is past the range of any
    // integer.
    for (near, far, depth_step, rate_step) in [
        (0.5, 5.0, 1e-12, RATE_STEP),
        (0.5, 5.0, DEPTH_STEP, 1e-15),
        (1e-310, 5.0, DEPTH_STEP, RATE_STEP),
        (1e-6, 5.0, DEPTH_STEP, RATE_STEP),
        (1.0, 1.0, DEPTH_STEP, 1e-21),
        (1.0, 1.0, DEPTH_STEP, f64::MIN_POSITIVE),
        (1e-300, 1e-300, DEPTH_STEP, RATE_STEP),
        (5e-324, 5e-324, DEPTH_STEP, RATE_STEP),
        (f64::MIN_POSITIVE, 1.0, DEPTH_STEP, RATE_STEP),
        (1.0, 2.0, 1e-300, RATE_STEP),
        (1.0, 2.0, f64::MIN_POSITIVE, RATE_STEP),
    ] {
        assert!(
            refused(near, far, depth_step, rate_step),
            "depths {near} to {far}, steps {depth_step} and {rate_step}"
        );
    }

    // A tiny depth step over one depth is a grid of that depth.
    let one = hypothesis_grid(&origin, 1.0, 1.0, 1e-300, RATE_STEP).unwrap();
    assert!(!one.is_empty());
    assert!(one.iter().all(|h| (h.depth - 1.0).abs() < 1e-15));
}
