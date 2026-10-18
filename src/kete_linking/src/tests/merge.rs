// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Merging confirmed pieces of one object, on observations from the Earth of
//! the test kernels of objects moving under the n-body model.

use std::ops::RangeInclusive;

use crate::{Candidate, ConfirmRules, Confirmed, confirm, merge_pieces};

use super::harness::{MAX_RMS, SPAN, Scene, Sky, nightly, on_threads};
use super::reference;
use super::synthetic::Population;

/// Largest time between the two time groups of a confirmation with only two,
/// in days, so that two nights confirm a piece.
const MAX_GAP: f64 = 1.5;

/// Rules which confirm the pieces of these tests.
fn rules() -> ConfirmRules {
    ConfirmRules {
        min_sights: 4,
        max_rms: MAX_RMS,
        min_sights_per_side: 2,
        group_span: SPAN,
        max_gap: MAX_GAP,
    }
}

/// The observations of object `label` on these nights, sorted.
fn seen_on(scene: &Scene, label: usize, nights: &RangeInclusive<i32>) -> Vec<usize> {
    let night = |i: usize| (scene.observations[i].epoch().jd() - reference().jd()).round() as i32;
    (0..scene.labels.len())
        .filter(|&i| scene.labels[i] == label && nights.contains(&night(i)))
        .collect()
}

/// The confirmation of the observations of object `label` on these nights,
/// from its true state.
fn piece(scene: &Scene, label: usize, nights: RangeInclusive<i32>) -> Confirmed {
    let candidate = Candidate {
        sights: seen_on(scene, label, &nights),
        state: scene.objects[label].clone(),
    };
    let mut found = confirm(&scene.observations, &[candidate], &rules()).unwrap();
    assert_eq!(found.len(), 1, "object {label} on nights {nights:?}");
    found.remove(0)
}

#[test]
fn pieces_of_one_object_merge() {
    // Main belt objects on six nights of paired visits, each confirmed in
    // two pieces, its first three nights and its last three, as two passes
    // of a search may confirm them. Each object is merged whole, in the place
    // of its first piece, alike on one thread and on eight.
    let scene = Sky::new(Population::MainBelt, 4, &nightly(-3, 2, 2, 0.028))
        .on_earth()
        .scene();
    let objects = scene.objects.len();
    let on = |threads: usize| {
        let pieces: Vec<Confirmed> = (0..objects)
            .map(|label| piece(&scene, label, -3..=-1))
            .chain((0..objects).map(|label| piece(&scene, label, 0..=2)))
            .collect();
        let merged = on_threads(threads, || {
            merge_pieces(&scene.observations, pieces, &rules())
        })
        .unwrap();
        merged.into_iter().map(|c| c.sights).collect::<Vec<_>>()
    };
    let one = on(1);
    let whole: Vec<Vec<usize>> = (0..objects)
        .map(|label| seen_on(&scene, label, &(-3..=2)))
        .collect();
    assert_eq!(one, whole);
    assert_eq!(one, on(8));
}

#[test]
fn three_pieces_merge_in_two_rounds() {
    // Main belt objects on nine nights of paired visits, each confirmed in
    // three pieces of three nights. A round merges each piece at most once,
    // so the first merges two of the pieces of each object, and the second
    // the merged object with the third, which looks for it among the new
    // pieces alone. Each object comes out whole, in the place of its first
    // piece.
    let scene = Sky::new(Population::MainBelt, 2, &nightly(-4, 4, 2, 0.028))
        .on_earth()
        .scene();
    let objects = scene.objects.len();
    let pieces: Vec<Confirmed> = [-4..=-2, -1..=1, 2..=4]
        .into_iter()
        .flat_map(|nights| (0..objects).map(move |label| (label, nights.clone())))
        .map(|(label, nights)| piece(&scene, label, nights))
        .collect();
    let merged: Vec<Vec<usize>> = merge_pieces(&scene.observations, pieces, &rules())
        .unwrap()
        .into_iter()
        .map(|c| c.sights)
        .collect();
    let whole: Vec<Vec<usize>> = (0..objects)
        .map(|label| seen_on(&scene, label, &(-4..=4)))
        .collect();
    assert_eq!(merged, whole);
}

#[test]
fn smaller_piece_predicts_the_larger() {
    // Main belt objects on seven nights of paired visits, each confirmed in
    // two pieces, its first four nights and its last three. The fit of the
    // larger piece is marked as about the Earth, so it gives no orbit: of
    // two pieces of one object confirmed as they are, the larger one's orbit
    // predicts the smaller whenever the fit of both would merge them. The
    // smaller piece finds the larger in the index and predicts it, the fit of
    // the two starts from the smaller piece's orbit, and each object is
    // merged whole, in the place of its first piece.
    let scene = Sky::new(Population::MainBelt, 2, &nightly(-3, 3, 2, 0.028))
        .on_earth()
        .scene();
    let objects = scene.objects.len();
    let pieces: Vec<Confirmed> = (0..objects)
        .map(|label| {
            let mut larger = piece(&scene, label, -3..=0);
            larger.fit.uncertain_state.elements.center_id = 399;
            larger
        })
        .chain((0..objects).map(|label| piece(&scene, label, 1..=3)))
        .collect();
    let merged: Vec<Vec<usize>> = merge_pieces(&scene.observations, pieces, &rules())
        .unwrap()
        .into_iter()
        .map(|c| c.sights)
        .collect();
    let whole: Vec<Vec<usize>> = (0..objects)
        .map(|label| seen_on(&scene, label, &(-3..=3)))
        .collect();
    assert_eq!(merged, whole);
}

#[test]
fn pieces_sharing_an_observation_are_refused() {
    // Two pieces of one object which share an observation, and one piece
    // holding an observation twice, are errors naming that observation.
    let scene = Sky::new(Population::MainBelt, 1, &nightly(-3, 2, 2, 0.028))
        .on_earth()
        .scene();
    let first = piece(&scene, 0, -3..=0);
    let second = piece(&scene, 0, 0..=2);
    let shared = *first
        .sights
        .iter()
        .find(|i| second.sights.contains(i))
        .unwrap();
    let message = merge_pieces(&scene.observations, vec![first, second], &rules())
        .unwrap_err()
        .to_string();
    assert!(
        message.contains(&format!("observation {shared} ")),
        "{message}"
    );
    let mut twice = piece(&scene, 0, -3..=0);
    twice.sights.push(twice.sights[0]);
    let message = merge_pieces(&scene.observations, vec![twice], &rules())
        .unwrap_err()
        .to_string();
    assert!(message.contains("more than one piece"), "{message}");
}

#[test]
fn different_objects_do_not_merge() {
    // Two main belt objects starting within a twentieth of a degree of each
    // other, the first confirmed on its first three nights and the second on
    // its last three. The fit of the two pieces as one is not confirmed with
    // two thirds of each, and they are not merged.
    let scene = Sky {
        radius: 0.05,
        ..Sky::new(Population::MainBelt, 2, &nightly(-3, 2, 2, 0.028))
    }
    .on_earth()
    .scene();
    let pieces = vec![piece(&scene, 0, -3..=-1), piece(&scene, 1, 0..=2)];
    let alone: Vec<Vec<usize>> = pieces.iter().map(|p| p.sights.clone()).collect();
    for state in [&scene.objects[0], &scene.objects[1]] {
        let union = Candidate {
            sights: [alone[0].as_slice(), alone[1].as_slice()].concat(),
            state: state.clone(),
        };
        let found = confirm(&scene.observations, &[union], &rules()).unwrap();
        let both = |c: &Confirmed| {
            alone.iter().all(|piece| {
                let kept = piece.iter().filter(|i| c.sights.contains(i)).count();
                3 * kept >= 2 * piece.len()
            })
        };
        assert!(!found.iter().any(both), "the union is confirmed");
    }
    let merged = merge_pieces(&scene.observations, pieces, &rules()).unwrap();
    let merged: Vec<Vec<usize>> = merged.into_iter().map(|c| c.sights).collect();
    assert_eq!(merged, alone);
}
