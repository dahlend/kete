// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! The fixed hypothesis directions: how well they cover moving objects, and
//! what a sight far from its axis costs.

use kete_core::frames::{Equatorial, Vector};

use crate::MIN_ALIGNMENT;
use crate::hypothesis::axes;

use super::synthetic::{
    ARCSEC, CASES, HOUR, Observer, Population, Rng, WINDOWS, add_noise, sight, true_hypothesis,
};
use super::{orbit_from_hypothesis, reference};

/// Alignment limits to compare, the one in use among them.
const LIMITS: [f64; 5] = [0.3, 0.4, MIN_ALIGNMENT, 0.5, 0.55];

/// For random objects of a population seen from the planet at 1 AU at 13
/// times over each window, the fraction with one axis aligned past each
/// limit with every sight, by limit and window.
fn coverage(population: Population) -> [[f64; 4]; 5] {
    let epoch = reference();
    let planet = Observer::planet(epoch);
    let mut rng = Rng(21);
    let objects: Vec<_> = (0..100).map(|_| population.draw(&mut rng, epoch)).collect();
    let mut covered = [[0.0; 4]; 5];
    for (column, window) in WINDOWS.into_iter().enumerate() {
        for object in &objects {
            let directions: Vec<Vector<Equatorial>> = (0..=12)
                .map(|k| sight(&planet, object, epoch + window * (f64::from(k) / 6.0 - 1.0)).0)
                .map(|seen| seen.direction)
                .collect();
            for (row, limit) in LIMITS.into_iter().enumerate() {
                let whole = axes()
                    .into_iter()
                    .any(|axis| directions.iter().all(|u| axis.dot(u) >= limit));
                covered[row][column] += f64::from(u8::from(whole)) / objects.len() as f64;
            }
        }
    }
    covered
}

#[test]
fn axis_coverage() {
    // At the alignment limit in use, nearly all objects which are not close
    // keep one axis for all of their sights over a week, and most over two
    // weeks. Close ones lose it within days.
    println!("\nRandom objects seen from `planet 1 AU`: percent with one axis for all of their");
    println!("sights over the window, by alignment limit.");
    print!("{:>20}", "window");
    for window in WINDOWS {
        print!("{:>8}", format!("{window} d"));
    }
    println!();
    let in_use = LIMITS.iter().position(|&l| l == MIN_ALIGNMENT).unwrap();
    for population in Population::BOUND {
        let covered = coverage(population);
        for (limit, row) in LIMITS.iter().zip(&covered) {
            print!("{:>20}", format!("{population:?} {limit}"));
            for value in row {
                print!("{:>8.0}", 100.0 * value);
            }
            println!();
        }
        let [_, _, week, two_weeks] = covered[in_use];
        if population == Population::Close {
            assert!(week < 0.5, "{population:?}: {week} over a week");
        } else {
            assert!(week >= 0.95, "{population:?}: {week} over a week");
            assert!(
                two_weeks >= 0.85,
                "{population:?}: {two_weeks} over two weeks"
            );
        }
    }
}

#[test]
fn alignment_cost() {
    // Directions turned from the line of sight at the reference time, at
    // several azimuths, for every bound object seen from the planet.
    let epoch = reference();
    let planet = Observer::planet(epoch);
    let origin = planet.state(epoch);
    // Sights near the limit drift past it within the window, so the last
    // cosine stays a little above it.
    let cosines = [1.0_f64, 0.8, 0.6, 0.5];
    // Distant objects, which move little on the sky, so that every pair has
    // nearly the alignment of the reference time.
    let distant = [
        Population::MainBelt,
        Population::Trojan,
        Population::Tno,
        Population::Tilted,
    ];
    println!("\nWorst over distant objects of: spread at the reference time of nine");
    println!("pairs an hour long over 7 days, depth and rate 1 percent too large, and with");
    println!("0.1 arcsec noise, both over the distance to the object, by the cosine between");
    println!("the hypothesis direction and the line of sight.");
    println!("{:>8}{:>12}{:>12}", "cosine", "1 percent", "noise");
    let mut worst = Vec::new();
    for cos in cosines {
        let (mut scaled, mut noisy) = (0.0_f64, 0.0_f64);
        for case in &CASES {
            if !distant.contains(&case.population) {
                continue;
            }
            let truth = case.object(&origin);
            let u = (truth.pos - origin.pos).normalize();
            let side = u.cross(&Vector::new([0.3, -0.5, 0.8])).normalize();
            let other = u.cross(&side);
            for turn in 0..4 {
                let (sin_t, cos_t) = (0.3 + f64::from(turn) * 1.57).sin_cos();
                let sideways = (side * cos_t + other * sin_t) * (1.0 - cos * cos).sqrt();
                let exact = true_hypothesis(&truth, &origin, (u * cos + sideways).normalize());
                for (error, noise) in [(0.01, 0.0), (0.0, 0.1 * ARCSEC)] {
                    let mut hypothesis = exact.clone();
                    hypothesis.depth *= 1.0 + error;
                    hypothesis.depth_rate *= 1.0 + error;
                    let mut rng = Rng(6);
                    let mut states = Vec::new();
                    for k in 0..9 {
                        let start = -7.0 + (14.0 - HOUR) * f64::from(k) / 8.0;
                        let mut a = sight(&planet, &truth, epoch + start).0;
                        let mut b = sight(&planet, &truth, epoch + (start + HOUR)).0;
                        add_noise(&mut a, noise, &mut rng);
                        add_noise(&mut b, noise, &mut rng);
                        let aligned = hypothesis.direction.dot(&a.direction);
                        if aligned.min(hypothesis.direction.dot(&b.direction)) < MIN_ALIGNMENT {
                            continue;
                        }
                        // A pair can still fail, as it may under any
                        // hypothesis.
                        states.extend(orbit_from_hypothesis(&a, &b, &hypothesis));
                    }
                    if states.len() < 9 {
                        continue;
                    }
                    let sum = states
                        .iter()
                        .fold(Vector::new([0.0; 3]), |sum, s| sum + s.pos);
                    let mean = sum / states.len() as f64;
                    let spread = states
                        .iter()
                        .map(|s| (s.pos - mean).norm())
                        .fold(0.0, f64::max)
                        / case.dist;
                    if noise == 0.0 {
                        scaled = scaled.max(spread);
                    } else {
                        noisy = noisy.max(spread);
                    }
                }
            }
        }
        println!("{cos:>8.2}{scaled:>12.1e}{noisy:>12.1e}");
        worst.push((scaled, noisy));
    }
    // The cost of a sight far from its axis grows between the inverse of
    // the cosine and its inverse square.
    let (first, last) = (worst[0], worst[worst.len() - 1]);
    let cos = cosines[cosines.len() - 1];
    for grew in [last.0 / first.0, last.1 / first.1] {
        assert!(grew > 0.8 / cos && grew < 1.25 / (cos * cos), "grew {grew}");
    }
}
