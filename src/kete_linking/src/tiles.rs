// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Tiles of the sky, so that a search can run over a whole survey one patch
//! at a time.
//!
//! A tile is a core, one `HEALPix` pixel, and a halo around it. A sight joins
//! every tile whose core is within its halo, which grows with the time of the
//! sight from a reference time at the largest rate of motion, so that every
//! object whose direction at the reference time lies in a core has all of its
//! sights in that tile, and tiles can be searched one at a time.

use std::collections::BTreeMap;
use std::f64::consts::PI;

use kete_core::frames::{Equatorial, Vector};
use kete_core::prelude::{Error, KeteResult};
use kete_core::time::{TDB, Time};
use kete_stats::healpix::{
    max_pixel_radius, n_pixels, nested_to_ang, nested_vertices, query_disc_nested,
};
use rayon::prelude::*;

use crate::Sight;
use crate::sight::{MAX_DAYS, UNUSABLE, angle, first_and_last};

/// Size of the core of a tile over the largest halo of a sight, at least,
/// for the `nside` [`tiles()`] takes by default, see there for the trade
/// between overlap and memory. The size of a core is the square root of its
/// solid angle. At this size the halo adds no more than the core itself to
/// the disc a sight joins tiles within, and halving the core would more than
/// double the overlap of the sights with the largest halo.
const CORE_OVER_HALO: f64 = 1.0;

/// Largest `nside` [`tiles()`] takes by default, which it takes when the halo
/// is zero.
const MAX_DEFAULT_NSIDE: u32 = 1 << 10;

/// Largest `nside` of `HEALPix`.
const MAX_NSIDE: u32 = 1 << 29;

/// Most tiles a sight may be tested against, see [`tiles()`]. Tiles so small
/// against the halo overlap so much that searching them costs far more than
/// searching all the sights at once.
const MAX_TESTS_PER_SIGHT: f64 = 1000.0;

/// Added to every halo, in radians, so that rounding in the angles cannot
/// leave a sight out of a tile it belongs to.
const ROUNDING: f64 = 1e-9;

/// Largest angle between the directions of an object seen at one time from
/// two observers `separation` AU apart, when it is at least `min_distance` AU
/// from them: `asin(separation / min_distance)`, or pi when the separation is
/// as large as the distance. This is the `parallax_margin` of [`tiles()`] for
/// observers that far apart and objects searched down to that distance, such
/// as the smallest depth of a search.
///
/// # Errors
///
/// [`Error::ValueError`] unless `separation` is finite and not negative and
/// `min_distance` is finite and positive.
pub fn parallax_margin(separation: f64, min_distance: f64) -> KeteResult<f64> {
    if !(separation >= 0.0 && separation.is_finite()) {
        return Err(Error::ValueError(format!(
            "Parallax margin: separation must be finite and not negative, got {separation}"
        )));
    }
    if !(min_distance > 0.0 && min_distance.is_finite()) {
        return Err(Error::ValueError(format!(
            "Parallax margin: min_distance must be finite and positive, got {min_distance}"
        )));
    }
    Ok(if separation >= min_distance {
        PI
    } else {
        (separation / min_distance).asin()
    })
}

/// The `nside` [`tiles()`] takes for a largest halo of `halo` radians: the
/// largest power of two whose cores are at least [`CORE_OVER_HALO`] times
/// the halo in size, at least 1 and at most [`MAX_DEFAULT_NSIDE`].
fn default_nside(halo: f64) -> u32 {
    let size = |nside: u32| (4.0 * PI / n_pixels(nside) as f64).sqrt();
    let mut nside = 1;
    while nside < MAX_DEFAULT_NSIDE && size(2 * nside) >= CORE_OVER_HALO * halo {
        nside *= 2;
    }
    nside
}

/// The center of a core, and the largest angle from it to a vertex, which is
/// the largest to any point of the core. The test
/// `core_radius_reaches_its_boundary` checks that.
fn core(nside: u32, pixel: u64) -> (Vector<Equatorial>, f64) {
    let (ra, dec) = nested_to_ang(nside, pixel);
    let center = Vector::from_ra_dec(ra, dec);
    let radius = nested_vertices(nside, pixel)
        .iter()
        .map(|&(ra, dec)| angle(&center, &Vector::from_ra_dec(ra, dec)))
        .fold(0.0, f64::max);
    (center, radius)
}

/// Tiles of the sky, each the indices of the sights in it, sorted.
///
/// The core of a tile is one `HEALPix` pixel at `nside`, in the nested
/// order. A sight at time `t` joins every tile whose core is within its
/// halo, `max_rate * |t - reference| + parallax_margin` radians: the angle
/// from the center of the core to the sight is at most the halo plus the
/// largest angle from that center to a point of the core. Tiles are in the
/// order of their pixel, and tiles with no sights are left out.
///
/// Every object whose direction at the reference time lies in a core then
/// has all of its sights in that tile, if its apparent rate of motion is
/// never more than `max_rate` and `parallax_margin` bounds the angle between
/// its directions from two observers at one time, see [`parallax_margin`].
/// An object's direction from an observer moves at most the rate times the
/// time from the reference time, and its direction from another observer at
/// the same time differs by at most the margin. The core holds the direction
/// at the reference time, so a sight is at most the halo from some point of
/// the core. Each direction lies in one core, so each such object is whole in
/// at least one tile. Objects which are not whole in a tile, such as faster
/// ones, may still be found in a tile which holds enough of their sights.
///
/// The halo grows with the time from the reference time, so the tiles of a
/// long window overlap more. [`crate::search_windows`] searches a long survey
/// in windows of time.
///
/// Unless `nside` is given, it is the largest power of two whose cores are
/// at least as large, as the square root of their solid angle, as the
/// largest halo of a sight, or 1 if none is, and at most 1024. A sight
/// joins about as many tiles as cores fit in a disc of its halo plus the
/// radius of a core, so the overlap between tiles, and with it the total
/// work, is least for cores large against the halo, while the memory of
/// searching a tile grows with its sights.
///
/// # Arguments
///
/// * `sights` - Lines of sight, in any order.
/// * `max_rate` - Largest apparent rate of motion of the objects, in radians
///   per day, the `max_rate` of a pass of the search, see
///   [`crate::SearchPass`].
/// * `parallax_margin` - Largest angle between the directions of an object
///   at one time from two observers of the sights, in radians:
///   [`parallax_margin`] of the largest separation of two observers at one
///   time and the smallest depth searched. A single observer needs the
///   margin of its own excursion about a smooth path, such as the diameter
///   of the Earth for one site on the ground: the turning of the Earth moves
///   the direction of a close object back and forth by more than the rate
///   its pairs, taken at about the same time of day, show.
/// * `nside` - `HEALPix` resolution of the cores, a power of two. `None`
///   takes it from the halo as above.
/// * `reference` - Time the cores hold the directions of the objects at.
///   `None` takes the middle of the time the sights span.
///
/// # Errors
///
/// [`Error::ValueError`] unless `max_rate` is finite and positive and
/// `parallax_margin` finite and not negative, if a sight is not
/// [`Sight::is_usable`], if `reference` is not within 9e15 days of JD 0, if
/// `nside` is not a power of two up to `2^29`, or if the tiles would be so
/// small against the halo that a sight would be tested against more than
/// 1000 of them.
pub fn tiles(
    sights: &[Sight],
    max_rate: f64,
    parallax_margin: f64,
    nside: Option<u32>,
    reference: Option<Time<TDB>>,
) -> KeteResult<Vec<Vec<usize>>> {
    if !(max_rate > 0.0 && max_rate.is_finite()) {
        return Err(Error::ValueError(format!(
            "Tiles: max_rate must be finite and positive, got {max_rate}"
        )));
    }
    if !(parallax_margin >= 0.0 && parallax_margin.is_finite()) {
        return Err(Error::ValueError(format!(
            "Tiles: parallax_margin must be finite and not negative, got {parallax_margin}"
        )));
    }
    if let Some(bad) = sights.iter().position(|sight| !sight.is_usable()) {
        return Err(Error::ValueError(format!(
            "Tiles: sight {bad} has {UNUSABLE}"
        )));
    }
    if let Some(time) = reference
        && !(time.jd().is_finite() && time.jd().abs() < MAX_DAYS)
    {
        return Err(Error::ValueError(format!(
            "Tiles: the reference time must be within 9e15 days of JD 0, got {}",
            time.jd()
        )));
    }
    if let Some(nside) = nside
        && !(nside.is_power_of_two() && nside <= MAX_NSIDE)
    {
        return Err(Error::ValueError(format!(
            "Tiles: nside must be a power of two up to 2^29, got {nside}"
        )));
    }
    let (first, last) = first_and_last(sights.iter().map(|sight| sight.time.jd()));
    if sights.is_empty() {
        return Ok(Vec::new());
    }
    let reference = reference.map_or(f64::midpoint(first, last), |time| time.jd());
    let halo = |jd: f64| max_rate * (jd - reference).abs() + parallax_margin + ROUNDING;
    let largest = halo(first).max(halo(last));
    let nside = nside.unwrap_or_else(|| default_nside(largest));
    let reach = max_pixel_radius(nside);
    let disc = (reach + largest).min(PI);
    let tests = n_pixels(nside) as f64 * (1.0 - disc.cos()) / 2.0;
    if tests > MAX_TESTS_PER_SIGHT {
        return Err(Error::ValueError(format!(
            "Tiles: at nside {nside} a sight would be tested against about {tests:.0} tiles, \
             take a smaller nside"
        )));
    }
    let joined: Vec<Vec<u64>> = sights
        .par_iter()
        .map(|sight| {
            let halo = halo(sight.time.jd());
            let (ra, dec) = sight.direction.to_ra_dec();
            let mut pixels = query_disc_nested(nside, ra, dec, (reach + halo).min(PI));
            pixels.retain(|&pixel| {
                let (center, radius) = core(nside, pixel);
                angle(&center, &sight.direction) <= radius + halo
            });
            pixels
        })
        .collect();
    let mut tiles: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
    for (index, pixels) in joined.into_iter().enumerate() {
        for pixel in pixels {
            tiles.entry(pixel).or_default().push(index);
        }
    }
    Ok(tiles.into_values().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sight at `days` after JD 2460000.5 from the Earth's distance,
    /// toward `ra` and `dec` in degrees.
    fn sight(days: f64, ra: f64, dec: f64) -> Sight {
        Sight {
            time: Time::<TDB>::new(2460000.5 + days),
            observer: Vector::new([1.0, 0.0, 0.0]),
            direction: Vector::from_ra_dec(ra.to_radians(), dec.to_radians()),
            sigma: 1e-6,
        }
    }

    /// The tile holding each of these indices, or `None`.
    fn holding(tiles: &[Vec<usize>], indices: &[usize]) -> Vec<usize> {
        (0..tiles.len())
            .filter(|&k| indices.iter().all(|i| tiles[k].binary_search(i).is_ok()))
            .collect()
    }

    #[test]
    fn core_radius_reaches_its_boundary() {
        // The vertices of the pixels sixteen times finer lie on and inside a
        // pixel, along its edges too. None is farther from its center than
        // the radius of the core.
        for nside in [1, 2, 4, 8] {
            let fine = 16 * nside;
            let mut largest = vec![0.0_f64; n_pixels(nside) as usize];
            let centers: Vec<(Vector<Equatorial>, f64)> =
                (0..n_pixels(nside)).map(|p| core(nside, p)).collect();
            for child in 0..n_pixels(fine) {
                let parent = (child / 256) as usize;
                for (ra, dec) in nested_vertices(fine, child) {
                    let off = angle(&centers[parent].0, &Vector::from_ra_dec(ra, dec));
                    largest[parent] = largest[parent].max(off);
                }
            }
            for (parent, (_, radius)) in centers.iter().enumerate() {
                assert!(
                    largest[parent] <= radius + 1e-12,
                    "nside {nside} pixel {parent}: {} > {radius}",
                    largest[parent]
                );
                assert!(largest[parent] > radius - 1e-12, "the vertices are sampled");
            }
        }
    }

    #[test]
    fn moving_objects_are_whole_in_the_tile_of_their_reference_direction() {
        // Objects moving at up to the rate limit in every direction, from
        // reference directions over the whole sky, seen over ten days. Each
        // is whole in the tile whose core holds its reference direction.
        let rate = 1.0_f64.to_radians();
        let nside = 8;
        let mut sights = Vec::new();
        let mut objects = Vec::new();
        for k in 0..400 {
            let ra = (f64::from(k) * 137.508) % 360.0;
            let dec = (2.0 * ((f64::from(k) + 0.5) / 400.0) - 1.0)
                .asin()
                .to_degrees();
            let heading = (f64::from(k) * 61.0).to_radians();
            let start = sights.len();
            for day in 0..=10 {
                // Move along a great circle at the rate limit, from day 5.
                let from = Vector::<Equatorial>::from_ra_dec(ra.to_radians(), dec.to_radians());
                let pole = Vector::new([0.0, 0.0, 1.0]);
                let east = pole.cross(&from).normalize();
                let north = from.cross(&east);
                let toward = east * heading.cos() + north * heading.sin();
                let turn = rate * (f64::from(day) - 5.0);
                let direction = (from * turn.cos() + toward * turn.sin()).normalize();
                sights.push(Sight {
                    direction,
                    ..sight(f64::from(day), 0.0, 0.0)
                });
            }
            let (pra, pdec) = (ra.to_radians(), dec.to_radians());
            objects.push((
                kete_stats::healpix::ang_to_nested(nside, pra, pdec),
                (start..sights.len()).collect::<Vec<_>>(),
            ));
        }
        let fast = tiles(&sights, rate, 0.0, Some(nside), None).unwrap();
        assert!(fast.len() > 100);
        for (pixel, members) in &objects {
            let whole = holding(&fast, members);
            assert!(
                !whole.is_empty(),
                "object of pixel {pixel} is in no tile whole"
            );
        }
        // A rate limit below the motion leaves objects split.
        let slow = tiles(&sights, 0.5 * rate, 0.0, Some(nside), None).unwrap();
        let split = objects
            .iter()
            .filter(|(_, members)| holding(&slow, members).is_empty())
            .count();
        assert!(split > 0);
    }

    #[test]
    fn halo_grows_with_the_time_from_the_reference() {
        // One sight at the center of a pixel, at the reference time and
        // later. At the reference time it is in its own tile and those whose
        // cores reach it, and later in more.
        let nside = 16;
        let (ra, dec) = nested_to_ang(nside, 1000);
        let (ra, dec) = (ra.to_degrees(), dec.to_degrees());
        let rate = 0.5_f64.to_radians();
        let count = |days: f64, margin: f64| {
            let sights = [sight(0.0, ra, dec), sight(days, ra, dec)];
            let found = tiles(
                &sights,
                rate,
                margin,
                Some(nside),
                Some(Time::new(2460000.5)),
            )
            .unwrap();
            found.iter().filter(|tile| tile.contains(&1)).count()
        };
        let at = count(0.0, 0.0);
        assert!((1..=9).contains(&at), "{at}");
        assert!(count(5.0, 0.0) > at);
        assert!(count(20.0, 0.0) > count(5.0, 0.0));
        // A parallax margin widens the halo as time does.
        assert_eq!(count(0.0, 5.0 * rate), count(5.0, 0.0));
        // The reference time defaults to the middle of the span, so two
        // sights equally far from it have equal halos.
        let sights = [sight(-5.0, ra, dec), sight(5.0, ra, dec)];
        let found = tiles(&sights, rate, 0.0, Some(nside), None).unwrap();
        assert!(found.iter().all(|tile| tile == &vec![0, 1]));
    }

    #[test]
    fn default_nside_keeps_cores_larger_than_the_halo() {
        for halo in [0.0, 1e-4, 0.01, 0.1, 0.3, 1.0, 10.0] {
            let nside = default_nside(halo);
            let size = |n: u32| (4.0 * PI / n_pixels(n) as f64).sqrt();
            assert!(nside.is_power_of_two() && nside <= MAX_DEFAULT_NSIDE);
            assert!(nside == 1 || size(nside) >= CORE_OVER_HALO * halo);
            assert!(nside == MAX_DEFAULT_NSIDE || size(2 * nside) < CORE_OVER_HALO * halo);
        }
    }

    #[test]
    fn parallax_margin_is_the_largest_angle() {
        assert_eq!(parallax_margin(0.0, 0.5).unwrap(), 0.0);
        let margin = parallax_margin(0.01, 0.5).unwrap();
        assert!((margin - 0.02_f64.asin()).abs() < 1e-15);
        assert_eq!(parallax_margin(1.0, 0.5).unwrap(), PI);
        assert!(parallax_margin(-1.0, 0.5).is_err());
        assert!(parallax_margin(0.01, 0.0).is_err());
        assert!(parallax_margin(f64::NAN, 0.5).is_err());
    }

    #[test]
    fn tiles_refuse_bad_inputs() {
        let sights = [sight(0.0, 10.0, 10.0), sight(1.0, 10.0, 10.0)];
        let message = |result: KeteResult<Vec<Vec<usize>>>| result.unwrap_err().to_string();
        assert!(message(tiles(&sights, -1.0, 0.0, None, None)).contains("max_rate"));
        assert!(message(tiles(&sights, f64::NAN, 0.0, None, None)).contains("max_rate"));
        assert!(message(tiles(&sights, 0.0, 0.0, None, None)).contains("max_rate"));
        assert!(message(tiles(&sights, 0.01, f64::INFINITY, None, None)).contains("parallax"));
        assert!(message(tiles(&sights, 0.01, 0.0, Some(3), None)).contains("power of two"));
        assert!(message(tiles(&sights, 0.01, 0.0, Some(0), None)).contains("power of two"));
        assert!(message(tiles(&sights, 0.5, 0.0, Some(1024), None)).contains("smaller nside"));
        for bad in [1e16, f64::NAN, f64::INFINITY] {
            let reference = Some(Time::new(bad));
            assert!(message(tiles(&sights, 0.01, 0.0, None, reference)).contains("reference"));
        }
        let mut bad = sights;
        bad[1].sigma = 0.0;
        assert!(message(tiles(&bad, 0.01, 0.0, None, None)).contains("sight 1"));
        assert!(tiles(&[], 0.01, 0.0, None, None).unwrap().is_empty());
    }
}
