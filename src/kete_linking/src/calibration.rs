// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! The stated uncertainties of the observations scaled per station, from the
//! residuals of the objects the first pass of a search confirms, see
//! [`crate::search()`].

use std::collections::BTreeMap;

use kete_fitting::AstrometricObservation;

use crate::{Confirmed, Sight};

/// Fewest observations of a station, kept by the fits of the confirmed
/// objects of the first pass, from which a factor on its uncertainties is
/// estimated, see [Stations](fn@crate::search#stations). The relative error
/// of the factor falls as one over the square root of their number, and
/// fewer leave a factor which is mostly noise.
const MIN_CALIBRATION_SIGHTS: usize = 100;

/// Fewest confirmed objects of the first pass with observations of a
/// station from which its factor is estimated, see
/// [Stations](fn@crate::search#stations). The residuals of one object are
/// correlated through its orbit, and an object which is a link of two, or
/// has an odd observation the fit kept, adds a run of large ones; a factor
/// from many objects is not set by a few such.
const MIN_CALIBRATION_OBJECTS: usize = 20;

/// The lines of sight of the observations as confirmation sees them: the
/// stated uncertainty of each observation times the factor of its station.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Scaled<'a> {
    /// The line of sight of every optical observation of the candidates,
    /// with its stated uncertainty, and `None` for every other observation,
    /// see [`crate::confirm::sights_of`].
    pub(crate) sights: &'a [Option<Sight>],

    /// The factor on the uncertainties of each observation, one where they
    /// are as stated.
    pub(crate) factors: &'a [f64],
}

impl Scaled<'_> {
    /// Whether observation `index` has a line of sight, as an optical
    /// observation of the candidates does.
    pub(crate) fn is_optical(&self, index: usize) -> bool {
        self.sights[index].is_some()
    }

    /// The line of sight of observation `index` with its scaled uncertainty,
    /// `None` without one.
    pub(crate) fn sight(&self, index: usize) -> Option<Sight> {
        let sight = self.sights[index]?;
        Some(Sight {
            sigma: sight.sigma * self.factors[index],
            ..sight
        })
    }

    /// Observation `index` of `observations` with the uncertainties of both
    /// axes of the sky scaled. Radar observations are not scaled.
    pub(crate) fn observation(
        &self,
        observations: &[AstrometricObservation],
        index: usize,
    ) -> AstrometricObservation {
        let mut observation = observations[index].clone();
        if let AstrometricObservation::Optical {
            sigma_ra,
            sigma_dec,
            ..
        } = &mut observation
        {
            *sigma_ra *= self.factors[index];
            *sigma_dec *= self.factors[index];
        }
        observation
    }
}

/// The factor on the stated uncertainties of each observation, from the fits
/// of `confirmed`, with `stations` the label of the station of each
/// observation, by the estimate of [Stations](fn@crate::search#stations),
/// with the thresholds [`MIN_CALIBRATION_SIGHTS`] and
/// [`MIN_CALIBRATION_OBJECTS`]. The residuals are those of the fits, over
/// the stated uncertainties in `observations`, whatever uncertainties the
/// fits were made with.
pub(crate) fn station_factors(
    observations: &[AstrometricObservation],
    stations: &[usize],
    confirmed: &[Confirmed],
) -> Vec<f64> {
    let kept = confirmed
        .iter()
        .filter_map(|object| kept_chi_squares(observations, object));
    let by_station = factors_by_station(stations, kept);
    stations
        .iter()
        .map(|station| by_station.get(station).copied().unwrap_or(1.0))
        .collect()
}

/// The index of each optical observation the fit of `object` kept, and its
/// chi-square over its stated uncertainties, see [`station_factors`]. `None`
/// if the observations of the fit do not line up with those of the object.
fn kept_chi_squares(
    observations: &[AstrometricObservation],
    object: &Confirmed,
) -> Option<Vec<(usize, f64)>> {
    // The fit holds its observations in time order, of equal times in the
    // order of their indices, see `fit_kept`.
    let epoch = |i: usize| observations[i].epoch().jd();
    let mut kept = object.sights.clone();
    kept.sort_by(|&a, &b| epoch(a).total_cmp(&epoch(b)).then(a.cmp(&b)));
    let fit = &object.fit;
    let fitted: Vec<_> = fit
        .observations
        .iter()
        .zip(&fit.residuals)
        .zip(&fit.included)
        .filter_map(|(pair, &included)| included.then_some(pair))
        .collect();
    if fitted.len() != kept.len() {
        return None;
    }
    let mut found = Vec::with_capacity(kept.len());
    for (&index, (observation, residual)) in kept.iter().zip(fitted) {
        if observation.epoch() != observations[index].epoch() {
            return None;
        }
        // Residuals and uncertainties in right ascension are both along the
        // coordinate, not the sky, so their ratio is that on the sky. The
        // weights hold the correlation of the two axes.
        let stated = &observations[index];
        if !stated.is_radar() {
            if residual.len() != 2 {
                return None;
            }
            let chi2 = residual.dot(&(stated.base_weight_matrix() * residual));
            found.push((index, chi2));
        }
    }
    Some(found)
}

/// The factor of each station seen in the fits, by its label, from the index
/// and chi-square of each kept optical observation of each confirmed object,
/// see [`station_factors`]. A chi-square which is not finite adds nothing,
/// and an object of three or fewer observations, whose orbit absorbs all of
/// their residuals, is passed over.
fn factors_by_station(
    stations: &[usize],
    objects: impl IntoIterator<Item = Vec<(usize, f64)>>,
) -> BTreeMap<usize, f64> {
    /// The sums of one station.
    #[derive(Default)]
    struct Sums {
        chi2: f64,
        expected: f64,
        sights: usize,
        objects: usize,
    }
    let mut sums: BTreeMap<usize, Sums> = BTreeMap::new();
    for kept in objects {
        if kept.len() <= 3 {
            continue;
        }
        let share = 2.0 * (1.0 - 3.0 / kept.len() as f64);
        let mut seen = Vec::new();
        for (index, chi2) in kept {
            if !chi2.is_finite() {
                continue;
            }
            let sum = sums.entry(stations[index]).or_default();
            sum.chi2 += chi2;
            sum.expected += share;
            sum.sights += 1;
            seen.push(stations[index]);
        }
        seen.sort_unstable();
        seen.dedup();
        for station in seen {
            sums.entry(station).or_default().objects += 1;
        }
    }
    sums.into_iter()
        .map(|(station, sum)| {
            let enough =
                sum.sights >= MIN_CALIBRATION_SIGHTS && sum.objects >= MIN_CALIBRATION_OBJECTS;
            let factor = if enough {
                (sum.chi2 / sum.expected).sqrt().max(1.0)
            } else {
                1.0
            };
            (station, factor)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `objects` objects of `n` kept observations each, every observation of
    /// `station`, numbered from `first`, each with the chi-square which makes
    /// the factor of a station of only such objects `factor`.
    fn objects_of(
        station: usize,
        objects: usize,
        n: usize,
        factor: f64,
        first: usize,
        stations: &mut Vec<usize>,
    ) -> Vec<Vec<(usize, f64)>> {
        let chi2 = factor.powi(2) * 2.0 * (1.0 - 3.0 / n as f64);
        (0..objects)
            .map(|k| {
                (0..n)
                    .map(|j| {
                        let index = first + k * n + j;
                        if stations.len() <= index {
                            stations.resize(index + 1, usize::MAX);
                        }
                        stations[index] = station;
                        (index, chi2)
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn station_factors_need_enough_and_only_grow() {
        // Station 3 is twice as scattered as it states, in 25 objects of 6
        // observations, 150 in all, and its factor is 2, with what the fits
        // absorb taken out. Station 5 is less scattered than it states and
        // keeps 1. Station 7 has 33 objects of 4 observations, one of each
        // with a chi-square which is not finite and adds nothing, so 99
        // count, and station 9 has 19 objects of 6: each too few, and keeps 1
        // however scattered. Labels need not be dense.
        let mut stations = Vec::new();
        let mut all = objects_of(3, 25, 6, 2.0, 0, &mut stations);
        all.extend(objects_of(5, 30, 8, 0.7, stations.len(), &mut stations));
        let mut seven = objects_of(7, 33, 4, 3.0, stations.len(), &mut stations);
        for object in &mut seven {
            object[3].1 = f64::NAN;
        }
        all.extend(seven);
        all.extend(objects_of(9, 19, 6, 3.0, stations.len(), &mut stations));
        let factors = factors_by_station(&stations, all.clone());
        assert!((factors[&3] - 2.0).abs() < 1e-12, "{factors:?}");
        assert_eq!(factors[&5], 1.0);
        assert_eq!(factors[&7], 1.0);
        assert_eq!(factors[&9], 1.0);
        assert_eq!(factors.len(), 4);

        // One more object of station 9 reaches 20 objects and 120
        // observations.
        all.extend(objects_of(9, 1, 6, 3.0, stations.len(), &mut stations));
        let factors = factors_by_station(&stations, all);
        assert!((factors[&9] - 3.0).abs() < 1e-12, "{factors:?}");

        // An object of observations of two stations counts as an object of
        // each, and objects of three or fewer observations are passed over.
        let stations = vec![0, 1, 0, 1, 0, 1];
        let mixed = vec![(0, 4.0), (1, 4.0), (2, 4.0), (3, 4.0)];
        let factors = factors_by_station(&stations, vec![mixed.clone(); 25]);
        assert_eq!(factors.len(), 2);
        assert_eq!((factors[&0], factors[&1]), (1.0, 1.0), "50 each: too few");
        let factors = factors_by_station(&stations, vec![mixed; 50]);
        // Each observation of an object of four expects 2 (1 - 3 / 4).
        assert!((factors[&0] - 8.0_f64.sqrt()).abs() < 1e-12, "{factors:?}");
        assert!((factors[&1] - 8.0_f64.sqrt()).abs() < 1e-12);
        let short = vec![(0, 100.0), (1, 100.0), (2, 100.0)];
        assert!(factors_by_station(&stations, vec![short; 200]).is_empty());
    }
}
