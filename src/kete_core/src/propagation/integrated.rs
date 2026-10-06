//! An [`Ephemeris`] that integrates the planets itself, from a saved set of states.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use nalgebra::{DVector, Vector3};

use super::batch::integrate_packed;
use crate::desigs::Desig;
use crate::ephemeris::Ephemeris;
use crate::errors::{Error, KeteResult};
use crate::forces::{GravParams, NonGravKind};
use crate::frames::Equatorial;
use crate::integrators::RadauDense;
use crate::state::State;
use crate::time::{TDB, Time};

/// An [`Ephemeris`] of the Sun, the planets, the Moon, Pluto and the five most massive
/// asteroids, integrated from saved states; it needs no SPICE kernels.
///
/// The integration starts from barycentric states at one epoch, stored in
/// `data/planet_states.tsv` (from JPL DE440 and JPL Horizons), and uses the batch
/// N-body force model: point masses, solar general relativity, and solar, Jovian and
/// Earth J2. States are served from the integrator's dense output.
///
/// Time is cut into segments of fixed length counted outward from the saved epoch. A
/// segment is integrated from a kept checkpoint at its inner boundary, so rebuilding it
/// gives bit-identical results and an answer never depends on earlier queries. Built
/// segments are cached up to a limit, beyond which the one furthest in time from the
/// newest is dropped. Share one instance between threads, so that each segment is
/// built once.
///
/// Against DE440 over +/-99 years from J2000, heliocentric positions agree within
/// 6-13 km for Mercury, Venus, Mars and Saturn, 50 km for the Earth and Jupiter, 125 km
/// for Uranus and 800-900 km for Neptune and Pluto, and the Moon about the Earth within
/// 400 km.
///
/// Serves NAIF ids 10 (Sun), 1, 2, 4-9 (planet system barycenters and Pluto), 399
/// (Earth), 301 (Moon), 3 (Earth-Moon barycenter), 20000001, 20000002, 20000004,
/// 20000010 and 20000704 (Ceres, Pallas, Vesta, Hygiea, Interamnia), on equatorial
/// axes. Center 0 is the center of mass of these bodies, which is up to about 140 km
/// from DE440's solar system barycenter, so barycentric states from elsewhere should be
/// brought in relative to the Sun.
#[derive(Debug)]
pub struct IntegratedEphemeris {
    /// Integrated bodies, in packed order, the Sun first.
    bodies: Vec<GravParams>,
    /// Slot of each served NAIF id.
    slots: HashMap<i32, usize>,
    /// Slots of the Earth and Moon. The Moon is integrated relative to the Earth, and
    /// the two give the Earth-Moon barycenter.
    earth_moon: (usize, usize),
    segment_days: f64,
    max_segments: usize,
    instance: u64,
    store: RwLock<Store>,
    /// Held while integrating, so a segment is built once.
    build: Mutex<()>,
}

impl IntegratedEphemeris {
    /// Default segment length in days.
    pub const DEFAULT_SEGMENT_DAYS: f64 = 365.25;

    /// Default number of segments kept in memory.
    pub const DEFAULT_MAX_SEGMENTS: usize = 300;

    /// An ephemeris with the default segment length and cache size.
    ///
    /// # Panics
    /// Panics if the built-in mass table lacks one of the saved bodies, which would be
    /// a build defect.
    #[must_use]
    pub fn new() -> Self {
        Self::with_cache(Self::DEFAULT_SEGMENT_DAYS, Self::DEFAULT_MAX_SEGMENTS)
    }

    /// An ephemeris integrated in segments of `segment_days`, keeping at most
    /// `max_segments` segments (at least 1) in memory; beyond that the one furthest in
    /// time from the newest is dropped.
    ///
    /// # Panics
    /// Panics if `segment_days` is not positive and finite, or if the built-in mass
    /// table lacks one of the saved bodies, which would be a build defect.
    #[must_use]
    pub fn with_cache(segment_days: f64, max_segments: usize) -> Self {
        assert!(
            segment_days.is_finite() && segment_days > 0.0,
            "segment_days must be positive, got {segment_days}"
        );
        let known = GravParams::known_masses();
        let bodies: Vec<GravParams> = SAVED
            .iter()
            .map(|state| {
                let Desig::Naif(id) = state.desig else {
                    unreachable!("saved states are designated by NAIF id")
                };
                known
                    .iter()
                    .find(|p| p.naif_id == id)
                    .unwrap_or_else(|| panic!("no mass for NAIF id {id}"))
                    .clone()
            })
            .collect();
        assert_eq!(
            bodies[0].naif_id, 10,
            "the Sun must be the first saved state"
        );
        let slots: HashMap<i32, usize> = bodies
            .iter()
            .enumerate()
            .map(|(slot, p)| (p.naif_id, slot))
            .collect();
        let earth_moon = (slots[&399], slots[&301]);

        let n = bodies.len();
        let mut pos = DVector::zeros(3 * n);
        let mut vel = DVector::zeros(3 * n);
        for (slot, state) in SAVED.iter().enumerate() {
            pos.fixed_rows_mut::<3>(3 * slot)
                .copy_from(&Vector3::from(state.pos));
            vel.fixed_rows_mut::<3>(3 * slot)
                .copy_from(&Vector3::from(state.vel));
        }
        let mut store = Store::default();
        let _ = store.checkpoints.insert(0, Arc::new((pos, vel)));

        Self {
            bodies,
            slots,
            earth_moon,
            segment_days,
            max_segments: max_segments.max(1),
            instance: NEXT_INSTANCE.fetch_add(1, Ordering::Relaxed),
            store: RwLock::new(store),
            build: Mutex::new(()),
        }
    }

    /// Number of segments currently held.
    ///
    /// # Panics
    /// Panics if the lock is poisoned.
    #[must_use]
    pub fn n_cached_segments(&self) -> usize {
        self.store.read().unwrap().segments.len()
    }
}

impl Default for IntegratedEphemeris {
    fn default() -> Self {
        Self::new()
    }
}

impl Ephemeris for IntegratedEphemeris {
    fn try_get_state_with_center(
        &self,
        id: i32,
        time: Time<TDB>,
        center: i32,
    ) -> KeteResult<State<Equatorial>> {
        let states = self.barycentric_states(time)?;
        let state_of = |id: i32| -> KeteResult<[f64; 6]> {
            match id {
                0 => Ok([0.0; 6]),
                // The Earth-Moon barycenter.
                3 => {
                    let (earth, moon) = self.earth_moon;
                    let (m_e, m_m) = (self.bodies[earth].mass, self.bodies[moon].mass);
                    Ok(std::array::from_fn(|k| {
                        (m_e * states[earth][k] + m_m * states[moon][k]) / (m_e + m_m)
                    }))
                }
                _ => self
                    .slots
                    .get(&id)
                    .map(|slot| states[*slot])
                    .ok_or_else(|| {
                        Error::Bounds(format!(
                            "NAIF id {id} is not covered by the integrated ephemeris."
                        ))
                    }),
            }
        };
        let (body, origin) = (state_of(id)?, state_of(center)?);
        let rel: [f64; 6] = std::array::from_fn(|k| body[k] - origin[k]);
        Ok(State::new(
            Desig::Naif(id),
            time,
            [rel[0], rel[1], rel[2]],
            [rel[3], rel[4], rel[5]],
            center,
        ))
    }
}

impl IntegratedEphemeris {
    /// Barycentric `[pos, vel]` of every integrated body at `time`, the center of mass
    /// of the integrated bodies at the origin.
    fn barycentric_states(&self, time: Time<TDB>) -> KeteResult<Arc<Vec<[f64; 6]>>> {
        if !time.jd().is_finite() {
            Err(Error::ValueError(format!(
                "Time {} is not finite.",
                time.jd()
            )))?;
        }
        #[allow(clippy::cast_possible_truncation, reason = "checked to be finite")]
        let index = ((time - SAVED[0].epoch).elapsed / self.segment_days).floor() as i64;
        let mut held = None;
        let cached = LAST_USE.with_borrow(|last| {
            let last = last
                .as_ref()
                .filter(|last| last.instance == self.instance)?;
            if last.time.same_instant(&time) {
                return Some(Arc::clone(&last.states));
            }
            if last.index == index {
                held = Some(Arc::clone(&last.segment));
            }
            None
        });
        if let Some(states) = cached {
            return Ok(states);
        }
        let segment = match held {
            Some(segment) => segment,
            None => self.segment(index)?,
        };
        // The dense output holds the Moon relative to the Earth.
        let (mut pos, mut vel) = segment.evaluate(time)?;
        let (earth, moon) = self.earth_moon;
        for k in 0..3 {
            pos[3 * moon + k] += pos[3 * earth + k];
            vel[3 * moon + k] += vel[3 * earth + k];
        }
        let total: f64 = self.bodies.iter().map(|p| p.mass).sum();
        let mut com = [0.0; 6];
        for (slot, body) in self.bodies.iter().enumerate() {
            for k in 0..3 {
                com[k] += body.mass * pos[3 * slot + k];
                com[3 + k] += body.mass * vel[3 * slot + k];
            }
        }
        let states: Arc<Vec<[f64; 6]>> = Arc::new(
            (0..self.bodies.len())
                .map(|slot| {
                    std::array::from_fn(|k| {
                        if k < 3 {
                            pos[3 * slot + k] - com[k] / total
                        } else {
                            vel[3 * slot + k - 3] - com[k] / total
                        }
                    })
                })
                .collect(),
        );
        LAST_USE.set(Some(LastUse {
            instance: self.instance,
            index,
            segment,
            time,
            states: Arc::clone(&states),
        }));
        Ok(states)
    }

    /// Dense output of segment `index`, built (with every checkpoint on the way) if not
    /// cached.
    fn segment(&self, index: i64) -> KeteResult<Arc<RadauDense>> {
        let read = || self.store.read().map_err(|_| Error::LockFailed);
        if let Some(segment) = read()?.segments.get(&index) {
            return Ok(Arc::clone(segment));
        }
        let _guard = self.build.lock().map_err(|_| Error::LockFailed)?;
        if let Some(segment) = read()?.segments.get(&index) {
            return Ok(Arc::clone(segment));
        }
        // Segment `i` is integrated outward from the saved epoch: forward from boundary
        // `i`, or backward from boundary `i + 1`.
        let (target, step) = if index >= 0 {
            (index, 1)
        } else {
            (index + 1, -1)
        };
        let mut boundary = target;
        while !read()?.checkpoints.contains_key(&boundary) {
            boundary -= step;
        }
        #[allow(
            clippy::cast_precision_loss,
            reason = "boundary counts are far below 2^52"
        )]
        let boundary_time = |b: i64| SAVED[0].epoch + b as f64 * self.segment_days;
        loop {
            let start = Arc::clone(&read()?.checkpoints[&boundary]);
            let mut dense = RadauDense::new();
            let (pos, vel) = integrate_packed::<NonGravKind>(
                &self.bodies,
                Vec::new(),
                start.0.clone(),
                start.1.clone(),
                boundary_time(boundary),
                boundary_time(boundary + step),
                Some(&mut dense),
            )?;
            let mut store = self.store.write().map_err(|_| Error::LockFailed)?;
            let _ = store
                .checkpoints
                .entry(boundary + step)
                .or_insert_with(|| Arc::new((pos, vel)));
            if boundary == target {
                let segment = Arc::new(dense);
                let _ = store.segments.insert(index, Arc::clone(&segment));
                if store.segments.len() > self.max_segments {
                    let furthest = store
                        .segments
                        .keys()
                        .copied()
                        .max_by_key(|key| key.abs_diff(index));
                    if let Some(furthest) = furthest {
                        let _ = store.segments.remove(&furthest);
                    }
                }
                return Ok(segment);
            }
            boundary += step;
        }
    }
}

/// The barycentric states the integration starts from, all at one epoch, see
/// `data/planet_states.tsv`.
static SAVED: std::sync::LazyLock<Vec<State<Equatorial>>> = std::sync::LazyLock::new(|| {
    let text = include_str!("../../data/planet_states.tsv");
    let states: Vec<State<Equatorial>> = text
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.trim_start().starts_with('#'))
        .map(|line| {
            let fields: Vec<&str> = line.split_whitespace().collect();
            assert_eq!(fields.len(), 8, "planet_states.tsv: bad row {line:?}");
            let parse = |idx: usize| -> f64 {
                fields[idx]
                    .parse()
                    .unwrap_or_else(|e| panic!("planet_states.tsv: bad value in {line:?}: {e}"))
            };
            let id: i32 = fields[0]
                .parse()
                .unwrap_or_else(|e| panic!("planet_states.tsv: bad id in {line:?}: {e}"));
            State::new(
                Desig::Naif(id),
                Time::new(parse(1)),
                [parse(2), parse(3), parse(4)],
                [parse(5), parse(6), parse(7)],
                0,
            )
        })
        .collect();
    assert!(
        !states.is_empty()
            && states
                .iter()
                .all(|state| state.epoch.same_instant(&states[0].epoch)),
        "planet_states.tsv: needs states, all at the same epoch"
    );
    states
});

/// Checkpoints and cached segments, behind the ephemeris' lock.
#[derive(Debug, Default)]
struct Store {
    /// Absolute packed `(pos, vel)` of every body at each segment boundary, by boundary
    /// index (boundary `b` is at `b * segment_days` from the saved epoch).
    checkpoints: HashMap<i64, Arc<(DVector<f64>, DVector<f64>)>>,
    /// Dense output of each segment by index (segment `i` spans boundaries `i` to
    /// `i + 1`).
    segments: HashMap<i64, Arc<RadauDense>>,
}

/// Source of distinct ids for [`IntegratedEphemeris`] instances, which key each
/// thread's [`LastUse`].
static NEXT_INSTANCE: AtomicU64 = AtomicU64::new(0);

/// A thread's last use of an [`IntegratedEphemeris`]: the segment it read, held so
/// that queries within it skip the shared store, and the states it last evaluated.
struct LastUse {
    instance: u64,
    index: i64,
    segment: Arc<RadauDense>,
    time: Time<TDB>,
    states: Arc<Vec<[f64; 6]>>,
}

thread_local! {
    /// This thread's last use of an ephemeris.
    static LAST_USE: std::cell::RefCell<Option<LastUse>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::prelude::*;

    const BODIES: [i32; 17] = [
        10, 1, 2, 3, 399, 301, 4, 5, 6, 7, 8, 9, 20000001, 20000002, 20000004, 20000010, 20000704,
    ];

    fn helio(eph: &IntegratedEphemeris, id: i32, time: Time<TDB>) -> Vector3<f64> {
        eph.try_get_state_with_center(id, time, 10)
            .unwrap()
            .pos
            .into()
    }

    /// At the saved epoch the served states are the saved ones, relative to each other,
    /// to roundoff.
    #[test]
    fn reproduces_the_saved_states() {
        let eph = IntegratedEphemeris::new();
        let t0 = SAVED[0].epoch;
        let sun = Vector3::from(SAVED[0].pos);
        for state in SAVED.iter() {
            let Desig::Naif(id) = state.desig else {
                unreachable!()
            };
            let served = helio(&eph, id, t0);
            let saved = Vector3::from(state.pos) - sun;
            let tol = 1e-15 * saved.norm().max(1.0);
            assert!(
                (served - saved).norm() < tol,
                "NAIF {id}: {:e} AU off",
                (served - saved).norm()
            );
        }
    }

    /// Answers do not depend on query history: evicting and rebuilding a segment, or
    /// asking in a different order, gives bit-identical states.
    #[test]
    fn rebuilt_segments_are_bit_identical() {
        let t0 = SAVED[0].epoch;
        let times = [t0 + 800.3, t0 - 1100.7, t0 + 10.0, t0 - 3.5];
        let first = IntegratedEphemeris::with_cache(365.25, 1);
        let reference: Vec<_> = times
            .iter()
            .map(|t| first.try_get_state_with_center(5, *t, 10).unwrap())
            .collect();
        // Re-query after the one-segment cache has evicted everything.
        for (t, expected) in times.iter().zip(&reference) {
            let again = first.try_get_state_with_center(5, *t, 10).unwrap();
            assert_eq!(Vector3::from(again.pos), Vector3::from(expected.pos));
            assert_eq!(Vector3::from(again.vel), Vector3::from(expected.vel));
        }
        assert_eq!(first.n_cached_segments(), 1);
        // A fresh instance asked in reverse order.
        let second = IntegratedEphemeris::new();
        for (t, expected) in times.iter().zip(&reference).rev() {
            let state = second.try_get_state_with_center(5, *t, 10).unwrap();
            assert_eq!(Vector3::from(state.pos), Vector3::from(expected.pos));
        }
    }

    /// Integrating in segments agrees with one uninterrupted integration, across
    /// several segment boundaries, to integrator precision.
    #[test]
    fn segments_match_one_integration() {
        let eph = IntegratedEphemeris::new();
        let t0 = SAVED[0].epoch;
        let end = t0 + 3.3 * 365.25;
        let start = Arc::clone(&eph.store.read().unwrap().checkpoints[&0]);
        let (pos, _) = integrate_packed::<NonGravKind>(
            &eph.bodies,
            Vec::new(),
            start.0.clone(),
            start.1.clone(),
            t0,
            end,
            None,
        )
        .unwrap();
        let direct = |id: i32| -> Vector3<f64> {
            let slot = eph.slots[&id];
            Vector3::new(pos[3 * slot], pos[3 * slot + 1], pos[3 * slot + 2])
                - Vector3::new(pos[0], pos[1], pos[2])
        };
        let mut worst = 0.0_f64;
        for id in [1, 2, 399, 301, 4, 5, 8, 9, 20000001] {
            worst = worst.max((helio(&eph, id, end) - direct(id)).norm());
        }
        println!("segments_match_one_integration: worst {worst:.2e} AU after 3.3 yr");
        assert!(
            worst < 1e-10,
            "segmented integration differs by {worst:e} AU"
        );
    }

    /// The center of mass is the origin, the Earth-Moon barycenter lies between the
    /// Earth and the Moon, and unknown bodies are refused.
    #[test]
    fn centers_and_coverage() {
        let eph = IntegratedEphemeris::new();
        let t = SAVED[0].epoch + 4321.0;
        let mut com = Vector3::zeros();
        let mut total = 0.0;
        for body in &eph.bodies {
            let state = eph.try_get_state_with_center(body.naif_id, t, 0).unwrap();
            com += Vector3::from(state.pos) * body.mass;
            total += body.mass;
        }
        assert!((com / total).norm() < 1e-15);
        let earth = helio(&eph, 399, t);
        let moon = helio(&eph, 301, t);
        let emb = helio(&eph, 3, t);
        assert!(((emb - earth).norm() + (moon - emb).norm() - (moon - earth).norm()).abs() < 1e-12);
        assert!(matches!(
            eph.try_get_state_with_center(499, t, 10),
            Err(Error::Bounds(_))
        ));
        let sun_from_earth = eph.try_get_state_with_center(10, t, 399).unwrap();
        assert!((Vector3::from(sun_from_earth.pos) + earth).norm() < 1e-15);
    }

    /// Queries from many threads, building segments concurrently, agree with
    /// sequential ones.
    #[test]
    fn parallel_queries_match_sequential() {
        let t0 = SAVED[0].epoch;
        let times: Vec<Time<TDB>> = (0..64)
            .map(|k: i32| t0 + (f64::from(k) - 32.0) * 47.3)
            .collect();
        let sequential = IntegratedEphemeris::with_cache(365.25, 4);
        let expected: Vec<Vec<Vector3<f64>>> = times
            .iter()
            .map(|t| {
                BODIES
                    .iter()
                    .map(|id| helio(&sequential, *id, *t))
                    .collect()
            })
            .collect();
        let parallel = IntegratedEphemeris::with_cache(365.25, 4);
        let got: Vec<Vec<Vector3<f64>>> = times
            .par_iter()
            .map(|t| BODIES.iter().map(|id| helio(&parallel, *id, *t)).collect())
            .collect();
        assert_eq!(got, expected);
    }
}
