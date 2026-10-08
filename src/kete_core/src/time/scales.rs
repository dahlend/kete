// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-FileCopyrightText: 2025 California Institute of Technology
// SPDX-License-Identifier: BSD-3-Clause

//! Available time scales: [`TDB`], [`TT`], [`TAI`] and [`UTC`].
//!
//! Each scale is defined by its offset from TDB. TT differs from TDB by a periodic
//! term of amplitude about 1.7 ms with a period of a year, from the relativistic
//! motion of an observer on Earth relative to the solar system barycenter. TAI is
//! offset from TT by a constant 32.184 s, and UTC from TAI by the leap seconds.
//! Conversions between any two scales go through TDB and include the periodic term.

use super::leap_second::{tai_minus_utc_at_tai, tai_minus_utc_at_utc};

/// Definitional offset from TT to TAI.
///
/// ``TT = TAI + TT_TO_TAI``
/// ``TAI = TT - TT_TO_TAI``
pub(crate) const TT_TO_TAI: f64 = 32.184 / 86400.0;

/// ``TDB - TT`` in days, at the given julian date.
///
/// The term is periodic over a year. Contributions of shorter period are not
/// included.
///
/// ``TDB - TT = K sin(E)``, where ``E`` is the eccentric anomaly of the
/// heliocentric orbit of the Earth-Moon barycenter. The form and constants
/// are from Moyer (1981), Celestial Mechanics 23, 33-56 and 57-68.
///
/// `jd` is nominally TDB. The epoch is used as given rather than iterated to
/// self-consistency; `tdb_minus_tt_is_insensitive_to_the_epoch_scale` bounds
/// what that costs.
fn tdb_minus_tt(jd: f64) -> f64 {
    /// Julian date of the J2000 epoch, the origin of the mean anomaly.
    const J2000_JD: f64 = 2_451_545.0;

    /// Amplitude of the periodic term, seconds.
    const K: f64 = 1.657e-3;

    /// Eccentricity of the orbit.
    const EB: f64 = 1.671e-2;

    /// Mean anomaly at J2000, radians.
    const M0: f64 = 6.239996;

    /// Rate of change of the mean anomaly, radians per second.
    const M1: f64 = 1.990_968_71e-7;

    let seconds_past_j2000 = (jd - J2000_JD) * 86400.0;
    let mean_anom = M0 + M1 * seconds_past_j2000;
    let ecc_anom = EB.mul_add(mean_anom.sin(), mean_anom);
    K * ecc_anom.sin() / 86400.0
}

/// Offset from JD to MJD
///
/// ``MJD = JD + JD_TO_MJD``
pub const JD_TO_MJD: f64 = -2_400_000.5;

/// Time Scaling support, all time scales must implement this.
///
/// A scale is described by its offset from TDB. The offsets change slowly, so they
/// are evaluated at the Julian date as a single f64 and added to the full precision
/// time.
pub trait TimeScale: Clone + Copy + std::fmt::Debug + PartialEq {
    /// This scale minus TDB, in days, at the TDB Julian date `jd`.
    fn from_tdb_offset(jd: f64) -> f64;

    /// TDB minus this scale, in days, at the Julian date `jd` on this scale.
    ///
    /// The default inverts [`Self::from_tdb_offset`] by fixed point, evaluating it at
    /// the time as given and then at the corrected time. That is exact to f64
    /// precision for an offset that changes smoothly and slowly. A scale whose offset
    /// steps, such as UTC at a leap second, defines this directly.
    #[must_use]
    fn to_tdb_offset(jd: f64) -> f64 {
        let offset = Self::from_tdb_offset(jd);
        -Self::from_tdb_offset(jd - offset)
    }
}

/// TDB Scaled JD time.
///
/// This is in good agreement with "Ephemeris Time" - Which is commonly referred
/// to as the time units used by the JPL Ephemeris, and is used in kete as the
/// base time scaling.
///
/// This is essentially the rate of time from an observer not on the surface of
/// the Earth (and as a result doesn't feel the relativistic dilation effects of
/// Earth).
///
/// This differs from TT by a periodic term of amplitude just under two
/// milliseconds, with a period of a year. See [`TT`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TDB;

impl TimeScale for TDB {
    fn from_tdb_offset(_jd: f64) -> f64 {
        0.0
    }
}

/// UTC Scaled JD time.
///
/// The international standard for communicating time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UTC;

impl TimeScale for UTC {
    fn from_tdb_offset(jd: f64) -> f64 {
        // TDB to TAI, then the leap seconds in effect at that TAI time.
        let tdb_to_tai = TAI::from_tdb_offset(jd);
        let leap = tai_minus_utc_at_tai(jd + tdb_to_tai + JD_TO_MJD);
        tdb_to_tai - leap
    }

    /// In the first second after a leap second, both leap second counts are
    /// consistent with the UTC label, so a fixed point cannot choose between them.
    /// The leap second file lists UTC dates, so the count is read directly.
    fn to_tdb_offset(jd: f64) -> f64 {
        let leap = tai_minus_utc_at_utc(jd + JD_TO_MJD);
        let tai = jd + leap;
        leap + TAI::to_tdb_offset(tai)
    }
}

/// TT (Terrestrial Time).
///
/// The time scale of a clock on the geoid, and the parallel time system of
/// some spacecraft clocks. It is offset from TAI by a constant 32.184 seconds
/// and differs from TDB by a periodic term of amplitude just under two
/// milliseconds with a period of a year.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TT;

impl TimeScale for TT {
    fn from_tdb_offset(jd: f64) -> f64 {
        -tdb_minus_tt(jd)
    }
}

/// TAI Time
/// This is the international standard for the measurement of time.
/// Atomic clocks around the world keep track of this time, which then gets
/// converted to UTC time which is commonly used.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TAI;

impl TimeScale for TAI {
    fn from_tdb_offset(jd: f64) -> f64 {
        TT::from_tdb_offset(jd) - TT_TO_TAI
    }
}

/// Secular drift rate between TCB and TDB, from IAU 2006 Resolution B3.
///
/// ``TCB - TDB = L_B_TCB * (TCB - TCB_EPOCH)``
const L_B_TCB: f64 = 1.550_519_768e-8;

/// Reference epoch for the TCB/TDB linear relation, in JD.
///
/// This is J1977 January 1.0003725 TDB = JD 2443144.5003725.
const TCB_EPOCH: f64 = 2_443_144.500_372_5;

/// TCB (Barycentric Coordinate Time).
///
/// TCB is the coordinate time of the solar system barycenter frame,
/// defined by IAU 2006 Resolution B3. It runs faster than TDB by the
/// secular drift rate `L_B` = 1.550519768e-8. At J2000.0 TCB is roughly
/// 11.3 seconds ahead of TDB; the drift accumulates at about 0.49 s/yr.
///
/// Conversions use the linear relation:
///   `TCB - TDB = L_B * (TCB - T_0)`
/// where `T_0` = 2443144.5003725 JD (J1977.0 TDB). The remaining periodic
/// deviation is below 2 ms and is not corrected here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TCB;

impl TimeScale for TCB {
    fn from_tdb_offset(jd: f64) -> f64 {
        // TCB = (TDB - L_B * T_0) / (1 - L_B), so TCB - TDB = L_B (TDB - T_0) / (1 - L_B)
        L_B_TCB * (jd - TCB_EPOCH) / (1.0 - L_B_TCB)
    }
}

#[cfg(test)]
mod tests {
    use super::super::Time;
    use super::{TAI, TDB, TT, TimeScale, tdb_minus_tt};

    /// ``TDB - TT`` in seconds, from an independent evaluation of the Moyer
    /// expression at each epoch.
    ///
    /// The first four are a quarter of an anomalistic year apart, so they
    /// sample the term at four phases rather than repeating one.
    const REFERENCE_TDB_MINUS_TT: [(f64, f64); 7] = [
        (2_451_545.0, -7.273_677_616_69e-5),
        (2_451_636.0, 1.656_156_033_28e-3),
        (2_451_727.0, 8.798_018_097_88e-5),
        (2_451_818.0, -1.652_203_500_27e-3),
        (2_440_587.5, -6.437_301_635_74e-5),
        (2_457_316.5, -1.600_563_526_15e-3),
        (2_469_807.5, -8.678_436_279_30e-5),
    ];

    #[test]
    fn tdb_minus_tt_matches_reference() {
        for (jd, expected) in REFERENCE_TDB_MINUS_TT {
            let got = tdb_minus_tt(jd) * 86400.0;
            // The reference values were read at an epoch held as a double, so
            // they carry tens of nanoseconds of rounding.
            assert!(
                (got - expected).abs() < 1e-7,
                "jd {jd}: got {got:e} want {expected:e}"
            );
        }
    }

    /// The term is periodic with a one year period and an amplitude just under
    /// two milliseconds, so it must not accumulate.
    #[test]
    fn tdb_minus_tt_is_bounded_and_periodic() {
        let mut max = 0.0_f64;
        for step in 0..4000 {
            let jd = 2_451_545.0 + f64::from(step) * 10.0;
            max = max.max((tdb_minus_tt(jd) * 86400.0).abs());
        }
        assert!(max < 1.7e-3, "amplitude {max:e} exceeds the expected bound");
        assert!(max > 1.6e-3, "amplitude {max:e} is implausibly small");
    }

    /// The epoch is fed in on whatever scale the caller holds, so the result
    /// must not depend much on which one that is. The scales sit at most a
    /// couple of minutes apart, counting leap seconds and the TT offset.
    #[test]
    fn tdb_minus_tt_is_insensitive_to_the_epoch_scale() {
        let mut worst = 0.0_f64;
        for step in 0..2000 {
            let jd = 2_451_545.0 + f64::from(step) * 0.2;
            for offset_seconds in [7.3e-5, 32.184, 64.184, 11.25] {
                let shifted = jd + offset_seconds / 86400.0;
                let diff = (tdb_minus_tt(shifted) - tdb_minus_tt(jd)).abs() * 86400.0;
                worst = worst.max(diff);
            }
        }
        assert!(
            worst < 1e-7,
            "epoch scale changed the result by {worst:e} s"
        );
    }

    #[test]
    fn tt_round_trips_through_tdb() {
        for step in 0..100 {
            let jd = 2_451_545.0 + f64::from(step) * 37.0;
            let tt = Time::<TT>::new(jd);
            let back = tt.tdb().tt();
            assert!(
                ((back - tt).elapsed).abs() < 1e-15,
                "round trip drifted at jd {jd}"
            );
        }
    }

    /// TAI and TDB differ by the constant TT offset plus the periodic term, so
    /// the gap must vary by the full amplitude over a year and never be fixed.
    #[test]
    fn tai_to_tdb_carries_the_periodic_term() {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for step in 0..400 {
            let jd = 2_451_545.0 + f64::from(step);
            let offset = -TAI::from_tdb_offset(jd) * 86400.0;
            lo = lo.min(offset);
            hi = hi.max(offset);
        }
        let spread = hi - lo;
        assert!(
            spread > 3.2e-3 && spread < 3.4e-3,
            "spread over a year was {spread:e}, expected twice the amplitude"
        );
    }

    #[test]
    fn tdb_is_the_identity() {
        assert_eq!(TDB::from_tdb_offset(2_457_316.5), 0.0);
    }
}
