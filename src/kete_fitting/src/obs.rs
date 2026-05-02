//! Observation types, predicted measurements, and geometric partial derivatives.
//!
// BSD 3-Clause License
//
// Copyright (c) 2026, Dar Dahlen
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

use kete_core::Band;
use kete_core::constants::{AU_KM, C_AU_PER_DAY, C_AU_PER_DAY_INV, GMS};
use kete_core::desigs::Desig;
use kete_core::frames::{Equatorial, SSB, Vector, geodetic_lat_lon_to_ecef};
use kete_core::prelude::{Error, KeteResult, State};
use kete_core::time::{TDB, Time};
use kete_spice::prelude::{LOADED_PCK, LOADED_SPK};
use nalgebra::{DVector, Matrix2x3, Matrix3x1, RowVector6, Vector3};

/// Solar Schwarzschild radius in AU: ``2 GM_sun / c^2``.
const SHAPIRO_RS_AU: f64 = 2.0 * GMS / (C_AU_PER_DAY * C_AU_PER_DAY);

/// Shapiro (gravitational) range delay along a one-way light path, in AU.
///
/// Returns the contribution that should be **added** to the geometric
/// one-way range `leg` to obtain the predicted radar light-time path
/// length.  The formula follows the standard parametrized post-Newtonian
/// expression with `gamma = 1`:
///
/// ```text
///     dr = (2 GM_sun / c^2) * ln((r1 + r2 + R) / (r1 + r2 - R))
/// ```
///
/// where `r1`, `r2` are the heliocentric distances of the two leg
/// endpoints (AU) and `R = leg` is the geometric leg length (AU).  For
/// near-Earth radar geometries the term is sub-meter; it grows to tens
/// of kilometers when the radar path passes near the Sun.
fn shapiro_range_au(r1: f64, r2: f64, leg: f64) -> f64 {
    let denom = r1 + r2 - leg;
    if denom <= 0.0 {
        return 0.0;
    }
    SHAPIRO_RS_AU * ((r1 + r2 + leg) / denom).ln()
}

/// Differential gravitational light deflection due to the Sun.
///
/// Adjusts the apparent heliocentric position of a Solar System object to
/// account for the difference between the solar gravitational bending of the
/// photon path from the object and the bending from background stars at
/// infinity.  On a plate-solved CCD frame the common-mode bending (same for
/// all reference stars and the object) cancels in the plate solution.  Only
/// the differential term, arising from the object being at finite heliocentric
/// distance rather than at infinity, survives.
///
/// Both `observer_helio` and `obj_lt_pos` are Sun-centered positions in AU.
/// Returns the corrected apparent heliocentric position.
pub(crate) fn differential_light_deflect(
    observer_helio: &Vector<Equatorial>,
    obj_lt_pos: Vector<Equatorial>,
) -> Vector<Equatorial> {
    let bend_factor = 2.0 * GMS / (C_AU_PER_DAY * C_AU_PER_DAY);

    let p = obj_lt_pos - observer_helio;
    let plen = p.norm();
    if plen < 1e-10 {
        return obj_lt_pos;
    }
    let olen = observer_helio.norm();
    let rlen = obj_lt_pos.norm();
    if olen < 1e-10 || rlen < 1e-10 {
        return obj_lt_pos;
    }

    let xprod = observer_helio.cross(&obj_lt_pos);
    let dir_unnorm = p.cross(&xprod);
    let dlen = dir_unnorm.norm();
    if dlen < 1e-30 {
        return obj_lt_pos;
    }
    let dir = dir_unnorm / dlen;

    let psi1 = (obj_lt_pos.dot(observer_helio) / (rlen * olen))
        .clamp(-1.0, 1.0)
        .acos();
    let psi2 = (p.dot(observer_helio) / (plen * olen))
        .clamp(-1.0, 1.0)
        .acos();

    let bending = bend_factor * ((psi2 / 2.0).tan() - (psi1 / 2.0).tan()) * plen;
    obj_lt_pos + dir * bending
}

/// Compute the SSB-centered Equatorial state of an Earth ground station at
/// the given epoch.
///
/// Uses the loaded PCK kernels for Earth orientation (delivers position +
/// inertial velocity, including Earth surface rotation) and the loaded SPK
/// kernels to recenter from geocentric to SSB.  This is the same conversion
/// used by `spice.earth_pos_to_ecliptic` on the Python side.
fn station_state_at(
    lat_rad: f64,
    lon_rad: f64,
    height_km: f64,
    epoch: Time<TDB>,
) -> KeteResult<State<Equatorial, SSB>> {
    let pos_ecef_km = geodetic_lat_lon_to_ecef(lat_rad, lon_rad, height_km);
    let pos_ecef_au: Vector3<f64> =
        Vector3::new(pos_ecef_km[0], pos_ecef_km[1], pos_ecef_km[2]) / AU_KM;

    let pcks = LOADED_PCK.try_read()?;
    let frame = pcks.try_get_orientation(3000, epoch)?;
    let (pos_eq, vel_eq) = frame.to_equatorial(pos_ecef_au, Vector3::zeros())?;

    let geocentric = State::<Equatorial>::new(Desig::Empty, epoch, pos_eq, vel_eq, 399);
    let spks = LOADED_SPK.try_read()?;
    spks.try_to_ssb(geocentric)
}

/// A single astrometric or radar observation.
///
/// Each variant carries the observer geometry, measured values, and
/// uncertainties. The observation epoch is taken from `observer.epoch`.
#[derive(Debug, Clone)]
pub enum AstrometricObservation {
    /// Optical astrometry: RA and Dec on the sky.
    ///
    /// Covers both telescope astrometry (catalog-referenced star positions)
    /// and stellar occultations (timing of an occulted background star);
    /// the math is identical.  The `is_occultation` flag preserves the
    /// distinction so that ingestion paths can apply different rules:
    /// occultations bypass the per-observatory residual table, EFCC18
    /// debiasing, and over-observation reweighting because their sigmas are
    /// already Gaia-anchored and per-event.
    Optical {
        /// Observer state (SSB-centered, Equatorial).
        observer: State<Equatorial, SSB>,
        /// Right ascension (radians).
        ra: f64,
        /// Declination (radians).
        dec: f64,
        /// 1-sigma RA uncertainty in the RA coordinate direction (radians, pure RA, no cos(dec)).
        sigma_ra: f64,
        /// 1-sigma Dec uncertainty (radians).
        sigma_dec: f64,
        /// Correlation coefficient between RA and Dec uncertainties, in
        /// `[-1, 1]`.  Set to `0.0` for axis-aligned (uncorrelated) sigmas.
        /// Values from ADES `rmscorr` or Gaia `ra_dec_corr_*` fields.  For
        /// non-zero values, the base position covariance is tilted:
        /// `Sigma = [[sigma_ra^2, corr * sigma_ra * sigma_dec],
        ///           [corr * sigma_ra * sigma_dec, sigma_dec^2]]`.
        sigma_corr: f64,
        /// 1-sigma timing uncertainty (seconds). When non-zero, the
        /// along-track positional uncertainty is inflated by
        /// `time_sigma * apparent_speed`. Set to 0 to disable.
        time_sigma: f64,
        /// True if this measurement was derived from a stellar occultation
        /// rather than direct astrometry.  Used by ingestion code to
        /// route sigma resolution and bias-correction differently;
        /// fitting math is identical for both cases.
        is_occultation: bool,
        /// Photometric band identifier.
        band: Band,
        /// Apparent magnitude.  `f64::NAN` when unavailable.
        mag: f64,
    },

    /// Radar range measurement.
    ///
    /// The two ground stations are stored as Earth-fixed (WGS84 geodetic)
    /// coordinates.  The receive epoch `t_rx` is stored explicitly; the
    /// transmit epoch `t_tx` is derived inside the residual computation
    /// from the predicted round-trip geometry.  Station inertial states
    /// at the appropriate epochs are produced via PCK lookups during the
    /// residual computation.
    RadarRange {
        /// Transmitter station: WGS84 geodetic latitude (radians).
        xmit_lat_rad: f64,
        /// Transmitter station: WGS84 geodetic longitude (radians).
        xmit_lon_rad: f64,
        /// Transmitter station: height above the WGS84 ellipsoid (km).
        xmit_height_km: f64,
        /// Receiver station: WGS84 geodetic latitude (radians).
        rcvr_lat_rad: f64,
        /// Receiver station: WGS84 geodetic longitude (radians).
        rcvr_lon_rad: f64,
        /// Receiver station: height above the WGS84 ellipsoid (km).
        rcvr_height_km: f64,
        /// Receive epoch `t_rx` (TDB).
        epoch: Time<TDB>,
        /// Measured range (AU).
        range: f64,
        /// 1-sigma range uncertainty (AU).
        sigma_range: f64,
    },

    /// Radar range-rate (Doppler) measurement.
    ///
    /// Same conventions as [`Self::RadarRange`].
    RadarRate {
        /// Transmitter station: WGS84 geodetic latitude (radians).
        xmit_lat_rad: f64,
        /// Transmitter station: WGS84 geodetic longitude (radians).
        xmit_lon_rad: f64,
        /// Transmitter station: height above the WGS84 ellipsoid (km).
        xmit_height_km: f64,
        /// Receiver station: WGS84 geodetic latitude (radians).
        rcvr_lat_rad: f64,
        /// Receiver station: WGS84 geodetic longitude (radians).
        rcvr_lon_rad: f64,
        /// Receiver station: height above the WGS84 ellipsoid (km).
        rcvr_height_km: f64,
        /// Receive epoch `t_rx` (TDB).
        epoch: Time<TDB>,
        /// Measured range-rate (AU/day, positive = receding).
        range_rate: f64,
        /// 1-sigma range-rate uncertainty (AU/day).
        sigma_range_rate: f64,
    },
}

impl AstrometricObservation {
    /// Owned observer state at the observation's primary epoch.
    ///
    /// For optical observations this clones the stored observer state.  For
    /// radar observations the receiver state at `t_rx` is computed via the
    /// loaded PCK and SPK kernels.
    ///
    /// # Errors
    /// Fails if the PCK / SPK lookup for radar fails.
    pub fn observer(&self) -> KeteResult<State<Equatorial, SSB>> {
        match self {
            Self::Optical { observer, .. } => Ok(observer.clone()),
            Self::RadarRange {
                rcvr_lat_rad,
                rcvr_lon_rad,
                rcvr_height_km,
                epoch,
                ..
            }
            | Self::RadarRate {
                rcvr_lat_rad,
                rcvr_lon_rad,
                rcvr_height_km,
                epoch,
                ..
            } => station_state_at(*rcvr_lat_rad, *rcvr_lon_rad, *rcvr_height_km, *epoch),
        }
    }

    /// Observation epoch.  For radar this is the receive epoch `t_rx`.
    pub fn epoch(&self) -> Time<TDB> {
        match self {
            Self::Optical { observer, .. } => observer.epoch,
            Self::RadarRange { epoch, .. } | Self::RadarRate { epoch, .. } => *epoch,
        }
    }

    /// Extract RA, Dec, and observer state from an Optical observation.
    ///
    /// # Errors
    /// Returns an error if the observation is not Optical.
    pub fn as_optical(&self) -> KeteResult<(f64, f64, &State<Equatorial, SSB>)> {
        match self {
            Self::Optical {
                observer, ra, dec, ..
            } => Ok((*ra, *dec, observer)),
            Self::RadarRange { .. } | Self::RadarRate { .. } => {
                Err(Error::ValueError("Expected an Optical observation".into()))
            }
        }
    }

    /// Returns true if this is a radar observation (range or range-rate).
    #[must_use]
    pub fn is_radar(&self) -> bool {
        matches!(self, Self::RadarRange { .. } | Self::RadarRate { .. })
    }

    /// Returns true if this optical measurement came from a stellar
    /// occultation rather than direct astrometry.  False for radar.
    #[must_use]
    pub fn is_occultation(&self) -> bool {
        matches!(
            self,
            Self::Optical {
                is_occultation: true,
                ..
            }
        )
    }

    /// Return a copy of this observation with `sigma_ra` and `sigma_dec`
    /// clamped to at least `floor` (in radians).  Radar observations are
    /// returned unchanged.  Used during the windowed-expansion passes of
    /// `fit_orbit` to prevent high-precision observations (e.g. occultations,
    /// Gaia transits) from dominating the gradient before the orbit has
    /// converged near the true solution.
    #[must_use]
    pub fn with_sigma_floor(&self, floor: f64) -> Self {
        match self {
            Self::Optical {
                observer,
                ra,
                dec,
                sigma_ra,
                sigma_dec,
                sigma_corr,
                time_sigma,
                is_occultation,
                band,
                mag,
            } => Self::Optical {
                observer: observer.clone(),
                ra: *ra,
                dec: *dec,
                sigma_ra: sigma_ra.max(floor),
                sigma_dec: sigma_dec.max(floor),
                sigma_corr: *sigma_corr,
                time_sigma: *time_sigma,
                is_occultation: *is_occultation,
                band: *band,
                mag: *mag,
            },
            Self::RadarRange { .. } | Self::RadarRate { .. } => self.clone(),
        }
    }

    /// Number of measurement components (2 for optical, 1 for radar).
    #[must_use]
    pub fn measurement_dim(&self) -> usize {
        match self {
            Self::Optical { .. } => 2,
            Self::RadarRange { .. } | Self::RadarRate { .. } => 1,
        }
    }

    /// Diagonal weight vector (1 / sigma^2 for each measurement component).
    ///
    /// This is the marginal `1 / sigma^2` per axis and does **not** include
    /// the RA/Dec correlation or the timing correction.  For numerically
    /// correct chi^2 accumulation, use [`Self::base_weight_matrix`] or
    /// [`Self::weight_matrix`] which carry the full (possibly non-diagonal)
    /// information matrix.  This method is retained only for consumers that
    /// need a quick per-axis scalar or the measurement dimension.
    #[must_use]
    pub fn weights(&self) -> DVector<f64> {
        match self {
            Self::Optical {
                sigma_ra,
                sigma_dec,
                ..
            } => DVector::from_column_slice(&[
                1.0 / (sigma_ra * sigma_ra),
                1.0 / (sigma_dec * sigma_dec),
            ]),
            Self::RadarRange { sigma_range, .. } => {
                DVector::from_column_slice(&[1.0 / (sigma_range * sigma_range)])
            }
            Self::RadarRate {
                sigma_range_rate, ..
            } => DVector::from_column_slice(&[1.0 / (sigma_range_rate * sigma_range_rate)]),
        }
    }

    /// Full base inverse-covariance matrix for this observation, **without**
    /// timing correction.
    ///
    /// For optical observations with non-zero `sigma_corr`, the base
    /// covariance is the tilted 2x2 matrix
    ///
    /// ```text
    /// Sigma = [[sigma_ra^2,            corr * sigma_ra * sigma_dec],
    ///          [corr * sigma_ra * sigma_dec, sigma_dec^2]]
    /// ```
    ///
    /// and the returned information matrix is `Sigma^{-1}`.  With
    /// `sigma_corr == 0` the result is `diag(1/sigma_ra^2, 1/sigma_dec^2)`.
    ///
    /// Radar observations return a 1x1 matrix (same as [`Self::weights`]).
    #[must_use]
    pub fn base_weight_matrix(&self) -> nalgebra::DMatrix<f64> {
        match self {
            Self::Optical {
                sigma_ra,
                sigma_dec,
                sigma_corr,
                ..
            } => optical_base_weight_matrix(*sigma_ra, *sigma_dec, *sigma_corr),
            Self::RadarRange { sigma_range, .. } => {
                nalgebra::DMatrix::from_column_slice(1, 1, &[1.0 / (sigma_range * sigma_range)])
            }
            Self::RadarRate {
                sigma_range_rate, ..
            } => nalgebra::DMatrix::from_column_slice(
                1,
                1,
                &[1.0 / (sigma_range_rate * sigma_range_rate)],
            ),
        }
    }

    /// Full weight matrix (inverse noise covariance) for this observation,
    /// including the timing correction.
    ///
    /// Starts from [`Self::base_weight_matrix`] (which carries the RA/Dec
    /// correlation) and, when `time_sigma > 0`, applies a Sherman-Morrison
    /// update that convolves the position Gaussian with the timing
    /// Gaussian:
    ///
    ///   `W_new` = W - (Wu)(Wu)^T / (1 + u^T W u)
    ///
    /// where `u = (motion_ra_rad_per_day * time_sigma_days,
    ///             motion_dec_rad_per_day * time_sigma_days)` is the timing
    /// displacement in radians.
    ///
    /// The Sherman-Morrison math is valid for any symmetric positive-definite
    /// `W`, so tilted base covariances propagate correctly through the
    /// timing stretch.  For radar observations, `motion_ra` and `motion_dec`
    /// are ignored and the base matrix is returned unchanged.
    #[must_use]
    pub fn weight_matrix(
        &self,
        motion_ra_rad_per_day: f64,
        motion_dec_rad_per_day: f64,
    ) -> nalgebra::DMatrix<f64> {
        let w = self.base_weight_matrix();
        match self {
            Self::Optical { time_sigma, .. } => {
                if *time_sigma == 0.0 {
                    return w;
                }
                // Timing displacement vector u = time_sigma_days * motion.
                let t_days = time_sigma / 86_400.0;
                let u_ra = t_days * motion_ra_rad_per_day;
                let u_dec = t_days * motion_dec_rad_per_day;
                apply_timing_to_weight_matrix(&w, u_ra, u_dec)
            }
            Self::RadarRange { .. } | Self::RadarRate { .. } => w,
        }
    }
}

/// Build the 2x2 information matrix (inverse covariance) for an optical
/// observation with RA/Dec correlation.
///
/// Returns `diag(1/sigma_ra^2, 1/sigma_dec^2)` when `sigma_corr == 0`.
/// Otherwise inverts the tilted 2x2 covariance analytically:
///
/// ```text
/// Sigma   = [[sr^2, c*sr*sd], [c*sr*sd, sd^2]]
/// det     = sr^2 * sd^2 * (1 - c^2)
/// Sigma^{-1} = 1/det * [[sd^2, -c*sr*sd], [-c*sr*sd, sr^2]]
///            = 1/(1-c^2) * [[1/sr^2, -c/(sr*sd)], [-c/(sr*sd), 1/sd^2]]
/// ```
fn optical_base_weight_matrix(
    sigma_ra: f64,
    sigma_dec: f64,
    sigma_corr: f64,
) -> nalgebra::DMatrix<f64> {
    if sigma_corr == 0.0 {
        return nalgebra::DMatrix::from_diagonal(&DVector::from_column_slice(&[
            1.0 / (sigma_ra * sigma_ra),
            1.0 / (sigma_dec * sigma_dec),
        ]));
    }
    let one_minus_c2 = 1.0 - sigma_corr * sigma_corr;
    // Caller is responsible for rejecting degenerate (|corr| = 1)
    // observations before they reach this point; guard against numerical
    // divide-by-zero with a small floor.
    let denom = one_minus_c2.max(1e-12);
    let w_rr = 1.0 / (sigma_ra * sigma_ra * denom);
    let w_dd = 1.0 / (sigma_dec * sigma_dec * denom);
    let w_rd = -sigma_corr / (sigma_ra * sigma_dec * denom);
    nalgebra::DMatrix::from_row_slice(2, 2, &[w_rr, w_rd, w_rd, w_dd])
}

/// Apply the Sherman-Morrison timing update to a 2x2 information matrix.
///
/// `(W_pos)^{-1} + sigma_t^2 * v v^T = (W_new)^{-1}`, equivalently
/// `W_new = W - (W u)(W u)^T / (1 + u^T W u)` with `u = sigma_t * v`.
/// The caller supplies `u = (u_ra, u_dec)` already scaled by `sigma_t`.
fn apply_timing_to_weight_matrix(
    w: &nalgebra::DMatrix<f64>,
    u_ra: f64,
    u_dec: f64,
) -> nalgebra::DMatrix<f64> {
    let w00 = w[(0, 0)];
    let w01 = w[(0, 1)];
    let w11 = w[(1, 1)];
    // Wu = W * u.
    let wu_ra = w00 * u_ra + w01 * u_dec;
    let wu_dec = w01 * u_ra + w11 * u_dec;
    // Denominator 1 + u^T W u is always >= 1 for positive-definite W.
    let denom = 1.0 + u_ra * wu_ra + u_dec * wu_dec;
    nalgebra::DMatrix::from_row_slice(
        2,
        2,
        &[
            w00 - wu_ra * wu_ra / denom,
            w01 - wu_ra * wu_dec / denom,
            w01 - wu_ra * wu_dec / denom,
            w11 - wu_dec * wu_dec / denom,
        ],
    )
}

impl AstrometricObservation {
    /// Compute the predicted measurement and residual (observed - computed).
    ///
    /// The input `obj_state` is the object at the observation epoch (before
    /// light-time correction). Light-time correction is applied internally.
    ///
    /// Returns `(residual, predicted)` where both are [`DVector`].
    ///
    /// # Errors
    /// Fails if two-body propagation for light-time correction fails.
    pub fn residual(
        &self,
        obj_state: State<Equatorial>,
    ) -> KeteResult<(DVector<f64>, DVector<f64>)> {
        let obs = self.observer()?;
        let spk = LOADED_SPK.try_read()?;
        // Convert object to heliocentric for two-body light-time correction.
        let obj_sun = spk.try_to_sun(obj_state)?;
        // Convert observer to heliocentric so the light-travel offset is
        // computed from the correct observer position.
        let obs_helio_sun = spk.try_to_sun(obs)?;
        // Apply light-time correction (heliocentric in, heliocentric out).
        let obj_lt_helio = kete_core::kepler::light_time_correct(&obj_sun, &obs_helio_sun.pos)?;
        // Apply differential gravitational light deflection (solar bending relative
        // to background stars at infinity).  The common-mode bending cancels in
        // plate-solved frames; only this differential term survives.
        let deflected_pos = differential_light_deflect(&obs_helio_sun.pos, obj_lt_helio.pos);
        let obj_lt_deflected = State {
            pos: deflected_pos,
            ..obj_lt_helio
        };
        // Convert back to SSB so residual_predicted_from_corrected sees consistent positions.
        let obj_lt_ssb = spk.try_to_ssb(obj_lt_deflected)?;
        self.residual_predicted_from_corrected(&obj_lt_ssb)
    }

    /// Compute residual from an already light-time-corrected object state.
    ///
    /// This avoids a redundant two-body propagation when the caller has
    /// already applied `light_time_correct`.
    ///
    /// # Errors
    /// Fails if PCK / SPK lookups for radar station states fail.
    pub fn residual_from_corrected(
        &self,
        obj_lt: &State<Equatorial, SSB>,
    ) -> KeteResult<DVector<f64>> {
        Ok(self.residual_predicted_from_corrected(obj_lt)?.0)
    }

    /// Core residual computation from a light-time-corrected state.
    ///
    /// Returns `(residual, predicted)`.
    ///
    /// # Errors
    /// Fails if PCK / SPK lookups for radar station states fail.
    fn residual_predicted_from_corrected(
        &self,
        obj_lt: &State<Equatorial, SSB>,
    ) -> KeteResult<(DVector<f64>, DVector<f64>)> {
        match self {
            Self::Optical {
                observer, ra, dec, ..
            } => {
                let (ra_pred, dec_pred) = (obj_lt.pos - observer.pos).to_ra_dec();
                // Wrap RA residual to [-pi, pi]
                let mut d_ra = ra - ra_pred;
                if d_ra > std::f64::consts::PI {
                    d_ra -= 2.0 * std::f64::consts::PI;
                } else if d_ra < -std::f64::consts::PI {
                    d_ra += 2.0 * std::f64::consts::PI;
                }
                Ok((
                    DVector::from_column_slice(&[d_ra, dec - dec_pred]),
                    DVector::from_column_slice(&[ra_pred, dec_pred]),
                ))
            }
            Self::RadarRange {
                xmit_lat_rad,
                xmit_lon_rad,
                xmit_height_km,
                rcvr_lat_rad,
                rcvr_lon_rad,
                rcvr_height_km,
                epoch,
                range,
                ..
            } => {
                // Round-trip model: (incoming + outgoing) / 2 + Shapiro delay.
                // Compute rcvr at t_rx and iteratively refine xmit at t_tx
                // via predicted geometry.
                let rcvr_state =
                    station_state_at(*rcvr_lat_rad, *rcvr_lon_rad, *rcvr_height_km, *epoch)?;
                let incoming = (obj_lt.pos - rcvr_state.pos).norm();
                let mut outgoing = incoming;
                let mut xmit_state = rcvr_state.clone();
                for _ in 0..3 {
                    let rtt_days = (incoming + outgoing) * C_AU_PER_DAY_INV;
                    let t_tx = *epoch - rtt_days;
                    xmit_state =
                        station_state_at(*xmit_lat_rad, *xmit_lon_rad, *xmit_height_km, t_tx)?;
                    outgoing = (obj_lt.pos - xmit_state.pos).norm();
                }
                // Shapiro (gravitational) delay along each leg.  Sun position
                // at t_rx is used as a single reference; Sun moves <5e-3 AU
                // over a round-trip light time so the heliocentric distances
                // are insensitive to which epoch is sampled.
                let spk = LOADED_SPK.try_read()?;
                let sun_pos = spk.try_get_state::<Equatorial>(10, *epoch)?.pos;
                let r_obj = (obj_lt.pos - sun_pos).norm();
                let r_rcvr = (rcvr_state.pos - sun_pos).norm();
                let r_xmit = (xmit_state.pos - sun_pos).norm();
                let shapiro_in = shapiro_range_au(r_obj, r_rcvr, incoming);
                let shapiro_out = shapiro_range_au(r_obj, r_xmit, outgoing);
                let pred = f64::midpoint(incoming + shapiro_in, outgoing + shapiro_out);
                Ok((
                    DVector::from_column_slice(&[range - pred]),
                    DVector::from_column_slice(&[pred]),
                ))
            }
            Self::RadarRate {
                xmit_lat_rad,
                xmit_lon_rad,
                xmit_height_km,
                rcvr_lat_rad,
                rcvr_lon_rad,
                rcvr_height_km,
                epoch,
                range_rate,
                ..
            } => {
                // Full relativistic two-way Doppler:
                //   f_obs / f_emit = sqrt((1 - beta_rx)(1 - beta_tx) /
                //                        ((1 + beta_rx)(1 + beta_tx)))
                // where beta = v_radial / c for each leg (dimensionless).
                // Predicted range-rate = -c/2 * (f_obs/f_emit - 1) in AU/day.
                //
                // The transmitter epoch t_tx is iteratively refined from the
                // current orbit's predicted geometry; both stations are then
                // produced by PCK lookup at their respective epochs.
                //
                // The Jacobian retains the classical approximation because the
                // relativistic correction is O(v^2/c^2) ~ 1e-8 and its gradient
                // contribution is negligible.
                let rcvr_state =
                    station_state_at(*rcvr_lat_rad, *rcvr_lon_rad, *rcvr_height_km, *epoch)?;
                let d_pos_rx = obj_lt.pos - rcvr_state.pos;
                let range_rx = d_pos_rx.norm();

                // Iterate to converge on t_tx and the corresponding xmit state.
                let mut range_tx = range_rx;
                let mut xmit_state = rcvr_state.clone();
                for _ in 0..3 {
                    let rtt_days = (range_rx + range_tx) * C_AU_PER_DAY_INV;
                    let t_tx = *epoch - rtt_days;
                    xmit_state =
                        station_state_at(*xmit_lat_rad, *xmit_lon_rad, *xmit_height_km, t_tx)?;
                    range_tx = (obj_lt.pos - xmit_state.pos).norm();
                }
                let d_pos_tx = obj_lt.pos - xmit_state.pos;

                // Radial velocities (AU/day, positive = receding).
                let v_rel_rx = obj_lt.vel - rcvr_state.vel;
                let v_rel_tx = obj_lt.vel - xmit_state.vel;
                let rr_rx = d_pos_rx.dot(&v_rel_rx) / range_rx;
                let rr_tx = d_pos_tx.dot(&v_rel_tx) / range_tx;

                // Convert to dimensionless beta = v_radial / c.
                // C_AU_PER_DAY_INV = 1/c in units of day/AU; multiply to get AU/day -> dimensionless.
                let c_au_per_day = 1.0 / C_AU_PER_DAY_INV;
                let beta_rx = rr_rx / c_au_per_day;
                let beta_tx = rr_tx / c_au_per_day;

                // Relativistic two-way Doppler factor.
                let doppler_factor = ((1.0 - beta_rx) * (1.0 - beta_tx)
                    / ((1.0 + beta_rx) * (1.0 + beta_tx)))
                    .sqrt();

                // Predicted range-rate: derived from frequency shift via
                //   df/f = doppler_factor - 1,  range_rate = -c * df / (2f)
                let pred = -c_au_per_day * 0.5 * (doppler_factor - 1.0);
                Ok((
                    DVector::from_column_slice(&[range_rate - pred]),
                    DVector::from_column_slice(&[pred]),
                ))
            }
        }
    }
}

/// Optical partials: d(RA,Dec)/d(pos) as a 2x3 matrix.
///
/// Velocity partials are zero (RA/Dec do not depend on velocity at the
/// instant of observation, neglecting light-time rate corrections).
fn optical_partials_pos(
    obj: &State<Equatorial, SSB>,
    obs: &State<Equatorial, SSB>,
) -> Matrix2x3<f64> {
    let d = obj.pos - obs.pos;
    let dx = d[0];
    let dy = d[1];
    let dz = d[2];
    let rho2 = d.norm_squared();
    let xy2 = dx * dx + dy * dy;

    // Guard against the pole singularity (dec near +/-90 deg).
    // When xy2 -> 0 the RA partial is undefined and the Dec partial
    // diverges.  Clamp to a small floor so the Jacobian stays finite;
    // the residual itself is still well-defined, and the solver will
    // not be driven by a single near-pole observation.
    let xy2_safe = xy2.max(1e-30);
    let xy = xy2_safe.sqrt();

    // dRA/d(pos)
    let dra_dx = -dy / xy2_safe;
    let dra_dy = dx / xy2_safe;

    // dDec/d(pos)
    let ddec_dx = -dx * dz / (rho2 * xy);
    let ddec_dy = -dy * dz / (rho2 * xy);
    let ddec_dz = xy / rho2;

    Matrix2x3::new(dra_dx, dra_dy, 0.0, ddec_dx, ddec_dy, ddec_dz)
}

/// Radar range partials: d(range)/d(pos) as a 3x1 column vector.
///
/// Round-trip average `(incoming + outgoing) / 2`.  With xmit and rcvr
/// positions both held fixed (pre-computed by ingestion), there is no
/// indirect correction term: `d(outgoing)/d(r_obj) = unit_tx` only.
fn range_partials_pos(
    obj: &State<Equatorial, SSB>,
    xmit: &State<Equatorial, SSB>,
    rcvr: &State<Equatorial, SSB>,
) -> Matrix3x1<f64> {
    let d_rx = obj.pos - rcvr.pos;
    let d_tx = obj.pos - xmit.pos;
    let unit_rx = d_rx / d_rx.norm();
    let unit_tx = d_tx / d_tx.norm();
    let avg = (unit_rx + unit_tx) * 0.5;
    Matrix3x1::new(avg[0], avg[1], avg[2])
}

/// Radar range-rate partials: `d(range_rate)/d(pos,vel)` as a 1x6 row vector.
///
/// Classical two-way Doppler partial; the relativistic correction in the
/// residual is O(v^2/c^2), so its gradient contribution is negligible and
/// is dropped here.  With xmit and rcvr held fixed there are no indirect
/// terms: only the direct geometric derivatives remain.
fn range_rate_partials(
    obj: &State<Equatorial, SSB>,
    xmit: &State<Equatorial, SSB>,
    rcvr: &State<Equatorial, SSB>,
) -> RowVector6<f64> {
    let d_pos_rx = obj.pos - rcvr.pos;
    let d_pos_tx = obj.pos - xmit.pos;
    let range_rx = d_pos_rx.norm();
    let range_tx = d_pos_tx.norm();
    let d_hat_rx = d_pos_rx / range_rx;
    let d_hat_tx = d_pos_tx / range_tx;

    let v_rel_rx = obj.vel - rcvr.vel;
    let v_rel_tx = obj.vel - xmit.vel;
    let rr_rx = d_pos_rx.dot(&v_rel_rx) / range_rx;
    let rr_tx = d_pos_tx.dot(&v_rel_tx) / range_tx;

    // d(rr_rx)/d(r) = (v_rel_rx - d_hat_rx * rr_rx) / range_rx
    let dr_rx = (v_rel_rx - d_hat_rx * rr_rx) * (1.0 / range_rx);
    // d(rr_tx)/d(r) = (v_rel_tx - d_hat_tx * rr_tx) / range_tx
    let dr_tx = (v_rel_tx - d_hat_tx * rr_tx) * (1.0 / range_tx);
    let dr = (dr_rx + dr_tx) * 0.5;

    // d(rr)/d(v) = (d_hat_rx + d_hat_tx) / 2  (no indirect term)
    let dv = (d_hat_rx + d_hat_tx) * 0.5;

    RowVector6::new(dr[0], dr[1], dr[2], dv[0], dv[1], dv[2])
}

impl AstrometricObservation {
    /// Compute the local geometric partial derivatives (`H_local`).
    ///
    /// Returns an m x 6 matrix where m is the measurement dimension
    /// (2 for optical, 1 for radar). Columns correspond to
    /// `[dx, dy, dz, dvx, dvy, dvz]` of the object state.
    ///
    /// The input `obj_state` should already be light-time-corrected.
    ///
    /// For radar observations the xmit and rcvr station states are
    /// looked up via PCK at `t_rx` (a fixed approximation -- the
    /// indirect-correction term from the orbit-dependence of `t_tx` is
    /// `O(v_surface/c)` and dropped).
    ///
    /// # Errors
    /// Fails if PCK / SPK lookups for radar station states fail.
    pub fn partials(
        &self,
        obj_state: &State<Equatorial, SSB>,
    ) -> KeteResult<nalgebra::DMatrix<f64>> {
        match self {
            Self::Optical { observer, .. } => {
                let h = optical_partials_pos(obj_state, observer);
                let mut out = nalgebra::DMatrix::zeros(2, 6);
                out.view_mut((0, 0), (2, 3)).copy_from(&h);
                Ok(out)
            }
            Self::RadarRange {
                xmit_lat_rad,
                xmit_lon_rad,
                xmit_height_km,
                rcvr_lat_rad,
                rcvr_lon_rad,
                rcvr_height_km,
                epoch,
                ..
            } => {
                let rcvr_state =
                    station_state_at(*rcvr_lat_rad, *rcvr_lon_rad, *rcvr_height_km, *epoch)?;
                let xmit_state =
                    station_state_at(*xmit_lat_rad, *xmit_lon_rad, *xmit_height_km, *epoch)?;
                let h = range_partials_pos(obj_state, &xmit_state, &rcvr_state);
                let mut out = nalgebra::DMatrix::zeros(1, 6);
                out[(0, 0)] = h[0];
                out[(0, 1)] = h[1];
                out[(0, 2)] = h[2];
                Ok(out)
            }
            Self::RadarRate {
                xmit_lat_rad,
                xmit_lon_rad,
                xmit_height_km,
                rcvr_lat_rad,
                rcvr_lon_rad,
                rcvr_height_km,
                epoch,
                ..
            } => {
                let rcvr_state =
                    station_state_at(*rcvr_lat_rad, *rcvr_lon_rad, *rcvr_height_km, *epoch)?;
                let xmit_state =
                    station_state_at(*xmit_lat_rad, *xmit_lon_rad, *xmit_height_km, *epoch)?;
                let h = range_rate_partials(obj_state, &xmit_state, &rcvr_state);
                let mut out = nalgebra::DMatrix::zeros(1, 6);
                for j in 0..6 {
                    out[(0, j)] = h[j];
                }
                Ok(out)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kete_core::constants::C_AU_PER_DAY_INV;
    use kete_core::desigs::Desig;
    use kete_core::frames::{Equatorial, SunCenter};
    use kete_core::kepler::propagate_two_body;
    use kete_core::prelude::State;

    /// Helper: build a simple state at the given position/velocity.
    fn make_state(pos: [f64; 3], vel: [f64; 3], jd: f64) -> State<Equatorial, SSB> {
        State {
            desig: Desig::Empty,
            epoch: jd.into(),
            pos: pos.into(),
            vel: vel.into(),
            center: SSB,
        }
    }

    /// Finite-difference helper for an arbitrary scalar-or-vector predictor.
    /// Perturb component `idx` of the object state by `eps`, recompute the
    /// predicted value, return the numerical partial derivative.
    fn fd_partial<F, T>(predictor: F, obj: &State<Equatorial, SSB>, idx: usize, eps: f64) -> T
    where
        F: Fn(&State<Equatorial, SSB>) -> T,
        T: std::ops::Sub<T, Output = T> + std::ops::Mul<f64, Output = T>,
    {
        let mut pos_p = [obj.pos[0], obj.pos[1], obj.pos[2]];
        let mut vel_p = [obj.vel[0], obj.vel[1], obj.vel[2]];
        let mut pos_m = pos_p;
        let mut vel_m = [obj.vel[0], obj.vel[1], obj.vel[2]];

        if idx < 3 {
            pos_p[idx] += eps;
            pos_m[idx] -= eps;
        } else {
            vel_p[idx - 3] += eps;
            vel_m[idx - 3] -= eps;
        }

        let obj_p = make_state(pos_p, vel_p, obj.epoch.jd);
        let obj_m = make_state(pos_m, vel_m, obj.epoch.jd);

        (predictor(&obj_p) - predictor(&obj_m)) * (1.0 / (2.0 * eps))
    }

    /// Direct optical prediction without light-time (for FD tests).
    fn predict_optical_for_fd(
        observer: &State<Equatorial, SSB>,
        obj: &State<Equatorial, SSB>,
    ) -> DVector<f64> {
        let (ra, dec) = (obj.pos - observer.pos).to_ra_dec();
        DVector::from_vec(vec![ra, dec])
    }

    /// Direct round-trip range prediction with explicit xmit/rcvr states
    /// (for FD tests of `range_partials_pos`).
    fn predict_range_for_fd(
        xmit: &State<Equatorial, SSB>,
        rcvr: &State<Equatorial, SSB>,
        obj: &State<Equatorial, SSB>,
    ) -> f64 {
        let incoming = (obj.pos - rcvr.pos).norm();
        let outgoing = (obj.pos - xmit.pos).norm();
        f64::midpoint(incoming, outgoing)
    }

    /// Direct two-way Doppler prediction (classical) with explicit xmit/rcvr
    /// states (for FD tests of `range_rate_partials`).
    fn predict_rate_for_fd(
        xmit: &State<Equatorial, SSB>,
        rcvr: &State<Equatorial, SSB>,
        obj: &State<Equatorial, SSB>,
    ) -> f64 {
        let d_pos_rx = obj.pos - rcvr.pos;
        let d_pos_tx = obj.pos - xmit.pos;
        let v_rel_rx = obj.vel - rcvr.vel;
        let v_rel_tx = obj.vel - xmit.vel;
        let rr_rx = d_pos_rx.dot(&v_rel_rx) / d_pos_rx.norm();
        let rr_tx = d_pos_tx.dot(&v_rel_tx) / d_pos_tx.norm();
        f64::midpoint(rr_rx, rr_tx)
    }

    #[test]
    fn test_optical_partials_vs_fd() {
        // Object ~1 AU from observer, not on any axis to exercise all terms
        let observer = make_state([1.0, 0.0, 0.0], [0.0, 0.01, 0.0], 2460000.5);
        let obj = make_state([1.5, 0.8, 0.3], [0.001, -0.002, 0.0005], 2460000.5);

        let obs = AstrometricObservation::Optical {
            observer: observer.clone(),
            ra: 0.0,
            dec: 0.0,
            sigma_ra: 1e-6,
            sigma_dec: 1e-6,
            sigma_corr: 0.0,
            time_sigma: 0.0,
            is_occultation: false,
            band: Band::Unknown([0; 8]),
            mag: f64::NAN,
        };

        let h = obs.partials(&obj).unwrap();
        let observer = match &obs {
            AstrometricObservation::Optical { observer, .. } => observer.clone(),
            AstrometricObservation::RadarRange { .. }
            | AstrometricObservation::RadarRate { .. } => {
                unreachable!()
            }
        };
        let eps = 1e-8;

        for idx in 0..6 {
            let fd = fd_partial(|o| predict_optical_for_fd(&observer, o), &obj, idx, eps);
            for row in 0..2 {
                let analytic = h[(row, idx)];
                let numeric = fd[row];
                let abs_err = (analytic - numeric).abs();
                let scale = analytic.abs().max(numeric.abs()).max(1e-15);
                assert!(
                    abs_err < 1e-7 || abs_err / scale < 1e-5,
                    "Optical partial ({row}, {idx}) mismatch: analytic={analytic}, fd={numeric}",
                );
            }
        }
    }

    #[test]
    fn test_range_partials_vs_fd() {
        // Test the analytic Jacobian of `range_partials_pos` directly against
        // FD using arbitrary state vectors -- no PCK lookup involved.
        let rcvr = make_state([1.0, 0.0, 0.0], [0.0, 0.01, 0.0], 2460000.5);
        let xmit = make_state([0.99, 0.001, 0.0], [0.0, 0.0099, 0.0], 2460000.5);
        let obj = make_state([2.3, 0.5, -0.2], [0.002, -0.001, 0.001], 2460000.5);

        let h = range_partials_pos(&obj, &xmit, &rcvr);
        let eps = 1e-8;

        for idx in 0..6 {
            let fd = fd_partial(|o| predict_range_for_fd(&xmit, &rcvr, o), &obj, idx, eps);
            // Range partials only depend on position (idx 0..3); velocity
            // partials are zero by construction.
            let analytic = if idx < 3 { h[idx] } else { 0.0 };
            let abs_err = (analytic - fd).abs();
            let scale = analytic.abs().max(fd.abs()).max(1e-15);
            assert!(
                abs_err < 1e-7 || abs_err / scale < 1e-5,
                "Range partial ({idx}) mismatch: analytic={analytic}, fd={fd}",
            );
        }
    }

    #[test]
    fn test_range_rate_partials_vs_fd() {
        // Test the analytic Jacobian of `range_rate_partials` directly against
        // FD using arbitrary state vectors -- no PCK lookup involved.
        let rcvr = make_state([1.0, 0.0, 0.0], [0.0, 0.01, 0.0], 2460000.5);
        let xmit = make_state([0.99, 0.001, 0.0], [0.0, 0.0099, 0.0], 2460000.5);
        let obj = make_state([2.3, 0.5, -0.2], [0.002, -0.001, 0.001], 2460000.5);

        let h = range_rate_partials(&obj, &xmit, &rcvr);
        let eps = 1e-8;

        for idx in 0..6 {
            let fd = fd_partial(|o| predict_rate_for_fd(&xmit, &rcvr, o), &obj, idx, eps);
            let analytic = h[idx];
            let abs_err = (analytic - fd).abs();
            let scale = analytic.abs().max(fd.abs()).max(1e-15);
            assert!(
                abs_err < 1e-7 || abs_err / scale < 1e-5,
                "Range-rate partial ({idx}) mismatch: analytic={analytic}, fd={fd}",
            );
        }
    }

    #[test]
    fn test_light_time_correction() {
        // Object at 2 AU from observer. Light time ~ 2 * C_AU_PER_DAY_INV ~ 0.01155 days
        let observer = make_state([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 2460000.5);
        let obj = State::<Equatorial, SunCenter> {
            desig: Desig::Empty,
            epoch: 2460000.5.into(),
            pos: [2.0, 0.0, 0.0].into(),
            vel: [0.0, 0.01, 0.0].into(),
            center: SunCenter,
        };

        let tau_lt = (obj.pos - observer.pos).norm() * C_AU_PER_DAY_INV;
        let corrected = propagate_two_body(&obj, obj.epoch - tau_lt).unwrap();

        // Corrected epoch should be earlier
        let tau = 2.0 * C_AU_PER_DAY_INV;
        assert!((corrected.epoch.jd - (2460000.5 - tau)).abs() < 1e-12);

        // Position should be slightly different due to back-propagation
        assert!((corrected.pos[0] - obj.pos[0]).abs() < 1e-4);
        // dy should shift: the object moves in y at 0.01 AU/day, backed by ~0.01 day
        assert!(corrected.pos[1] < obj.pos[1]);
    }

    #[test]
    fn test_residual_optical() {
        // Observer at ~1 AU (Earth-like), object at ~2 AU along +x.
        let observer = make_state([1.0, 0.0, 0.0], [0.0, 0.017, 0.0], 2460000.5);
        let obj = make_state([2.0, 0.0, 0.0], [0.0, 0.012, 0.0], 2460000.5);

        // True RA/Dec for object along +x from observer is RA~0, Dec~0.
        let d = obj.pos - observer.pos;
        let (true_ra, true_dec) = d.to_ra_dec();

        let obs = AstrometricObservation::Optical {
            observer,
            ra: true_ra + 0.01,
            dec: true_dec + 0.005,
            sigma_ra: 1e-6,
            sigma_dec: 1e-6,
            sigma_corr: 0.0,
            time_sigma: 0.0,
            is_occultation: false,
            band: Band::Unknown([0; 8]),
            mag: f64::NAN,
        };

        let (resid, _pred) = obs.residual(obj.into()).unwrap();
        // Residual should be close to the injected offset.
        // (Predicted RA may wrap by 2*pi; the residual handles wrapping.)
        assert!((resid[0] - 0.01).abs() < 0.01);
        assert!((resid[1] - 0.005).abs() < 0.01);
    }

    #[test]
    fn test_weights() {
        let obs = AstrometricObservation::Optical {
            observer: make_state([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 2460000.5),
            ra: 0.0,
            dec: 0.0,
            sigma_ra: 0.5,
            sigma_dec: 0.25,
            sigma_corr: 0.0,
            time_sigma: 0.0,
            is_occultation: false,
            band: Band::Unknown([0; 8]),
            mag: f64::NAN,
        };
        let w = obs.weights();
        // 1/0.5^2 = 4
        assert!((w[0] - 4.0).abs() < 1e-12);
        // 1/0.25^2 = 16
        assert!((w[1] - 16.0).abs() < 1e-12);
    }

    #[test]
    fn test_base_weight_matrix_zero_correlation_equals_diagonal() {
        let obs = AstrometricObservation::Optical {
            observer: make_state([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 2460000.5),
            ra: 0.0,
            dec: 0.0,
            sigma_ra: 0.5,
            sigma_dec: 0.25,
            sigma_corr: 0.0,
            time_sigma: 0.0,
            is_occultation: false,
            band: Band::Unknown([0; 8]),
            mag: f64::NAN,
        };
        let w = obs.base_weight_matrix();
        assert_eq!(w.nrows(), 2);
        assert!((w[(0, 0)] - 4.0).abs() < 1e-12);
        assert!((w[(1, 1)] - 16.0).abs() < 1e-12);
        assert!(w[(0, 1)].abs() < 1e-12);
        assert!(w[(1, 0)].abs() < 1e-12);
    }

    #[test]
    fn test_base_weight_matrix_with_correlation_inverts_tilted_covariance() {
        // Correlation coefficient 0.5, sigmas 1.0 and 2.0.
        let sigma_ra = 1.0;
        let sigma_dec = 2.0;
        let corr = 0.5;
        let obs = AstrometricObservation::Optical {
            observer: make_state([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 2460000.5),
            ra: 0.0,
            dec: 0.0,
            sigma_ra,
            sigma_dec,
            sigma_corr: corr,
            time_sigma: 0.0,
            is_occultation: false,
            band: Band::Unknown([0; 8]),
            mag: f64::NAN,
        };
        let w = obs.base_weight_matrix();

        // Build Sigma directly and invert it numerically; W should match.
        let sigma = nalgebra::Matrix2::new(
            sigma_ra * sigma_ra,
            corr * sigma_ra * sigma_dec,
            corr * sigma_ra * sigma_dec,
            sigma_dec * sigma_dec,
        );
        let sigma_inv = sigma.try_inverse().unwrap();
        for i in 0..2 {
            for j in 0..2 {
                let abs_err = (w[(i, j)] - sigma_inv[(i, j)]).abs();
                assert!(
                    abs_err < 1e-12,
                    "W[{i},{j}] = {} but Sigma^-1 = {}",
                    w[(i, j)],
                    sigma_inv[(i, j)]
                );
            }
        }
    }
}
