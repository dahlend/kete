//! # Modified Equinoctial Orbital Elements
//!
//! Conversion to and from modified equinoctial orbital elements and [`State`].
//
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

use super::gm_sqrt_for_center;
use crate::errors::Error;
use crate::frames::{CenterBody, DynCenter, Ecliptic, InertialFrame};
use crate::prelude::{Desig, KeteResult, State};
use crate::time::{TDB, Time};

use nalgebra::{Matrix6, Vector3, Vector6};
use std::f64::consts::TAU;

/// Vector Orbital Elements.
///
/// Modified equinoctial orbital elements, in the formulation of Walker, Ireland and
/// Owens (1985).
///
/// Six floats, no constraints between them, describing one conic about a gravitating
/// body. *Modified* is load bearing: this set is parameterized by the semi-latus rectum
/// and the true longitude rather than the semi major axis and the mean longitude, and
/// that is what makes it regular at unit eccentricity, where the classical equinoctial
/// set diverges along with the semi major axis.
///
/// | field | symbol | meaning |
/// |---|---|---|
/// | `semi_latus` | `p` | semi-latus rectum in AU, positive on every conic |
/// | `ecc_f` | `f` | `e cos(omega + Omega)` |
/// | `ecc_g` | `g` | `e sin(omega + Omega)` |
/// | `pole_h` | `h` | `tan(i/2) cos(Omega)` |
/// | `pole_k` | `k` | `tan(i/2) sin(Omega)` |
/// | `true_lon` | `L_0` | true longitude at the epoch, `Omega + omega + nu`, radians |
///
/// The classical singularities are absent. Zero eccentricity and zero inclination are
/// ordinary points: `(f, g)` carries the eccentricity as a vector and so needs no
/// perihelion to measure an argument from, and `(h, k)` carries the orbit pole and so
/// needs no ascending node. Every conversion below is closed form, with no Kepler solve
/// and no branch on eccentricity.
///
/// Two properties matter downstream. Five of the six floats are constants of two-body
/// motion, so two-body evolution moves only `true_lon`. And the six carry no constraint,
/// so a covariance over them is directly `6 x 6` and full rank, with no chart to build
/// and no rank deficiency to handle.
///
/// # Frame
/// The storage frame is the Ecliptic and is not a type parameter. `(h, k)` is a
/// stereographic projection of the orbit pole from the storage frame's `-z`, so changing
/// frames is not a rotation of the stored floats: it requires decoding to a state,
/// rotating, and re-encoding, and it moves the singularity below. Fixing the frame keeps
/// that operation from happening implicitly. Rotating a [`State`] between frames stays
/// linear and is unaffected.
///
/// # The seam
/// `(h, k)` grows as `tan(i/2)` and is undefined at the ecliptic retrograde pole,
/// `i = 180` degrees exactly, which [`Self::from_state`] rejects. Near it the *point*
/// representation remains usable much further than a *covariance* over it does:
///
/// - The condition number of [`Self::state_jacobian`] grows as `csc^2(psi/2)` in `psi`, the
///   angular distance from the seam, but that growth is unit bookkeeping. After row and
///   column equilibration the number is flat, because a stereographic projection is
///   conformal and so introduces no anisotropy. See
///   `equinoctial_seam_conditioning_profile`.
/// - A covariance picks up the square of the inverse Jacobian on top of the dynamic range a
///   fitted orbit covariance already carries, so it loses range while the point
///   representation is still accurate, and past that point its small eigenvalues are not
///   determined at all. Where that happens, and how much margin
///   [`equinoctial_covariance_domain`](crate::state::equinoctial_covariance_domain) leaves
///   against it, is what `equinoctial_seam_covariance_range` reports.
///
/// A scalar phase coordinate on a bound orbit always has a seam and any two-float pole
/// encoding always excludes one pole direction; neither is avoidable. Placing the seam at
/// the ecliptic retrograde pole puts it where the population is thinnest: the fraction of
/// an isotropic pole distribution within `psi` of it is `(1 - cos psi) / 2`, and real
/// populations are more concentrated away from retrograde than isotropic.
///
/// Units are AU, days, and radians.
#[derive(Debug, Clone)]
pub struct EquinoctialElements {
    /// Designation of the object
    pub desig: Desig,

    /// Epoch of fit
    pub epoch: Time<TDB>,

    /// Semi-latus rectum in AU, `p`. Finite and positive on every conic.
    pub semi_latus: f64,

    /// Eccentricity component along the equinoctial `fhat`, `f = e cos(omega + Omega)`.
    pub ecc_f: f64,

    /// Eccentricity component along the equinoctial `ghat`, `g = e sin(omega + Omega)`.
    pub ecc_g: f64,

    /// Pole component `h = tan(i/2) cos(Omega)`.
    pub pole_h: f64,

    /// Pole component `k = tan(i/2) sin(Omega)`.
    pub pole_k: f64,

    /// True longitude at the epoch in radians, `L_0 = Omega + omega + nu`.
    ///
    /// This coordinate wraps. [`Self::from_state`] returns it in `(-pi, pi]`;
    /// [`Self::displaced_by`] does not reduce it. Differences must go through
    /// [`Self::offset_to`], which reduces to the shortest signed angle.
    pub true_lon: f64,

    /// NAIF ID of the central body (default: 10 for the Sun)
    pub center_id: i32,

    /// Square root of the gravitational parameter of the central body.
    /// Units: AU^(3/2) / Day
    pub gm_sqrt: f64,
}

impl EquinoctialElements {
    /// Create equinoctial elements from a state.
    ///
    /// # Errors
    /// Fails if the state's center has no known mass, if the motion is rectilinear, if
    /// the orbit is exactly retrograde, or if the state contains non-finite values.
    pub fn from_state<C: CenterBody>(state: &State<Ecliptic, C>) -> KeteResult<Self>
    where
        DynCenter: From<C>,
    {
        let gm_sqrt = gm_sqrt_for_center(state.center_id())?;
        Self::from_pos_vel(
            state.desig.clone(),
            state.epoch,
            &state.pos.into(),
            &state.vel.into(),
            state.center_id(),
            gm_sqrt,
        )
    }

    /// Convert equinoctial elements to a [`State`] if possible.
    ///
    /// # Errors
    /// Fails when the orbit equation `1 + f cos L_0 + g sin L_0` is not positive, which
    /// places the true longitude outside the asymptotes of an open orbit. This cannot occur
    /// below unit eccentricity.
    pub fn try_to_state(&self) -> KeteResult<State<Ecliptic>> {
        let [pos, vel] = self.to_pos_vel()?;
        Ok(State::new(
            self.desig.clone(),
            self.epoch,
            pos,
            vel,
            self.center_id,
        ))
    }

    /// Jacobian of the epoch position and velocity with respect to the six stored floats,
    /// evaluated at the current point.
    ///
    /// Rows are the three position components in AU followed by the three velocity
    /// components in AU/day; columns are `p`, `f`, `g`, `h`, `k` and `L_0`. Closed form,
    /// with no Kepler solve and no branch on eccentricity.
    ///
    /// **The frame parameter selects which frame the rows are expressed in**, and naming
    /// it is required at every call site. The elements are stored in the Ecliptic while
    /// much of the propagation stack runs in the Equatorial, and composing a Jacobian
    /// from one frame with a state transition matrix from another is silent and gives a
    /// plausible wrong answer. This is the only place in the crate that rotates between
    /// the two, so a consumer asks for the frame it needs and never writes the rotation
    /// itself.
    ///
    /// The phase column is the Keplerian flow scaled by the true-longitude rate, which is
    /// an exact and undifferenced identity:
    ///
    /// ```text
    /// d(pos)/dL * (dL/dt) = vel        d(vel)/dL * (dL/dt) = -GM pos / |pos|^3
    /// dL/dt = |r x v| / r^2 = sqrt(GM p) / r^2
    /// ```
    ///
    /// Shifting the origin to another body leaves this unchanged, because that offset is
    /// a function of time alone and not of the state.
    ///
    /// # Errors
    /// Fails when the true longitude is outside the physical domain, matching
    /// [`Self::try_to_state`].
    pub fn state_jacobian<F: InertialFrame>(&self) -> KeteResult<Matrix6<f64>> {
        let orbit_eq = self.orbit_equation();
        if orbit_eq <= 0.0 {
            return Err(Self::domain_error());
        }
        let (sin_lon, cos_lon) = self.true_lon.sin_cos();
        let (pole_h, pole_k) = (self.pole_h, self.pole_k);
        let scale_sq = 1.0 + pole_h * pole_h + pole_k * pole_k;

        // Unnormalized basis directions, and their derivatives with respect to the pole
        // components through the quotient rule.
        let num_f = Vector3::new(
            1.0 + pole_h * pole_h - pole_k * pole_k,
            2.0 * pole_h * pole_k,
            -2.0 * pole_k,
        );
        let num_g = Vector3::new(
            2.0 * pole_h * pole_k,
            1.0 - pole_h * pole_h + pole_k * pole_k,
            2.0 * pole_h,
        );
        let f_hat = num_f / scale_sq;
        let g_hat = num_g / scale_sq;
        let inv_sq = 1.0 / (scale_sq * scale_sq);
        let df_dh = Vector3::new(2.0 * pole_h, 2.0 * pole_k, 0.0) / scale_sq
            - num_f * (2.0 * pole_h * inv_sq);
        let df_dk = Vector3::new(-2.0 * pole_k, 2.0 * pole_h, -2.0) / scale_sq
            - num_f * (2.0 * pole_k * inv_sq);
        let dg_dh = Vector3::new(2.0 * pole_k, -2.0 * pole_h, 2.0) / scale_sq
            - num_g * (2.0 * pole_h * inv_sq);
        let dg_dk = Vector3::new(2.0 * pole_h, 2.0 * pole_k, 0.0) / scale_sq
            - num_g * (2.0 * pole_k * inv_sq);

        let radius = self.semi_latus / orbit_eq;
        let radial = f_hat * cos_lon + g_hat * sin_lon;
        let speed = self.gm_sqrt / self.semi_latus.sqrt();
        let vel_f = -(self.ecc_g + sin_lon);
        let vel_g = self.ecc_f + cos_lon;
        let vel = (f_hat * vel_f + g_hat * vel_g) * speed;

        let columns = [
            (radial / orbit_eq, -vel / (2.0 * self.semi_latus)),
            (radial * (-radius * cos_lon / orbit_eq), g_hat * speed),
            (radial * (-radius * sin_lon / orbit_eq), -f_hat * speed),
            (
                (df_dh * cos_lon + dg_dh * sin_lon) * radius,
                (df_dh * vel_f + dg_dh * vel_g) * speed,
            ),
            (
                (df_dk * cos_lon + dg_dk * sin_lon) * radius,
                (df_dk * vel_f + dg_dk * vel_g) * speed,
            ),
            (
                radial * (radius * (self.ecc_f * sin_lon - self.ecc_g * cos_lon) / orbit_eq)
                    + (f_hat * -sin_lon + g_hat * cos_lon) * radius,
                (f_hat * -cos_lon + g_hat * -sin_lon) * speed,
            ),
        ];

        // The columns are built in the storage frame and rotated into `F` here. A change
        // of frame acts on the state, so it is a rotation applied on the left, block
        // diagonally and identically to the position and velocity halves of each column.
        let rotation = Ecliptic::rotation_to_frame::<F>();
        let mut jac = Matrix6::zeros();
        for (col, (d_pos, d_vel)) in columns.iter().enumerate() {
            jac.fixed_view_mut::<3, 1>(0, col)
                .copy_from(&(rotation * d_pos));
            jac.fixed_view_mut::<3, 1>(3, col)
                .copy_from(&(rotation * d_vel));
        }
        Ok(jac)
    }

    /// Jacobian of the six stored floats with respect to the epoch position and velocity,
    /// the inverse of [`Self::state_jacobian`].
    ///
    /// By the inverse function theorem this is exactly the inverse of that matrix, so it
    /// is computed as one rather than derived a second time. The frame parameter has the
    /// same meaning and the same reason as it does there: it names the frame the
    /// *columns* are in.
    ///
    /// # Errors
    /// Fails when the true longitude is outside the physical domain, or if the forward
    /// Jacobian is singular to working precision, which happens as the seam is
    /// approached.
    pub fn state_jacobian_inverse<F: InertialFrame>(&self) -> KeteResult<Matrix6<f64>> {
        self.state_jacobian::<F>()?.try_inverse().ok_or_else(|| {
            Error::ValueError(
                "The elements to state Jacobian is singular to working precision and \
                 cannot be inverted."
                    .into(),
            )
        })
    }

    /// The same orbit displaced by an offset of the six stored floats.
    ///
    /// The storage is unconstrained, so this is addition. The true longitude is not
    /// reduced; every use of it is through a sine or a cosine, and reducing here would
    /// cost the low bits of a small offset.
    #[must_use]
    pub fn displaced_by(&self, delta: &Vector6<f64>) -> Self {
        Self {
            desig: self.desig.clone(),
            epoch: self.epoch,
            semi_latus: self.semi_latus + delta[0],
            ecc_f: self.ecc_f + delta[1],
            ecc_g: self.ecc_g + delta[2],
            pole_h: self.pole_h + delta[3],
            pole_k: self.pole_k + delta[4],
            true_lon: self.true_lon + delta[5],
            center_id: self.center_id,
            gm_sqrt: self.gm_sqrt,
        }
    }

    /// The offset carrying `self` to `other`, the inverse of [`Self::displaced_by`].
    ///
    /// Five coordinates are a plain difference. The true longitude is reduced to the
    /// shortest signed angle, so two orbits a whole turn apart read as coincident rather
    /// than as wildly separated. The reduction subtracts a whole multiple of `2 pi`, so
    /// an offset already within half a turn is returned bit for bit; adding and removing
    /// `pi` instead would lose the low bits of a small difference.
    #[must_use]
    pub fn offset_to(&self, other: &Self) -> Vector6<f64> {
        let mut delta = Vector6::new(
            other.semi_latus - self.semi_latus,
            other.ecc_f - self.ecc_f,
            other.ecc_g - self.ecc_g,
            other.pole_h - self.pole_h,
            other.pole_k - self.pole_k,
            other.true_lon - self.true_lon,
        );
        delta[5] -= TAU * (delta[5] / TAU).round();
        delta
    }

    /// Eccentricity.
    #[must_use]
    pub fn eccentricity(&self) -> f64 {
        self.ecc_f.hypot(self.ecc_g)
    }

    /// Inverse of the semi major axis in 1/AU.
    ///
    /// The sign selects the conic and the value passes smoothly through zero at unit
    /// eccentricity, where the semi major axis itself diverges. Formed as
    /// `(1 - f^2 - g^2) / p` rather than from the eccentricity, which holds the absolute
    /// error near `eps / p` through the parabolic limit.
    #[must_use]
    pub fn inverse_semi_major(&self) -> f64 {
        (1.0 - self.ecc_f.powi(2) - self.ecc_g.powi(2)) / self.semi_latus
    }

    /// Semi major axis in AU. Infinity is returned if the orbit is parabolic.
    #[must_use]
    pub fn semi_major(&self) -> f64 {
        self.inverse_semi_major().recip()
    }

    /// Perihelion distance in AU. Finite and positive for every conic.
    #[must_use]
    pub fn peri_dist(&self) -> f64 {
        self.semi_latus / (1.0 + self.eccentricity())
    }

    /// Aphelion distance in AU. Infinity is returned if the orbit is not bound.
    #[must_use]
    pub fn aphelion(&self) -> f64 {
        let ecc = self.eccentricity();
        if ecc >= 1.0 {
            return f64::INFINITY;
        }
        self.semi_latus / (1.0 - ecc)
    }

    /// Orbital period in days. Infinity is returned if the orbit is not bound.
    #[must_use]
    pub fn orbital_period(&self) -> f64 {
        let alpha = self.inverse_semi_major();
        if alpha <= 0.0 {
            return f64::INFINITY;
        }
        TAU / (self.gm_sqrt * alpha.powf(1.5))
    }

    /// Inclination in radians, between 0 and pi.
    ///
    /// Regular at both poles as an output even though the storage is not: this is
    /// `2 atan(hypot(h, k))`, which is smooth in `(h, k)` everywhere and simply saturates
    /// at pi as the pair grows.
    #[must_use]
    pub fn inclination(&self) -> f64 {
        2.0 * self.pole_h.hypot(self.pole_k).atan()
    }

    /// Longitude of ascending node in radians, between 0 and 2 pi.
    ///
    /// Undefined for an uninclined orbit, where the pair `(h, k)` is zero and there is no
    /// node; zero is returned there rather than an error, matching `atan2`.
    #[must_use]
    pub fn lon_of_ascending(&self) -> f64 {
        self.pole_k.atan2(self.pole_h).rem_euclid(TAU)
    }

    /// Longitude of perihelion in radians, `omega + Omega`, between 0 and 2 pi.
    ///
    /// Unlike the argument of perihelion this is defined for an uninclined orbit, since it
    /// is measured from the reference direction rather than from the node.
    #[must_use]
    pub fn lon_of_peri(&self) -> f64 {
        self.ecc_g.atan2(self.ecc_f).rem_euclid(TAU)
    }

    /// Argument of perihelion in radians, between 0 and 2 pi.
    ///
    /// Meaningful only when both the eccentricity and the inclination are non-zero, since
    /// it is the angle from the node to perihelion and either can be absent.
    #[must_use]
    pub fn peri_arg(&self) -> f64 {
        (self.lon_of_peri() - self.lon_of_ascending()).rem_euclid(TAU)
    }

    /// True anomaly at the epoch in radians, between 0 and 2 pi.
    ///
    /// Closed form, `L_0` less the longitude of perihelion, so it is accurate to rounding
    /// rather than to a solver tolerance and is defined at every eccentricity. Measured
    /// from perihelion, so it is only meaningful when the eccentricity is non-zero.
    #[must_use]
    pub fn true_anomaly(&self) -> f64 {
        (self.true_lon - self.ecc_g.atan2(self.ecc_f)).rem_euclid(TAU)
    }

    /// Distance from the central body at the epoch in AU.
    ///
    /// This is `p / (1 + f cos L_0 + g sin L_0)`, the orbit equation evaluated at the
    /// epoch. It is only physical while that denominator is positive, see
    /// [`Self::try_to_state`].
    #[must_use]
    pub fn epoch_distance(&self) -> f64 {
        self.semi_latus / self.orbit_equation()
    }

    /// Construct from a position and velocity vector, in AU and AU/Day.
    ///
    /// The epoch of the elements is the time of the supplied state, so the true longitude
    /// is read directly off the position and no Kepler solve is needed.
    ///
    /// # Errors
    /// Fails if the motion is rectilinear, where the angular momentum vanishes and the
    /// orbit plane is undefined; if the orbit is exactly retrograde, where the pole sits
    /// on the seam and `(h, k)` is unbounded; or if the input contains non-finite values.
    pub(super) fn from_pos_vel(
        desig: Desig,
        epoch: Time<TDB>,
        pos: &Vector3<f64>,
        vel: &Vector3<f64>,
        center_id: i32,
        gm_sqrt: f64,
    ) -> KeteResult<Self> {
        let mu = gm_sqrt * gm_sqrt;
        let radius = pos.norm();
        let speed = vel.norm();
        if !(radius + speed).is_finite() {
            return Err(Error::ValueError(
                "Position and velocity must be finite.".into(),
            ));
        }

        let ang_vec = pos.cross(vel);
        let ang_mag = ang_vec.norm();

        // The orbit plane is undefined for rectilinear motion. Compare against the scale
        // of the cross product which produced it, so that the test is on whether the
        // angular momentum is distinguishable from rounding noise.
        if ang_mag <= f64::EPSILON * radius * speed {
            return Err(Error::ValueError(
                "Angular momentum is zero, the orbit is rectilinear and has no defined \
                 orbit plane."
                    .into(),
            ));
        }

        let w_hat = ang_vec / ang_mag;
        // The seam. `(h, k)` is a stereographic projection of the pole from the ecliptic
        // south pole, so an exactly retrograde orbit projects to infinity.
        let denom = 1.0 + w_hat.z;
        if denom <= 0.0 {
            return Err(Error::ValueError(
                "The orbit is exactly retrograde to the ecliptic, an inclination of 180 \
                 degrees, where the equinoctial pole components are unbounded."
                    .into(),
            ));
        }
        let pole_k = w_hat.x / denom;
        let pole_h = -w_hat.y / denom;

        let ecc_vec = ((speed.powi(2) - mu / radius) * pos - pos.dot(vel) * vel) / mu;

        // The pole components have to be formed before the basis, and the basis before
        // the remaining three.
        let scale_sq = 1.0 + pole_h * pole_h + pole_k * pole_k;
        let f_hat = Vector3::new(
            1.0 + pole_h * pole_h - pole_k * pole_k,
            2.0 * pole_h * pole_k,
            -2.0 * pole_k,
        ) / scale_sq;
        let g_hat = Vector3::new(
            2.0 * pole_h * pole_k,
            1.0 - pole_h * pole_h + pole_k * pole_k,
            2.0 * pole_h,
        ) / scale_sq;

        Ok(Self {
            desig,
            epoch,
            semi_latus: ang_mag * ang_mag / mu,
            ecc_f: ecc_vec.dot(&f_hat),
            ecc_g: ecc_vec.dot(&g_hat),
            pole_h,
            pole_k,
            true_lon: pos.dot(&g_hat).atan2(pos.dot(&f_hat)),
            center_id,
            gm_sqrt,
        })
    }

    /// Convert elements into a cartesian position and velocity at the epoch, in AU and
    /// AU/Day.
    ///
    /// This is closed form. There is no Kepler solve and no branch on eccentricity.
    ///
    /// # Errors
    /// Fails when the orbit equation `1 + f cos L_0 + g sin L_0` is not positive. The
    /// true longitude then points outside the asymptotes of an open orbit, which is not a
    /// configuration the object can reach. This cannot occur below unit eccentricity, where
    /// the orbit equation is bounded away from zero; at exactly unit eccentricity it
    /// vanishes at the single true longitude opposite perihelion.
    pub(super) fn to_pos_vel(&self) -> KeteResult<[[f64; 3]; 2]> {
        let orbit_eq = self.orbit_equation();
        if orbit_eq <= 0.0 {
            return Err(Self::domain_error());
        }
        let (sin_lon, cos_lon) = self.true_lon.sin_cos();
        let (f_hat, g_hat, _) = self.basis();

        let pos = (f_hat * cos_lon + g_hat * sin_lon) * (self.semi_latus / orbit_eq);
        // The identity `v = sqrt(GM/p) what x (e + rhat)` expanded in the basis. It is
        // exact on every conic and has no small denominators.
        let vel = (f_hat * -(self.ecc_g + sin_lon) + g_hat * (self.ecc_f + cos_lon))
            * (self.gm_sqrt / self.semi_latus.sqrt());

        Ok([[pos.x, pos.y, pos.z], [vel.x, vel.y, vel.z]])
    }

    /// The equinoctial basis `(fhat, ghat, what)`, orthonormal and right handed, with
    /// `what` the orbit pole.
    fn basis(&self) -> (Vector3<f64>, Vector3<f64>, Vector3<f64>) {
        let (pole_h, pole_k) = (self.pole_h, self.pole_k);
        let scale_sq = 1.0 + pole_h * pole_h + pole_k * pole_k;
        (
            Vector3::new(
                1.0 + pole_h * pole_h - pole_k * pole_k,
                2.0 * pole_h * pole_k,
                -2.0 * pole_k,
            ) / scale_sq,
            Vector3::new(
                2.0 * pole_h * pole_k,
                1.0 - pole_h * pole_h + pole_k * pole_k,
                2.0 * pole_h,
            ) / scale_sq,
            Vector3::new(
                2.0 * pole_k,
                -2.0 * pole_h,
                1.0 - pole_h * pole_h - pole_k * pole_k,
            ) / scale_sq,
        )
    }

    /// The orbit equation denominator `1 + f cos L_0 + g sin L_0`, which is `p / r` at
    /// the epoch. Positive exactly where the elements describe a reachable point.
    fn orbit_equation(&self) -> f64 {
        let (sin_lon, cos_lon) = self.true_lon.sin_cos();
        1.0 + self.ecc_f * cos_lon + self.ecc_g * sin_lon
    }

    /// The error returned wherever the true longitude leaves the reachable arc.
    fn domain_error() -> Error {
        Error::ValueError(
            "The true longitude lies outside the asymptotes of the open orbit, \
             1 + f cos(L) + g sin(L) must be positive."
                .into(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::GMS_SQRT;
    use crate::elements::CometElements;
    use crate::frames::Equatorial;
    use nalgebra::Rotation3;

    /// Build a position and velocity from classical elements, independent of the element
    /// implementation under test.
    ///
    /// Uses the conic equation `r = p / (1 + e cos nu)` with the perifocal velocity
    /// `sqrt(mu/p) (-sin nu, e + cos nu)`, then rotates by the argument of perihelion,
    /// the inclination, and the longitude of ascending node.
    fn conic_pos_vel(
        semi_latus: f64,
        ecc: f64,
        nu: f64,
        incl: f64,
        lon_asc: f64,
        peri_arg: f64,
    ) -> (Vector3<f64>, Vector3<f64>) {
        let (s_nu, c_nu) = nu.sin_cos();
        let radius = semi_latus / (1.0 + ecc * c_nu);
        let pos_plane = Vector3::new(radius * c_nu, radius * s_nu, 0.0);
        let vel_plane = Vector3::new(-s_nu, ecc + c_nu, 0.0) * (GMS_SQRT / semi_latus.sqrt());

        let rot = Rotation3::from_axis_angle(&Vector3::z_axis(), lon_asc)
            * Rotation3::from_axis_angle(&Vector3::x_axis(), incl)
            * Rotation3::from_axis_angle(&Vector3::z_axis(), peri_arg);
        (rot * pos_plane, rot * vel_plane)
    }

    /// The conic grid the Jacobian and conditioning checks sweep. Spans circular through
    /// hyperbolic, four epoch true anomalies including periapsis and apoapsis, and three
    /// inclinations including retrograde.
    const CHART_ECCS: [f64; 8] = [0.0, 0.01, 0.3, 0.7, 0.95, 0.999, 1.0, 1.5];
    const CHART_NUS: [f64; 4] = [0.0, 1.3, std::f64::consts::PI, 4.4];
    const CHART_INCLS: [f64; 3] = [
        0.0,
        std::f64::consts::FRAC_PI_2,
        // Retrograde, 162 degrees.
        std::f64::consts::PI * 0.9,
    ];

    // -----------------------------------------------------------------------
    // Modified equinoctial elements.
    //
    // Certification of `EquinoctialElements`. The grid spans circular through
    // hyperbolic and both degenerate inclinations, and every bound below is a measured
    // number printed by the test that asserts it.
    // -----------------------------------------------------------------------

    /// Build equinoctial elements from classical ones, independent of the
    /// implementation under test.
    fn build_equinoctial(
        semi_latus: f64,
        ecc: f64,
        nu: f64,
        incl: f64,
        lon_asc: f64,
        peri_arg: f64,
    ) -> (EquinoctialElements, Vector3<f64>, Vector3<f64>) {
        let (pos, vel) = conic_pos_vel(semi_latus, ecc, nu, incl, lon_asc, peri_arg);
        let elem = EquinoctialElements::from_pos_vel(
            Desig::Empty,
            2451545.0.into(),
            &pos,
            &vel,
            10,
            GMS_SQRT,
        )
        .expect("construction of valid elements must succeed");
        (elem, pos, vel)
    }

    /// The same at the fixed orientation the other tests use.
    fn build_equi(
        semi_latus: f64,
        ecc: f64,
        nu: f64,
        incl: f64,
    ) -> (EquinoctialElements, Vector3<f64>, Vector3<f64>) {
        build_equinoctial(semi_latus, ecc, nu, incl, 1.1, 2.3)
    }

    /// Walk the same conic grid the Jacobian checks use.
    fn equinoctial_grid(mut visit: impl FnMut(EquinoctialElements, Vector3<f64>, Vector3<f64>)) {
        for ecc in CHART_ECCS {
            for nu in CHART_NUS {
                // Apoapsis is not on an open conic, and it is where `1 + e cos(nu)`
                // vanishes for the parabolic case.
                if ecc >= 1.0 && 1.0 + ecc * nu.cos() < 1e-3 {
                    continue;
                }
                for incl in CHART_INCLS {
                    let (elem, _, _) = build_equi(1.7, ecc, nu, incl);
                    let [pos, vel] = elem.to_pos_vel().expect("grid elements are physical");
                    visit(elem, pos.into(), vel.into());
                }
            }
        }
    }

    /// The basis is what everything else is expressed in, and it is the one place where
    /// a transcription error would be invisible to every self-consistent check: a round
    /// trip and a Jacobian both use the same basis on the way in and the way out, so
    /// they pass under any labeling of `(h, k)`.
    ///
    /// This pins it externally. The basis must be orthonormal and right handed with
    /// `fhat x ghat = what`, `what` must be the orbit pole, and the stored pair must
    /// satisfy `h = tan(i/2) cos(Omega)` and `k = tan(i/2) sin(Omega)` against
    /// independently constructed angles.
    #[test]
    fn equinoctial_basis_matches_its_definition() {
        const TOL: f64 = 1e-14;
        let mut worst_frame = 0.0_f64;
        let mut worst_pole = 0.0_f64;

        for &incl in &[0.0, 0.3, 1.2, std::f64::consts::FRAC_PI_2, 2.4, 3.0] {
            for &lon_asc in &[0.0, 0.7, 2.5, 4.9] {
                let (elem, pos, vel) = build_equinoctial(1.6, 0.4, 0.9, incl, lon_asc, 1.3);
                let (f_hat, g_hat, w_hat) = elem.basis();

                worst_frame = worst_frame
                    .max((f_hat.norm() - 1.0).abs())
                    .max((g_hat.norm() - 1.0).abs())
                    .max((w_hat.norm() - 1.0).abs())
                    .max(f_hat.dot(&g_hat).abs())
                    .max(f_hat.dot(&w_hat).abs())
                    .max(g_hat.dot(&w_hat).abs())
                    .max((f_hat.cross(&g_hat) - w_hat).norm());

                // `what` is the orbit pole, independently of the elements.
                let ang = pos.cross(&vel).normalize();
                worst_frame = worst_frame.max((w_hat - ang).norm());

                // The stored pair against its definition, relative to its own size:
                // `tan(i/2)` reaches 14 at the most inclined case here and carries the
                // rounding of the state it was built from with it. The node is undefined
                // at zero inclination, where both components vanish and this is trivial.
                let tan_half = (incl / 2.0).tan();
                worst_pole = worst_pole
                    .max((elem.pole_h - tan_half * lon_asc.cos()).abs() / tan_half.max(1.0))
                    .max((elem.pole_k - tan_half * lon_asc.sin()).abs() / tan_half.max(1.0));
            }
        }
        println!("basis orthonormality {worst_frame:e}, pole definition {worst_pole:e}");
        assert!(worst_frame < TOL, "basis residual {worst_frame:e}");
        assert!(worst_pole < TOL, "pole definition residual {worst_pole:e}");
    }

    #[test]
    fn equinoctial_roundtrip_all_conics() {
        const TOL: f64 = 1e-14;

        // Spans circular through hyperbolic, straddling both branch points that
        // `CometElements` has to special case.
        let eccs = [
            0.0,
            1e-12,
            1e-6,
            0.1,
            0.5,
            0.9,
            1.0 - 1e-12,
            1.0,
            1.0 + 1e-12,
            1.5,
            3.0,
        ];
        // Includes the polar and both degenerate-node inclinations.
        let incls = [0.0, 1e-12, 0.4, std::f64::consts::FRAC_PI_2, 3.0];
        // Kept inside |nu| < arccos(-1/e) so every case stays on the physical branch of
        // the most eccentric orbit tested.
        let nus = [0.0, 0.7, 1.5, -1.3];

        let mut worst = 0.0_f64;
        let mut worst_case = String::new();
        for &ecc in &eccs {
            for &incl in &incls {
                for &nu in &nus {
                    let (elem, pos, vel) = build_equi(1.7, ecc, nu, incl);
                    let [new_pos, new_vel] = elem.to_pos_vel().expect("conversion must succeed");
                    let err = ((Vector3::from(new_pos) - pos).norm() / pos.norm())
                        .max((Vector3::from(new_vel) - vel).norm() / vel.norm());
                    if err > worst {
                        worst = err;
                        worst_case = format!("e={ecc} i={incl} nu={nu}");
                    }
                }
            }
        }
        println!("worst state roundtrip relative error: {worst:e} at {worst_case}");
        assert!(worst < TOL, "roundtrip error {worst:e} exceeded {TOL:e}");
    }

    #[test]
    fn equinoctial_regular_at_branch_points() {
        // `CometElements` switches formulas at |e - 1| < PARABOLIC_ECC_LIMIT and at
        // e < 1e-6. This set has no branch, so these must round trip at the precision of
        // any other eccentricity.
        const TOL: f64 = 4e-16;

        let mut worst = 0.0_f64;
        for &ecc in &[0.0, 1e-14, 1.0 - 1e-14, 1.0, 1.0 + 1e-14] {
            let (elem, pos, _) = build_equi(1.9, ecc, 1.1, 0.3);
            let [new_pos, _] = elem.to_pos_vel().expect("conversion must succeed");
            let err = (Vector3::from(new_pos) - pos).norm() / pos.norm();
            println!("ecc {ecc:e}: roundtrip {err:e}");
            worst = worst.max(err);

            // The semi-latus rectum and perihelion distance stay finite and positive
            // through the parabolic limit, unlike the semi major axis.
            assert!(elem.semi_latus > 0.0);
            assert!(elem.peri_dist() > 0.0);
        }
        assert!(
            worst < TOL,
            "branch point roundtrip {worst:e} exceeded {TOL:e}"
        );
    }

    /// Every derived angle and size against `CometElements` built from the same state.
    ///
    /// This is the check that certifies the derived quantities as a group: `i`, `Omega`,
    /// `omega` and `nu` are each recovered from `(h, k)`, `(f, g)` and `L_0` by an
    /// `atan2` whose convention cannot be verified from inside the element set.
    #[test]
    fn equinoctial_agree_with_comet_elements() {
        const TOL: f64 = 1e-11;
        let mut worst = 0.0_f64;

        for &ecc in &[0.1, 0.5, 0.9, 1.5] {
            for &incl in &[0.3, 1.2, 2.6] {
                for &lon_asc in &[0.4, 3.9] {
                    let (elem, pos, vel) = build_equinoctial(2.1, ecc, 0.8, incl, lon_asc, 1.7);
                    let comet = CometElements::from_pos_vel(
                        Desig::Empty,
                        2451545.0.into(),
                        &pos,
                        &vel,
                        10,
                        GMS_SQRT,
                    );

                    // Angles are compared modulo a turn: the two sets place their branch
                    // cuts differently, so a case sitting near one of them otherwise
                    // reads as `2 pi` apart while describing one orbit.
                    let angle_diff = |a: f64, b: f64| {
                        let d = a - b;
                        (d - TAU * (d / TAU).round()).abs()
                    };
                    worst = worst
                        .max((elem.eccentricity() - comet.eccentricity).abs())
                        .max((elem.peri_dist() - comet.peri_dist).abs())
                        .max(angle_diff(elem.inclination(), comet.inclination))
                        .max(angle_diff(elem.lon_of_ascending(), comet.lon_of_ascending))
                        .max(angle_diff(elem.peri_arg(), comet.peri_arg));
                    if ecc < 1.0 {
                        worst = worst.max((elem.semi_major() - comet.semi_major()).abs());
                        assert!(
                            (elem.orbital_period() - comet.orbital_period()).abs() < 1e-6,
                            "period mismatch"
                        );
                    }

                    // The true anomaly comparison has a different error budget from the
                    // rest. `CometElements` reaches it through state -> peri_time ->
                    // mean anomaly -> eccentric anomaly, so it carries the Newton solve's
                    // tolerance amplified by `dnu/dE`. The equinoctial value is closed form
                    // and is certified alone in `equinoctial_true_anomaly_closed_form`.
                    let nu_diff = angle_diff(elem.true_anomaly(), comet.true_anomaly().unwrap());
                    assert!(nu_diff < 1e-9, "true anomaly difference {nu_diff:e}");
                }
            }
        }
        println!("worst derived quantity difference vs CometElements: {worst:e}");
        assert!(worst < TOL, "derived quantity difference {worst:e}");
    }

    #[test]
    fn equinoctial_true_anomaly_closed_form() {
        // `nu = L_0 - atan2(g, f)` in the stored components, so it reproduces the
        // constructed value to rounding at every eccentricity with no solver in the path.
        const TOL: f64 = 4e-15;

        let mut worst = 0.0_f64;
        for &ecc in &[0.1, 0.5, 0.9, 1.0, 1.5, 3.0] {
            for &nu in &[0.3, 0.8, 1.5] {
                let (elem, _, _) = build_equi(1.6, ecc, nu, 0.5);
                worst = worst.max((elem.true_anomaly() - nu).abs());
            }
        }
        println!("worst closed form true anomaly error: {worst:e}");
        assert!(worst < TOL, "true anomaly error {worst:e} exceeded {TOL:e}");
    }

    #[test]
    fn equinoctial_comet_conversion_roundtrip() {
        const TOL: f64 = 1e-10;

        for &ecc in &[0.1, 0.7, 1.5] {
            let (elem, _, _) = build_equi(1.4, ecc, 0.9, 0.7);
            let comet = CometElements::try_from(&elem).expect("conversion must succeed");
            let back = EquinoctialElements::try_from(&comet).expect("conversion must succeed");

            assert!((back.semi_latus - elem.semi_latus).abs() / elem.semi_latus < TOL);
            assert!((back.ecc_f - elem.ecc_f).abs() < TOL);
            assert!((back.ecc_g - elem.ecc_g).abs() < TOL);
            assert!((back.pole_h - elem.pole_h).abs() < TOL);
            assert!((back.pole_k - elem.pole_k).abs() < TOL);
            assert!((back.true_lon - elem.true_lon).abs() < TOL);
        }
    }

    /// The phase column carries the derivation. It is the Keplerian flow scaled by the
    /// true-longitude rate, an exact undifferenced identity, and is the sharpest
    /// available check on the whole Jacobian.
    #[test]
    fn equinoctial_jacobian_phase_column_is_kepler_flow() {
        // The identity itself is exact. What is measured is the rounding of the two
        // sides, which is set by the orbit equation's cancellation: it is formed as a
        // difference from one, so its relative error is `eps / orbit_eq`, and the bound
        // is set by the grid point where that denominator is smallest.
        const TOL: f64 = 1e-13;
        let mut worst = 0.0_f64;
        let mut worst_case = String::new();
        equinoctial_grid(|elem, pos, vel| {
            let jac = elem
                .state_jacobian::<Ecliptic>()
                .expect("jacobian must exist on the grid");
            // dL/dt = sqrt(GM p) / r^2
            let rate = (elem.gm_sqrt.powi(2) * elem.semi_latus).sqrt() / pos.norm_squared();
            let d_pos: Vector3<f64> = jac.fixed_view::<3, 1>(0, 5).into();
            let d_vel: Vector3<f64> = jac.fixed_view::<3, 1>(3, 5).into();

            let accel = -pos * (elem.gm_sqrt.powi(2) / pos.norm().powi(3));
            let residual = ((d_pos * rate - vel).norm() / vel.norm())
                .max((d_vel * rate - accel).norm() / accel.norm());
            if residual > worst {
                worst = residual;
                worst_case = format!(
                    "e={:.4} nu={:.3} r_0={:.4} eps/orbit_eq={:.2e}",
                    elem.eccentricity(),
                    elem.true_anomaly(),
                    elem.epoch_distance(),
                    f64::EPSILON / elem.orbit_equation()
                );
            }
        });
        println!("worst phase column vs Keplerian flow: {worst:e} at {worst_case}");
        assert!(worst < TOL, "phase column {worst:e} exceeded {TOL:e}");
    }

    /// The frame parameter on the Jacobian must actually rotate, and must rotate the
    /// right way.
    ///
    /// This is the check a self-consistent one cannot make. Every other Jacobian test uses
    /// the storage frame on both sides, so it would pass unchanged if the rotation were
    /// transposed, inverted, or absent. Here the reference is a central difference of the
    /// state *expressed in the target frame*, built by rotating each perturbed state
    /// through [`State::into_frame`], which is independent machinery.
    #[test]
    fn equinoctial_jacobian_carries_its_frame() {
        let (elem, _, _) = build_equi(1.6, 0.35, 0.9, 1.1);
        let ecliptic = elem.state_jacobian::<Ecliptic>().unwrap();
        let equatorial = elem.state_jacobian::<Equatorial>().unwrap();

        // Non-vacuous: the two must differ by the obliquity, or nothing below means
        // anything.
        let separation = (equatorial - ecliptic).norm() / ecliptic.norm();
        println!("ecliptic vs equatorial Jacobian: {separation:e} relative");
        assert!(separation > 0.1, "the frames are indistinguishable");

        // Against central differences of the Equatorial state.
        let orbit_eq = elem.semi_latus / elem.epoch_distance();
        let steps = [
            1e-6 * elem.semi_latus,
            1e-6 * orbit_eq,
            1e-6 * orbit_eq,
            1e-6,
            1e-6,
            1e-6 * orbit_eq,
        ];
        let mut worst = 0.0_f64;
        for (col, step) in steps.iter().copied().enumerate() {
            let mut delta = Vector6::zeros();
            delta[col] = step;
            let plus: State<Equatorial> = elem
                .displaced_by(&delta)
                .try_to_state()
                .unwrap()
                .into_frame();
            delta[col] = -step;
            let minus: State<Equatorial> = elem
                .displaced_by(&delta)
                .try_to_state()
                .unwrap()
                .into_frame();

            let mut fd = Vector6::zeros();
            for row in 0..3 {
                fd[row] = (plus.pos[row] - minus.pos[row]) / (2.0 * step);
                fd[row + 3] = (plus.vel[row] - minus.vel[row]) / (2.0 * step);
            }
            worst = worst.max((fd - equatorial.column(col)).norm() / equatorial.column(col).norm());
        }
        println!("equatorial Jacobian vs central differences: {worst:e} relative");
        assert!(worst < 1e-8, "frame-carried Jacobian {worst:e}");

        // And the inverse inverts in the same frame.
        let inverse = elem.state_jacobian_inverse::<Equatorial>().unwrap();
        let residual = (inverse * equatorial - Matrix6::identity()).norm();
        assert!(residual < 1e-8, "inverse residual {residual:e}");
    }

    #[test]
    fn equinoctial_jacobian_matches_central_differences() {
        // Multiple of the predicted floor that counts as agreement. The floor estimate
        // ignores the O(step^2) truncation half and the constant in front of the
        // rounding half, so it is a scale rather than a bound.
        const TOL: f64 = 20.0;
        // Relative floor, for columns where the rounding term is negligible.
        const REL_FLOOR: f64 = 1e-9;
        let mut worst = 0.0_f64;
        let mut worst_case = String::new();

        equinoctial_grid(|elem, _, _| {
            let jac = elem
                .state_jacobian::<Ecliptic>()
                .expect("jacobian must exist on the grid");

            // Each step is scaled to the curvature of the function along that
            // coordinate. Five of the six are dimensionless or radians with O(1)
            // curvature; the semi-latus rectum column is scaled by its own size. The
            // shape and phase columns carry the orbit equation `1 + f cos L + g sin L`
            // in their denominators, whose higher derivatives blow up as it shrinks, so
            // their steps carry that factor.
            let orbit_eq = elem.orbit_equation();
            let steps = [
                1e-6 * elem.semi_latus,
                1e-6 * orbit_eq,
                1e-6 * orbit_eq,
                1e-6,
                1e-6,
                1e-6 * orbit_eq,
            ];

            for (col, step) in steps.iter().copied().enumerate() {
                let mut delta = Vector6::zeros();
                delta[col] = step;
                let plus = elem.displaced_by(&delta);
                delta[col] = -step;
                let minus = elem.displaced_by(&delta);
                let (Ok([p_pos, p_vel]), Ok([m_pos, m_vel])) =
                    (plus.to_pos_vel(), minus.to_pos_vel())
                else {
                    continue;
                };

                let mut fd = Vector6::zeros();
                for row in 0..3 {
                    fd[row] = (p_pos[row] - m_pos[row]) / (2.0 * step);
                    fd[row + 3] = (p_vel[row] - m_vel[row]) / (2.0 * step);
                }
                // Rounding in the difference of two states of size `|state|`, divided by
                // the step, is the error the reference cannot go below.
                let state_scale = Vector6::from_column_slice(&[
                    p_pos[0], p_pos[1], p_pos[2], p_vel[0], p_vel[1], p_vel[2],
                ])
                .norm();
                let analytic = jac.column(col);
                // The orbit equation is formed by cancellation, so its absolute rounding
                // is `eps` however small it becomes, and the distance `p / orbit_eq`
                // carries that as a relative error of `eps / orbit_eq`. Only the columns
                // whose perturbation moves the orbit equation see it in the difference;
                // for the others it is identical on both sides and cancels exactly. At
                // apoapsis of the `e = 0.999` grid point it is a thousand times the plain
                // differencing floor and is what actually limits the reference there.
                let cancellation = if matches!(col, 1 | 2 | 5) {
                    orbit_eq
                } else {
                    1.0
                };
                let floor = f64::EPSILON * state_scale / (cancellation * step)
                    + REL_FLOOR * analytic.norm();
                let ratio = (fd - analytic).norm() / floor;
                if ratio > worst {
                    worst = ratio;
                    worst_case = format!(
                        "col={col} e={:.4} nu={:.3} r_0={:.4} floor/|col|={:.2e}",
                        elem.eccentricity(),
                        elem.true_anomaly(),
                        elem.epoch_distance(),
                        floor / analytic.norm()
                    );
                }
            }
        });
        println!(
            "jacobian vs central differences, worst multiple of floor: {worst:.3} at {worst_case}"
        );
        assert!(worst < TOL, "jacobian {worst:.3} exceeded {TOL} floors");
    }

    /// `J K - I` in the coordinates' own per-coordinate scale. The product maps the
    /// element coordinates to themselves, so its entries carry units of one coordinate
    /// over another and its raw norm says nothing.
    #[test]
    fn equinoctial_jacobian_inverse_residual() {
        const TOL: f64 = 1e-6;
        let mut worst = 0.0_f64;
        equinoctial_grid(|elem, _, _| {
            let jac = elem
                .state_jacobian::<Ecliptic>()
                .expect("jacobian exists on the grid");
            let Ok(inv) = elem.state_jacobian_inverse::<Ecliptic>() else {
                return;
            };
            // scale[j] is the state response to a unit step in coordinate j, so
            // dividing entry (i, j) by scale[j] and multiplying by scale[i] turns it into
            // a relative error. The raw norm carries units of one coordinate over another
            // and says nothing.
            let scale: Vec<f64> = (0..6).map(|j| jac.column(j).norm()).collect();
            let raw = inv * jac - Matrix6::identity();
            for row in 0..6 {
                for col in 0..6 {
                    worst = worst.max((raw[(row, col)] * scale[row] / scale[col]).abs());
                }
            }
        });
        println!("worst scaled inverse residual: {worst:e}");
        assert!(worst < TOL, "inverse residual {worst:e} exceeded {TOL:e}");
    }

    /// The conditioning of the state Jacobian over the conic grid, raw and after row and
    /// column equilibration. Most of the raw number is unit bookkeeping: the matrix mixes
    /// AU with AU/day down its rows and AU, dimensionless and radians across its columns.
    #[test]
    fn equinoctial_jacobian_conditioning() {
        let mut raws = Vec::new();
        let mut scaleds = Vec::new();
        equinoctial_grid(|elem, _, _| {
            let jac = elem
                .state_jacobian::<Ecliptic>()
                .expect("jacobian exists on the grid");
            let (raw, scaled) = cond_pair(&jac);
            raws.push(raw);
            scaleds.push(scaled);
        });
        raws.sort_by(f64::total_cmp);
        scaleds.sort_by(f64::total_cmp);
        let pick = |v: &[f64], numer: usize, denom: usize| v[(v.len() - 1) * numer / denom];
        println!(
            "cond(K) raw:         median {:.3e}  90th {:.3e}  worst {:.3e}",
            pick(&raws, 1, 2),
            pick(&raws, 9, 10),
            raws[raws.len() - 1]
        );
        println!(
            "cond(K) equilibrated: median {:.3e}  90th {:.3e}  worst {:.3e}",
            pick(&scaleds, 1, 2),
            pick(&scaleds, 9, 10),
            scaleds[scaleds.len() - 1]
        );
        // The equilibrated number is the real anisotropy of the coordinates and is what the
        // seam profile below tracks. The bound only catches the Jacobian going singular
        // somewhere on the grid; the worst case sits at the apoapsis of the `e = 0.999`
        // orbit, a deliberately extreme epoch placement rather than anything kete fits.
        assert!(
            scaleds[scaleds.len() - 1] < 1e5,
            "equilibrated conditioning {:.3e}",
            scaleds[scaleds.len() - 1]
        );
    }

    /// The true longitude wraps, and differencing must reduce to the shortest signed
    /// angle. Two orbits a whole turn apart are the same orbit and must read as
    /// coincident, not as separated by `2 pi`.
    #[test]
    fn equinoctial_offset_reduces_the_true_longitude() {
        let (elem, _, _) = build_equi(1.7, 0.3, 0.9, 0.5);

        // A whole turn is the identity on the orbit.
        let mut turn = Vector6::zeros();
        turn[5] = TAU;
        let wrapped = elem.displaced_by(&turn);
        let offset = elem.offset_to(&wrapped);
        println!("offset after a full turn: {:e} rad", offset[5].abs());
        assert!(offset[5].abs() < 1e-15, "full turn read as {:e}", offset[5]);

        // Nine turns and a bit, reduced onto the principal branch.
        let mut far = Vector6::zeros();
        far[5] = 9.0_f64.mul_add(TAU, 0.75);
        let recovered = elem.offset_to(&elem.displaced_by(&far))[5];
        assert!(
            (recovered - 0.75).abs() < 1e-14,
            "reduced offset {recovered:e}"
        );

        // A small offset is returned bit for bit. Subtracting a whole multiple of 2 pi
        // leaves it untouched, where a reduction written as an add and a subtract of pi
        // would round it away entirely. The pair is built by setting the field rather
        // than by displacing, since adding 1e-18 to a longitude of order one is itself
        // absorbed by rounding.
        let mut near = elem.clone();
        near.true_lon = 0.0;
        let mut tiny = near.clone();
        tiny.true_lon = 1e-18;
        assert_eq!(near.offset_to(&tiny)[5], 1e-18);

        // The other five are a plain difference, so they round trip to the rounding of
        // the addition itself and to nothing worse.
        let delta = Vector6::new(0.1, -0.02, 0.03, 0.04, -0.05, 0.2);
        let back = elem.offset_to(&elem.displaced_by(&delta));
        let worst = (0..6)
            .map(|i| (back[i] - delta[i]).abs())
            .fold(0.0_f64, f64::max);
        println!("worst offset round trip: {worst:e}");
        assert!(worst < 1e-15, "offset round trip {worst:e}");
    }

    #[test]
    fn equinoctial_physical_accessors() {
        const TOL: f64 = 1e-13;
        for &ecc in &[0.0, 0.3, 0.9, 1.0, 1.5] {
            let (elem, pos, _) = build_equi(1.45, ecc, 0.85, 0.9);

            assert!((elem.eccentricity() - ecc).abs() < TOL);
            assert!((elem.epoch_distance() - pos.norm()).abs() / pos.norm() < TOL);
            assert!((elem.peri_dist() - 1.45 / (1.0 + ecc)).abs() < TOL);
            assert!((elem.inverse_semi_major() - (1.0 - ecc * ecc) / 1.45).abs() < TOL);

            if ecc < 1.0 {
                assert!((elem.aphelion() - 1.45 / (1.0 - ecc)).abs() < TOL);
                assert!(elem.orbital_period().is_finite());
            } else {
                // At exactly unit eccentricity the computed value lands a few ulp either
                // side of one, so these come back either infinite or astronomically
                // large. Both say the orbit is not bound; neither can be sharpened,
                // because unit eccentricity is not a detectable state in floating point.
                assert!(elem.aphelion() > 1e15, "aphelion {}", elem.aphelion());
                assert!(
                    elem.orbital_period() > 1e15,
                    "period {}",
                    elem.orbital_period()
                );
            }
        }
    }

    /// Rectilinear motion has no orbit plane, and the retrograde pole is the one
    /// direction the storage cannot encode. Both are rejected at construction.
    #[test]
    fn equinoctial_reject_undefined_orbits() {
        let radial = Vector3::new(1.0, 0.0, 0.0);
        assert!(
            EquinoctialElements::from_pos_vel(
                Desig::Empty,
                2451545.0.into(),
                &radial,
                &(radial * 0.01),
                10,
                GMS_SQRT,
            )
            .is_err()
        );

        // Exactly retrograde: the pole is the ecliptic south pole and `(h, k)` diverges.
        let (pos, vel) = conic_pos_vel(1.0, 0.2, 0.5, std::f64::consts::PI, 0.0, 0.0);
        let err = EquinoctialElements::from_pos_vel(
            Desig::Empty,
            2451545.0.into(),
            &pos,
            &vel,
            10,
            GMS_SQRT,
        );
        assert!(err.is_err(), "exactly retrograde must be rejected");

        // One degree away from it is ordinary.
        let (pos, vel) = conic_pos_vel(1.0, 0.2, 0.5, 179.0_f64.to_radians(), 0.0, 0.0);
        let near = EquinoctialElements::from_pos_vel(
            Desig::Empty,
            2451545.0.into(),
            &pos,
            &vel,
            10,
            GMS_SQRT,
        )
        .expect("one degree from the seam is an ordinary orbit");
        assert!((near.inclination().to_degrees() - 179.0).abs() < 1e-10);
    }

    /// For `e > 1` a displacement of the shape or phase coordinates can drive the true
    /// longitude outside the asymptotes, whose only symptom otherwise is a silently
    /// negative distance.
    #[test]
    fn equinoctial_reject_unreachable_longitude() {
        let (elem, _, _) = build_equi(1.5, 3.0, 0.5, 0.4);
        let mut delta = Vector6::zeros();
        // Move the phase to the far side, where `1 + f cos L + g sin L` is negative.
        delta[5] = std::f64::consts::PI;
        let flipped = elem.displaced_by(&delta);
        assert!(flipped.orbit_equation() < 0.0);
        assert!(flipped.to_pos_vel().is_err());
        assert!(flipped.state_jacobian::<Ecliptic>().is_err());

        // The same displacement on a bound orbit cannot leave the domain.
        let (bound, _, _) = build_equi(1.5, 0.3, 0.5, 0.4);
        assert!(bound.displaced_by(&delta).to_pos_vel().is_ok());
    }

    /// Condition number, raw and after row/column equilibration.
    fn cond_pair(jac: &Matrix6<f64>) -> (f64, f64) {
        // nalgebra's SVD iterates with no cap and does not terminate on a non-finite
        // entry, so this has to be checked rather than discovered.
        if !jac.iter().all(|v| v.is_finite()) {
            return (f64::INFINITY, f64::INFINITY);
        }
        let sv = jac.svd(false, false).singular_values;
        let raw = sv.max() / sv.min();
        let mut scaled = *jac;
        for mut row in scaled.row_iter_mut() {
            let n = row.norm();
            if n > 0.0 {
                row /= n;
            }
        }
        for mut col in scaled.column_iter_mut() {
            let n = col.norm();
            if n > 0.0 {
                col /= n;
            }
        }
        let ssv = scaled.svd(false, false).singular_values;
        (raw, ssv.max() / ssv.min())
    }

    /// Build a state at inclination `180 - psi` degrees, i.e. `psi` degrees from the
    /// seam, with the given eccentricity.
    fn seam_state(psi_deg: f64, ecc: f64) -> State<Ecliptic> {
        let elements = CometElements {
            desig: Desig::Empty,
            epoch: Time::new(2460000.5),
            eccentricity: ecc,
            // Hold the semi-latus rectum fixed at 1 AU rather than the perihelion
            // distance. Fixing `q` instead confounds eccentricity with a shrinking
            // perihelion and makes the conditioning look far worse than it is at high
            // `e`.
            peri_dist: 1.0 / (1.0 + ecc),
            peri_time: Time::new(2459950.5),
            lon_of_ascending: std::f64::consts::FRAC_PI_4,
            peri_arg: std::f64::consts::FRAC_PI_3,
            inclination: (180.0 - psi_deg).to_radians(),
            center_id: 10,
            gm_sqrt: GMS_SQRT,
        };
        // Deliberately NOT converted to another frame: `CometElements` is defined by
        // ecliptic angles, so a frame change tilts the pole by the obliquity and the
        // sweep would never reach the seam.
        elements.try_to_state().unwrap()
    }

    /// How the seam degrades with angular distance from it.
    ///
    /// The raw condition number grows as `csc^2(psi/2) ~ 4/psi^2` while the
    /// **equilibrated** number stays flat, because a stereographic projection is
    /// conformal and therefore introduces no anisotropy. The seam is unit bookkeeping in
    /// the point representation; the round-trip column says where the arithmetic
    /// actually fails.
    ///
    /// Run with `cargo test -- --ignored --nocapture`.
    #[test]
    #[ignore = "measurement, prints a table"]
    fn equinoctial_seam_conditioning_profile() {
        println!(
            "{:>12}  {:>10}  {:>11}  {:>11}  {:>11}",
            "psi (deg)", "tan(i/2)", "raw", "equilibrated", "roundtrip"
        );
        for &psi in &[
            90.0_f64, 30.0, 10.0, 3.0, 1.0, 0.3, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6,
        ] {
            let state = seam_state(psi, 0.3);
            let pos = Vector3::from(state.pos);
            let vel = Vector3::from(state.vel);
            let elem = EquinoctialElements::from_state(&state).unwrap();
            let (raw, equil) = cond_pair(&elem.state_jacobian::<Ecliptic>().unwrap());

            let [back_pos, back_vel] = elem.to_pos_vel().unwrap();
            let round = ((Vector3::from(back_pos) - pos).norm() / pos.norm())
                .max((Vector3::from(back_vel) - vel).norm() / vel.norm());

            println!(
                "{psi:>12.1e}  {:>10.3e}  {raw:>11.3e}  {equil:>11.3e}  {round:>11.3e}",
                elem.pole_h.hypot(elem.pole_k)
            );
        }

        println!(
            "\n  at psi = 45 deg, across eccentricity:\n{:>12}  {:>11}  {:>11}",
            "e", "raw", "equilibrated"
        );
        for &ecc in &[0.0_f64, 0.1, 0.3, 0.7, 0.95, 0.999, 1.0, 1.5, 3.0] {
            let state = seam_state(45.0, ecc);
            let elem = EquinoctialElements::from_state(&state).unwrap();
            let (raw, equil) = cond_pair(&elem.state_jacobian::<Ecliptic>().unwrap());
            println!("{ecc:>12.3}  {raw:>11.3e}  {equil:>11.3e}");
        }
    }
}
