// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! # Orbital Elements
//!
//! Two-body orbital element sets and their conversions to and from
//! [`State`](crate::state::State).
//!
//! [`CometElements`] is the classical set, defined by the perihelion distance,
//! eccentricity and three angles. It is what external catalogs publish and what
//! kete reports.
//!
//! [`EquinoctialElements`] is the modified equinoctial set: six unconstrained floats
//! with no singularity at zero eccentricity or zero inclination, which is what
//! [`UncertainState`](crate::state::UncertainState) stores a covariance over.

mod cometary;
mod equinoctial;

pub use cometary::CometElements;
pub use equinoctial::EquinoctialElements;

use crate::errors::Error;
use crate::forces::GravParams;
use crate::prelude::KeteResult;

use nalgebra::Vector3;

/// Square root of the gravitational parameter of a NAIF center id.
///
/// Elements carry `gm_sqrt` rather than `mu`, since that is the form every conversion
/// below uses. The lookup and its error live on [`GravParams::try_mass_from_naif_id`].
///
/// # Errors
/// Fails if `center_id` has no entry in the mass table.
fn gm_sqrt_for_center(center_id: i32) -> KeteResult<f64> {
    Ok(GravParams::try_mass_from_naif_id(center_id)?.sqrt())
}

/// [`CometElements`] is defined by ecliptic angles, which is the frame
/// [`EquinoctialElements`] stores, so the conversion needs no rotation.
impl TryFrom<&CometElements> for EquinoctialElements {
    type Error = Error;

    fn try_from(elem: &CometElements) -> KeteResult<Self> {
        let [pos, vel] = elem.to_pos_vel()?;
        Self::from_pos_vel(
            elem.desig.clone(),
            elem.epoch,
            &Vector3::from(pos),
            &Vector3::from(vel),
            elem.center_id,
            elem.gm_sqrt,
        )
    }
}

impl TryFrom<&EquinoctialElements> for CometElements {
    type Error = Error;

    fn try_from(elem: &EquinoctialElements) -> KeteResult<Self> {
        let [pos, vel] = elem.to_pos_vel()?;
        Ok(Self::from_pos_vel(
            elem.desig.clone(),
            elem.epoch,
            &Vector3::from(pos),
            &Vector3::from(vel),
            elem.center_id,
            elem.gm_sqrt,
        ))
    }
}
