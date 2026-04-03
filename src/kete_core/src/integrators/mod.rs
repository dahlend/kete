//! # Numerical Integrators
//! Numerical ODE integrators for orbit propagation.
//!
//! There are a number of integrators implemented here, with different tradeoffs in
//! terms of accuracy, speed, and robustness. After extensive testing of all of them,
//! attempts at tuning them to the best possible performance, the clear overall winner
//! in terms of accuracy is the 15th-order Radau integrator. Therefore, it is the
//! default for all propagation in kete. In some cases other integrators were faster,
//! but when testing a diverse set of realistic orbits, Radau was typically the best.
//!
//! The next best was actually the Picard integrator, however its accuracy came at a
//! higher cpu cost than Radau. It is am integrator which maps very well to GPUs, so
//! it may be revisited in the future if GPU support is added to kete.
//!
//! These integrators are being left here for completeness, and users may want to
//! experiment for themselves. There is a set of simple stress tests which may be
//! called to demonstrate some of the differences in performance between the different
//! integrators. These tests are very simple, but informative.
//!
//!

mod bulirsch_stoer;
mod gauss_jackson;
mod picard;
mod radau;
mod runge_kutta;
mod util;

pub use bulirsch_stoer::BulirschStoerIntegrator;
pub use gauss_jackson::GaussJacksonIntegrator;
pub use picard::{
    PC15, PC25, PicardIntegrator, PicardStep, PicardStepSecondOrder, dumb_picard_init,
    dumb_picard_init_second_order,
};
pub use radau::RadauIntegrator;
pub use runge_kutta::RK45Integrator;

#[cfg(test)]
mod stress_tests;
