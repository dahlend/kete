//! # Propagation
//! The motion of objects (represented by a [`crate::state::State`]) as a function of time.
//! There are multiple levels of precision available, each with different pros/cons
//! (usually performance related).
// BSD 3-Clause License
//
// Copyright (c) 2026, Dar Dahlen
// Copyright (c) 2025, California Institute of Technology
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

mod acceleration;
mod kepler;
mod nongrav;
mod picard;
mod radau;
mod runge_kutta;
mod util;

pub use acceleration::{
    AccelVecMeta, CentralAccelMeta, accel_grad, central_accel, central_accel_grad, vec_accel,
};
pub use kepler::{
    PARABOLIC_ECC_LIMIT, analytic_2_body, analytic_2_body_stm, compute_eccentric_anomaly,
    compute_true_anomaly, eccentric_anomaly_from_true, light_time_correct, moid,
    propagate_two_body,
};
pub use nongrav::NonGravModel;
pub use picard::{
PC15, PC25, PicardIntegrator, PicardStep, PicardStepSecondOrder, dumb_picard_init,
dumb_picard_init_second_order,
};
pub use radau::RadauIntegrator;
pub use runge_kutta::RK45Integrator;
