// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! SPICE text kernels: frames (FK), spacecraft clock (SCLK), text PCK,
//! instrument (IK) and meta-kernels.
//!
//! [`TextKernels`] holds the variables of all loaded text kernels in one
//! namespace. Thus a frames kernel can define a clock, and an SCLK kernel can
//! hold frame variables.
//!
//! # Layout
//!
//! - [`vars`] parses a text kernel into its variables.
//! - [`sclk`] builds the spacecraft clocks from the variables.
//! - [`fk`] builds the frame definitions of frames kernels.
//! - [`pck`] builds the body orientation models of text PCK kernels.
//! - [`ik`] reads the field of view definitions of instrument kernels.
//! - [`mk`] reads the list of kernels of a meta-kernel.
//!
//! [`TextKernels`] builds the clocks, frame definitions and body models when a
//! kernel loads, from all variables loaded so far.

pub mod fk;
pub mod ik;
pub mod mk;
pub mod pck;
pub mod sclk;
pub mod vars;

pub use vars::{TextKernelValue, TextKernelVars};

use crate::frames::{self, FrameDef};
use crossbeam::sync::ShardedLock;
use kete_core::errors::{Error, KeteResult};
use kete_core::frames::FrameId;
use pck::{BodyRotation, body_rotations};
use sclk::{ClockId, Sclk, clocks_from};
use std::collections::HashMap;

/// The loaded text kernels.
///
/// It holds their variables, and the spacecraft clocks, frame definitions and
/// body orientation models that the variables define.
#[derive(Debug, Default)]
pub struct TextKernels {
    vars: TextKernelVars,
    clocks: HashMap<ClockId, KeteResult<Sclk>>,
    frames: HashMap<FrameId, KeteResult<FrameDef>>,
    bodies: HashMap<i32, KeteResult<BodyRotation>>,
}

impl TextKernels {
    /// Load a text kernel, such as a frames or SCLK kernel.
    ///
    /// The variables of a meta-kernel load, except [`mk::META_KERNEL_VARS`].
    /// The kernels it lists do not; see [`crate::load_kernels`].
    ///
    /// Nothing from the file loads if this function fails. A malformed clock,
    /// frame definition or body model does not make it fail. The error comes
    /// when the clock, frame or body is used.
    ///
    /// # Errors
    /// [`Error::IOError`] if the file cannot be read or does not parse.
    pub fn load_file(&mut self, filename: &str) -> KeteResult<()> {
        let text = vars::read_kernel_text(filename)?;
        self.load_text(&text)
            .map_err(|e| crate::add_context(e, &format!("Text kernel {filename}")))
    }

    /// Load the text of a text kernel; see [`Self::load_file`].
    ///
    /// # Errors
    /// As [`Self::load_file`].
    pub fn load_text(&mut self, text: &str) -> KeteResult<()> {
        let mut vars = self.vars.clone();
        vars.load_text(text)?;
        for name in mk::META_KERNEL_VARS {
            let _ = vars.remove(name);
        }
        self.clocks = clocks_from(&vars);
        self.frames = fk::definitions(&vars);
        self.bodies = body_rotations(&vars);
        self.vars = vars;
        Ok(())
    }

    /// Remove all loaded text kernels.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// The variables of the loaded text kernels.
    #[must_use]
    pub fn vars(&self) -> &TextKernelVars {
        &self.vars
    }

    /// The spacecraft clock with NAIF ID `id`, such as -226.
    ///
    /// # Errors
    /// [`Error::ValueError`] if no loaded text kernel defines the clock, or its
    /// definition is incomplete or inconsistent.
    pub fn clock(&self, id: ClockId) -> KeteResult<&Sclk> {
        self.clocks
            .get(&id)
            .ok_or_else(|| {
                Error::ValueError(format!("SCLK clock for spacecraft ID {id} not found."))
            })?
            .as_ref()
            .map_err(Clone::clone)
    }

    /// The NAIF IDs of the loaded spacecraft clocks whose definitions are
    /// valid.
    #[must_use]
    pub fn clock_ids(&self) -> Vec<ClockId> {
        self.clocks
            .iter()
            .filter(|(_, clock)| clock.is_ok())
            .map(|(id, _)| *id)
            .collect()
    }

    /// The definition of frame `id`, or `None` if it is neither built in nor
    /// defined in a loaded frames kernel.
    ///
    /// A built-in frame takes precedence over a frames kernel definition.
    ///
    /// # Errors
    /// [`Error::ValueError`] if the frame is of a class kete does not support,
    /// or its definition is incomplete or malformed.
    pub fn frame_definition(&self, id: FrameId) -> KeteResult<Option<FrameDef>> {
        frames::built_in(&self.vars, id)
            .or_else(|| self.frames.get(&id).cloned())
            .transpose()
    }

    /// The ID of the frame called `name`; see [`frames::id_from_name`].
    ///
    /// # Errors
    /// [`Error::ValueError`] if no built-in frame or loaded frames kernel has
    /// the name.
    pub fn frame_id(&self, name: &str) -> KeteResult<FrameId> {
        frames::id_from_name(&self.vars, name)
    }

    /// The name of frame `id`, if it is built in or defined in a loaded frames
    /// kernel.
    #[must_use]
    pub fn frame_name(&self, id: FrameId) -> Option<String> {
        frames::frame_name(&self.vars, id)
    }

    /// The text PCK orientation model of body `id`.
    ///
    /// The model is `None` if no loaded text kernel gives the pole of the body.
    ///
    /// # Errors
    /// [`Error::ValueError`] if its constants are incomplete or malformed.
    pub fn body_rotation(&self, id: i32) -> KeteResult<Option<&BodyRotation>> {
        self.bodies
            .get(&id)
            .map(|model| model.as_ref().map_err(Clone::clone))
            .transpose()
    }

    /// The field of view of instrument `id`; see [`ik::instrument_fov`].
    ///
    /// # Errors
    /// [`Error::ValueError`] if the definition is missing or malformed.
    pub fn instrument_fov(&self, id: i32) -> KeteResult<ik::InstrumentFov> {
        ik::instrument_fov(&self.vars, id)
    }

    /// The NAIF ID that `NAIF_BODY_NAME` and `NAIF_BODY_CODE` give the body or
    /// instrument called `name`, or `None` if they do not name it.
    ///
    /// The name is matched in upper case, with each run of blanks as one space.
    /// When the variables give a name more than once, the last one is used.
    #[must_use]
    pub fn body_id(&self, name: &str) -> Option<i32> {
        let canonical = |name: &str| {
            name.split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
                .to_uppercase()
        };
        let target = canonical(name);
        let Some(TextKernelValue::Strings(names)) = self.vars.get("NAIF_BODY_NAME") else {
            return None;
        };
        let codes = self.vars.integers("NAIF_BODY_CODE").ok().flatten()?;
        names
            .iter()
            .zip(codes)
            .filter(|(name, _)| canonical(name) == target)
            .map(|(_, code)| code)
            .next_back()
    }

    /// The clock ID of the CK frame with class ID `ck_id`; see
    /// [`fk::ck_clock_id`].
    ///
    /// # Errors
    /// [`Error::ValueError`] if `CK_<ck_id>_SCLK` is not one integer.
    pub fn ck_clock_id(&self, ck_id: i32) -> KeteResult<ClockId> {
        fk::ck_clock_id(&self.vars, ck_id)
    }

    /// The SPK ID of the spacecraft of the CK frame with class ID `ck_id`; see
    /// [`fk::ck_spk_id`].
    ///
    /// # Errors
    /// [`Error::ValueError`] if `CK_<ck_id>_SPK` is not one integer, or it is
    /// not set and `ck_id` is above -1000.
    pub fn ck_spk_id(&self, ck_id: i32) -> KeteResult<i32> {
        fk::ck_spk_id(&self.vars, ck_id)
    }
}

/// Text kernel singleton.
///
/// A [`ShardedLock`] protects the [`TextKernels`]. Use `.try_read()` for
/// read-only access.
pub static LOADED_TEXT_KERNELS: std::sync::LazyLock<ShardedLock<TextKernels>> =
    std::sync::LazyLock::new(|| ShardedLock::new(TextKernels::default()));

#[cfg(test)]
mod tests {
    use super::*;

    /// A malformed clock does not stop the other definitions of its kernel
    /// from loading. Its error comes when the clock is used.
    #[test]
    fn malformed_clock_does_not_block_its_kernel() {
        let mut text = TextKernels::default();
        text.load_text(
            "\\begindata\nFRAME_GOOD_TK = 1400950\nFRAME_1400950_NAME = 'GOOD_TK'\n\
             FRAME_1400950_CLASS = 4\nFRAME_1400950_CLASS_ID = 1400950\n\
             TKFRAME_1400950_RELATIVE = 'J2000'\nTKFRAME_1400950_SPEC = 'MATRIX'\n\
             TKFRAME_1400950_MATRIX = ( 1 0 0 0 1 0 0 0 1 )\n\
             SCLK_DATA_TYPE_88 = ( 1 )\n",
        )
        .unwrap();
        assert!(text.frame_definition(FrameId(1_400_950)).unwrap().is_some());
        assert!(
            text.clock(ClockId(-88))
                .unwrap_err()
                .to_string()
                .contains("SCLK01_N_FIELDS_88")
        );
        assert!(text.clock_ids().is_empty());
    }
}
