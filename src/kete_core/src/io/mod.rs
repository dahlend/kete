// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! File IO related tools

pub mod binary;
pub mod bytes;
#[cfg(feature = "polars")]
pub mod parquet;
