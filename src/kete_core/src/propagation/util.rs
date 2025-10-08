use nalgebra::{OVector, SVector};

use crate::errors::KeteResult;

/// Function will be of the form y' = F(time, y, metadata, bool)
/// This is the first-order general IVP solver.
pub(crate) type FirstOrderODE<'a, MType, const DIM: usize> =
    &'a dyn Fn(f64, &SVector<f64, DIM>, &mut MType, bool) -> KeteResult<SVector<f64, DIM>>;

/// Function will be of the form y'' = F(time, y, y', metadata, bool)
/// This is the input for a second-order general IVP solver.
pub(crate) type SecondOrderODE<'a, MType, D> = &'a dyn Fn(
    f64,
    &OVector<f64, D>,
    &OVector<f64, D>,
    &mut MType,
    bool,
) -> KeteResult<OVector<f64, D>>;
