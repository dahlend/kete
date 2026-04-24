//! Python bindings for the FCCT14 / EFCC18 star catalog debiasing tables.

use kete_fitting::{DebiasTable, DebiasVersion};
use pyo3::prelude::*;

/// Star-catalog astrometric bias lookup table (FCCT14 / EFCC18).
///
/// Provides per-tile, per-catalog (RA, Dec) corrections to align reported
/// optical astrometry to the reference catalog (Gaia-DR2 for EFCC18,
/// PPMXL subset for FCCT14).  The corrections returned here are the
/// values to **subtract** from the reported (RA, Dec) to align with the
/// reference frame.
///
/// Tables are typically loaded once per process via
/// :py:func:`kete.mpc._fetch_debias_table` and reused across all
/// observations.
#[pyclass(frozen, module = "kete._core", name = "DebiasTable")]
#[derive(Debug)]
pub struct PyDebiasTable {
    /// The underlying core table.
    pub inner: DebiasTable,
}

#[pymethods]
impl PyDebiasTable {
    /// Parse a table from the text content of a ``bias.dat`` file.
    ///
    /// Parameters
    /// ----------
    /// text : str
    ///     Full text of the JPL ``bias.dat`` file.
    #[staticmethod]
    fn from_ascii(text: &str) -> PyResult<Self> {
        let inner = DebiasTable::from_ascii(text)?;
        Ok(Self { inner })
    }

    /// Catalog codes (single ASCII characters) supported by this table.
    #[getter]
    fn catalogs(&self) -> String {
        self.inner.catalogs().iter().map(|&b| b as char).collect()
    }

    /// Release identifier: ``2014`` for FCCT14, ``2018`` for EFCC18.
    #[getter]
    fn version(&self) -> u32 {
        match self.inner.version() {
            DebiasVersion::Fcct14 => 2014,
            DebiasVersion::Efcc18 => 2018,
        }
    }

    /// Look up the bias correction for a single observation.
    ///
    /// Parameters
    /// ----------
    /// catalog : str
    ///     Single-character MPC catalog code (e.g. ``"V"`` for Gaia-DR2).
    ///     Empty or multi-character strings return ``None``.
    /// ra : float
    ///     Right ascension in degrees.
    /// dec : float
    ///     Declination in degrees.
    /// jd : float
    ///     Observation epoch (Julian days).
    ///
    /// Returns
    /// -------
    /// tuple[float, float] or None
    ///     ``(d_ra, d_dec)`` correction in arcseconds, or ``None`` if the
    ///     catalog code is not present in the table.  The ``cos(dec)``
    ///     factor has already been divided out of ``d_ra``, so both values
    ///     are direct RA/Dec coordinate corrections: subtract them from
    ///     the reported (RA, Dec) (after dividing by 3600 for degrees).
    fn lookup(&self, catalog: &str, ra: f64, dec: f64, jd: f64) -> Option<(f64, f64)> {
        let code = *catalog.as_bytes().first()?;
        let (d_ra, d_dec) = self
            .inner
            .lookup(code, ra.to_radians(), dec.to_radians(), jd)?;
        let arcsec = 180.0 * 3600.0 / std::f64::consts::PI;
        // The table stores RA bias with the cos(dec) factor included.
        // Divide it out here so the caller receives a plain RA coordinate shift.
        let cos_dec = dec.to_radians().cos();
        let d_ra_coord = if cos_dec.abs() > 1e-12 {
            d_ra / cos_dec
        } else {
            0.0
        };
        Some((d_ra_coord * arcsec, d_dec * arcsec))
    }

    fn __repr__(&self) -> String {
        let cats: String = self.inner.catalogs().iter().map(|&b| b as char).collect();
        format!(
            "DebiasTable(version={}, catalogs='{}')",
            match self.inner.version() {
                DebiasVersion::Fcct14 => 2014,
                DebiasVersion::Efcc18 => 2018,
            },
            cats
        )
    }
}
