// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-FileCopyrightText: 2025 California Institute of Technology
// SPDX-License-Identifier: BSD-3-Clause

//! Python support for time conversions.

use kete_core::{
    errors::Error,
    time::{TAI, TCB, TDB, TT, Time, UTC},
};
use pyo3::{IntoPyObjectExt, prelude::*};

/// A representation of time, always in JD with TDB scaling.
///
/// Note that TDB is not the same as UTC, there is often about 60 seconds or more
/// offset between these time formats. This class enables fast conversion to and from
/// UTC however, via the :py:meth:`~Time.from_mjd`, and :py:meth:`~Time.from_iso`.
/// UTC can be recovered from this object through :py:meth:`~Time.utc_mjd`,
/// :py:meth:`~Time.utc_jd`, or :py:meth:`~Time.iso`.
///
/// Future UTC Leap seconds cannot be predicted, as a result of this, UTC becomes a
/// bit fuzzy when attempting to represent future times. All conversion of future times
/// therefore ignores the possibility of leap seconds.
///
/// Times are held to about 10 picoseconds at any epoch. Reading one back as a single
/// float Julian date, from :py:attr:`~Time.jd`, rounds it to about 40 microseconds
/// near the present.
///
/// TT is converted to TDB with the periodic TDB - TT term, under 2 milliseconds. TCB
/// is converted via a linear secular drift of L_B = 1.550519768e-8 relative to TDB
/// (IAU 2006).
///
/// Adding or subtracting a number gives a new Time that many TDB days later or
/// earlier. Subtracting one Time from another gives the TDB days between them.
///
/// Parameters
/// ----------
/// jd:
///     Julian Date in days.
/// scaling:
///     Accepts 'tdb', 'tai', 'utc', 'tcb', and 'tt', but they are converted to TDB
///     immediately. Defaults to 'tdb'
#[pyclass(frozen, module = "kete", name = "Time")]
#[derive(Debug)]
pub struct PyTime(pub Time<TDB>);

impl<'a, 'py> FromPyObject<'a, 'py> for PyTime {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(jd) = ob.extract::<f64>() {
            return Ok(PyTime(Time::new(jd)));
        }
        Ok(PyTime(ob.cast_exact::<PyTime>()?.get().0))
    }
}

impl From<f64> for PyTime {
    fn from(value: f64) -> Self {
        PyTime(Time::new(value))
    }
}

impl From<Time<TDB>> for PyTime {
    fn from(value: Time<TDB>) -> Self {
        PyTime(value)
    }
}

impl From<PyTime> for Time<TDB> {
    fn from(value: PyTime) -> Self {
        value.0
    }
}

#[pymethods]
impl PyTime {
    /// Construct a new time object, TDB default.
    #[new]
    #[pyo3(signature = (jd, scaling="tdb"))]
    pub fn new(jd: PyTime, scaling: &str) -> PyResult<Self> {
        // Keep both parts of the Julian date, so a Time passed in keeps its precision.
        let (days, frac) = jd.0.jd_parts();
        Ok(match scaling.to_ascii_lowercase().as_str() {
            "tdb" => jd,
            "tt" => PyTime(Time::<TT>::from_parts(days, frac).tdb()),
            "tcb" => PyTime(Time::<TCB>::from_parts(days, frac).tdb()),
            "tai" => PyTime(Time::<TAI>::from_parts(days, frac).tdb()),
            "utc" => PyTime(Time::<UTC>::from_parts(days, frac).tdb()),
            s => Err(Error::ValueError(format!(
                "Scaling of type ({s}) is not supported, must be one of: 'tt', 'tdb', 'tcb', 'tai', 'utc'",
            )))?,
        })
    }

    /// Time from a modified julian date.
    ///
    /// Parameters
    /// ----------
    /// mjd:
    ///     Modified Julian Date in days.
    /// scaling:
    ///     Accepts 'tdb', 'tai', 'utc', 'tcb', and 'tt', but they are converted to TDB
    ///     immediately.
    #[staticmethod]
    #[pyo3(signature = (mjd, scaling="tdb"))]
    pub fn from_mjd(mjd: f64, scaling: &str) -> PyResult<Self> {
        let scaling = scaling.to_lowercase();

        Ok(match scaling.as_str() {
            "tt" => PyTime(Time::<TT>::from_mjd(mjd).tdb()),
            "tdb" => PyTime(Time::<TDB>::from_mjd(mjd)),
            "tcb" => PyTime(Time::<TCB>::from_mjd(mjd).tdb()),
            "tai" => PyTime(Time::<TAI>::from_mjd(mjd).tdb()),
            "utc" => PyTime(Time::<UTC>::from_mjd(mjd).tdb()),
            s => Err(Error::ValueError(format!(
                "Scaling of type ({s}) is not supported, must be one of: 'tt', 'tdb', 'tcb', 'tai', 'utc'",
            )))?,
        })
    }

    /// Time from an ISO formatted string.
    ///
    /// ISO formatted strings are assumed to be in UTC time scaling.
    ///
    /// This only supports RFC3339 - a strict subset of the ISO format which removes
    /// all ambiguity for the definition of time. There are many examples where the
    /// ISO standard does not have enough information to uniquely specify the exact
    /// time.
    ///
    /// The most common issue is failing to provide a timezone offset value. Typically
    /// these are numbers at the end of the UTC ISO string "+00:00". This function will
    /// check for that and add it if not found.
    ///
    /// Parameters
    /// ----------
    /// s:
    ///     ISO Formatted String.
    #[staticmethod]
    pub fn from_iso(s: &str) -> PyResult<Self> {
        // attempt to make life easier for the user by checking if they are missing
        // the timezone information. If they are, append it and return. Otherwise
        // let the conversion fail as it normally would.
        if !s.contains('+')
            && let Ok(t) = Time::<UTC>::from_iso(&(s.to_owned() + "+00:00"))
        {
            return Ok(PyTime(t.tdb()));
        }
        Ok(PyTime(Time::<UTC>::from_iso(s)?.tdb()))
    }

    /// Create time object from the Year, Month, and Day.
    ///
    /// These times are assumed to be in UTC and conversion is performed automatically.
    ///
    /// Parameters
    /// ----------
    /// year:
    ///     The Year, for example `2020`
    /// month:
    ///     The Month as an integer, 1 = January etc.
    /// day:
    ///     The day as an integer or float.
    #[staticmethod]
    pub fn from_ymd(year: i64, month: u32, day: f64) -> Self {
        let frac_day = day.rem_euclid(1.0);
        let day = day.div_euclid(1.0) as u32;
        PyTime(Time::<UTC>::from_year_month_day(year, month, day, frac_day).tdb())
    }

    /// Time in the current time.
    #[staticmethod]
    pub fn now() -> Self {
        PyTime(Time::<UTC>::now().tdb())
    }

    /// Return (year, month, day), where day is a float.
    ///
    /// >>> kete.Time.from_ymd(2010, 1, 1).ymd
    /// (2010, 1, 1.0)
    #[getter]
    pub fn ymd(&self) -> (i32, u32, f64) {
        let (y, m, d, f) = self.0.utc().year_month_day();
        (y, m, d as f64 + f)
    }

    /// Julian Date in TDB scaled time.
    #[getter]
    pub fn jd(&self) -> f64 {
        self.0.jd()
    }

    /// Modified Julian Date in TDB scaled time.
    #[getter]
    pub fn mjd(&self) -> f64 {
        self.0.mjd()
    }

    /// Julian Date in UTC scaled time.
    #[getter]
    pub fn utc_jd(&self) -> f64 {
        self.0.utc().jd()
    }

    /// Modified Julian Date in UTC scaled time.
    #[getter]
    pub fn utc_mjd(&self) -> f64 {
        self.0.utc().mjd()
    }

    /// Time in the UTC ISO time format.
    #[getter]
    pub fn iso(&self) -> PyResult<String> {
        Ok(self.0.utc().to_iso()?)
    }

    /// J2000 epoch time.
    #[staticmethod]
    pub fn j2000() -> Self {
        PyTime(Time::<TDB>::new(2451545.0))
    }

    /// Time as the UTC year in float form.
    ///
    /// Note that Time is TDB Scaled, causing UTC to be a few seconds different.
    ///
    /// >>> kete.Time.from_ymd(2010, 1, 1).year_float
    /// 2010.0
    ///
    /// >>> kete.Time(2457754.5, scaling='utc').year_float
    /// 2017.0
    ///
    /// 2016 was a leap year, so 366 days instead of 365.
    ///
    /// >>> kete.Time(2457754.5 - 366, scaling='utc').year_float
    /// 2016.0
    ///
    #[getter]
    pub fn year_float(&self) -> PyResult<f64> {
        Ok(self.0.utc().year_as_float()?)
    }

    fn __add__(&self, days: f64) -> Self {
        PyTime(self.0 + days)
    }

    fn __radd__(&self, days: f64) -> Self {
        PyTime(self.0 + days)
    }

    /// A Time minus a Time is the TDB days between them; a Time minus a number of
    /// days is a Time.
    fn __sub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(time) = other.cast_exact::<PyTime>() {
            return (self.0 - time.get().0).elapsed.into_py_any(py);
        }
        PyTime(self.0 - other.extract::<f64>()?).into_py_any(py)
    }

    /// TDB days from this time to `jd`, a Julian date.
    fn __rsub__(&self, jd: f64) -> f64 {
        (Time::<TDB>::new(jd) - self.0).elapsed
    }

    fn __repr__(&self) -> String {
        format!("Time({})", self.0.jd())
    }

    /// Times compare by their Julian date as an f64, the value :attr:`jd` reports.
    fn __eq__(&self, other: PyTime) -> bool {
        self.0.jd() == other.0.jd()
    }

    fn __lt__(&self, other: PyTime) -> bool {
        self.0.jd() < other.0.jd()
    }
}
