//! Python support for time conversions.

use kete_core::{
    errors::Error,
    time::{TAI, TDB, Time, UTC},
};
use pyo3::prelude::*;

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
/// This representation and conversion tools make some small tradeoff for performance
/// vs accuracy. Conversion between time scales is only accurate on the millisecond
/// scale, however internal representation accuracy is on the microsecond scale.
///
/// TDB is treated as equivalent to TT and TCB, because these times only differ by less
/// than milliseconds per century.
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

impl<'py> FromPyObject<'py> for PyTime {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(jd) = ob.extract::<f64>() {
            return Ok(PyTime(Time::new(jd)));
        }
        Ok(PyTime(ob.downcast_exact::<PyTime>()?.get().0))
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
    pub fn new(jd: f64, scaling: &str) -> PyResult<Self> {
        Ok(match scaling.to_ascii_lowercase().as_str() {
            "tt" => PyTime(Time::<TDB>::new(jd)),
            "tdb" => PyTime(Time::<TDB>::new(jd)),
            "tcb" => PyTime(Time::<TDB>::new(jd)),
            "tai" => PyTime(Time::<TAI>::new(jd).tdb()),
            "utc" => PyTime(Time::<UTC>::new(jd).tdb()),
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
    ///     Accepts 'tdb', 'tai', 'utc', and 'tt', but they are converted to TDB
    ///     immediately.
    #[staticmethod]
    #[pyo3(signature = (mjd, scaling="tdb"))]
    pub fn from_mjd(mjd: f64, scaling: &str) -> PyResult<Self> {
        let scaling = scaling.to_lowercase();

        Ok(match scaling.as_str() {
            "tt" => PyTime(Time::<TDB>::from_mjd(mjd)),
            "tdb" => PyTime(Time::<TDB>::from_mjd(mjd)),
            "tai" => PyTime(Time::<TAI>::from_mjd(mjd).tdb()),
            "utc" => PyTime(Time::<UTC>::from_mjd(mjd).tdb()),
            s => Err(Error::ValueError(format!(
                "Scaling of type ({s}) is not supported, must be one of: 'tt', 'tdb', 'tai', 'utc'",
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
        if !s.contains('+') {
            if let Ok(t) = Time::<UTC>::from_iso(&(s.to_owned() + "+00:00")) {
                return Ok(PyTime(t.tdb()));
            }
        }
        Ok(PyTime(Time::<UTC>::from_iso(s)?.tdb()))
    }

    /// Create time object from the Year, Month, and Day.
    ///
    /// These times are assumed to be in UTC amd conversion is performed automatically.
    ///
    /// Parameters
    /// ----------
    /// year:
    ///     The Year, for example `2020`
    /// month:
    ///     The Month as an integer, 0 = January etc.
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
        PyTime(Time::<UTC>::now().unwrap().tdb())
    }

    /// Return (year, month, day), where day is a float.
    ///
    /// >>> kete.Time.from_ymd(2010, 1, 1).ymd
    /// (2010, 1, 1.0)
    #[getter]
    pub fn ymd(&self) -> (i64, u32, f64) {
        let (y, m, d, f) = self.0.utc().year_month_day();
        (y, m, d as f64 + f)
    }

    /// Julian Date in TDB scaled time.
    /// The difference between TT and TDB is never more than a few milliseconds
    /// per century, so these are treated as equivalent.
    #[getter]
    pub fn jd(&self) -> f64 {
        self.0.jd
    }

    /// Modified Julian Date in TDB scaled time.
    /// The difference between TT and TDB is never more than a few milliseconds
    /// per century, so these are treated as equivalent.
    #[getter]
    pub fn mjd(&self) -> f64 {
        self.0.mjd()
    }

    /// Julian Date in UTC scaled time.
    #[getter]
    pub fn utc_jd(&self) -> f64 {
        self.0.utc().jd
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
    pub fn year_float(&self) -> f64 {
        self.0.utc().year_as_float()
    }

    fn __add__(&self, other: PyTime) -> Self {
        (self.0.jd + other.0.jd).into()
    }

    fn __sub__(&self, other: PyTime) -> Self {
        (self.0.jd - other.0.jd).into()
    }

    fn __rsub__(&self, other: PyTime) -> Self {
        (other.0.jd - self.0.jd).into()
    }

    fn __repr__(&self) -> String {
        format!("Time({})", self.0.jd)
    }
}
