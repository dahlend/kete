"""
Time conversion support, primarily provided by the :py:class:`Time` class.
"""

from __future__ import annotations

import calendar
import datetime
from zoneinfo import ZoneInfo

from ._core import Time

__all__ = ["Time", "d_h_m_s_to_float_days", "days_in_year"]


def _year_float(self) -> float:
    """
    Time as the UTC year in float form.

    Note that Time is TDB Scaled, causing UTC to be a few seconds different.

    >>> kete.Time.from_ymd(2010, 1, 1).year_float
    2010.0

    >>> kete.Time(2457754.5, scaling='utc').year_float
    2017.0

    2016 was a leap year, so 366 days instead of 365.

    >>> kete.Time(2457754.5 - 366, scaling='utc').year_float
    2016.0
    """
    tup = datetime.datetime.fromisoformat(self.iso).timetuple()
    frac = d_h_m_s_to_float_days(tup.tm_yday - 1, tup.tm_hour, tup.tm_min, tup.tm_sec)
    return tup.tm_year + frac / days_in_year(tup.tm_year)


def _local_time(self, timezone=None) -> str:
    """
    String representation of the Time in a localized time zone format.
    This will automatically take into account daylight savings time if necessary.

    Parameters
    ----------
    timezone:
        Optional, a ``datetime.timezone`` object which defines a time zone, if you
        are using python 3.9 or greater, then the ``zoneinfo`` package in base
        python maintains a list of pre-defined timezones:

            from zoneinfo import available_timezones, ZoneInfo
            available_timezones() # List all available timezones
            timezone = ZoneInfo("US/Pacific")
            kete.Time.j2000().local_time(timezone)

    """

    if timezone is None:
        timezone = datetime.datetime.now().astimezone().tzinfo
    t = datetime.datetime.fromisoformat(self.iso)
    return (t + timezone.utcoffset(t)).strftime("%Y-%m-%d %X.%f")


def _to_datetime(self) -> datetime.datetime:
    """
    Convert time to a Datetime object.
    """
    return datetime.datetime.fromisoformat(self.iso).replace(tzinfo=ZoneInfo("UTC"))


Time.year_float = property(
    fget=_year_float,
)

Time.local_time = _local_time
Time.to_datetime = _to_datetime


def d_h_m_s_to_float_days(
    days: float = 0, hours: float = 0, minutes: float = 0, seconds: float = 0
) -> float:
    """
    Convert floats of days, hours, minutes, and seconds into a single float in units of
    days.

    >>> kete.time.d_h_m_s_to_float_days(1, 11, 59, 60)
    1.5

    """
    dt = datetime.timedelta(
        days=days, hours=hours, minutes=minutes, seconds=seconds
    ).total_seconds()
    return dt / 60 / 60 / 24


def days_in_year(year: int) -> int:
    """
    Return the number of days in the specified year, specifically for leap years.

    >>> kete.time.days_in_year(2016)
    366

    >>> kete.time.days_in_year(2017)
    365

    Parameters
    ----------
    year :
        Year as an int.
    """
    return 365 + calendar.isleap(year)
