"""
Time conversion support, primarily provided by the :py:class:`Time` class.
"""

from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo

from ._core import Time, next_solar_noon, next_sunset_sunrise

__all__ = ["Time", "next_sunset_sunrise", "next_solar_noon"]


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


Time.local_time = _local_time
Time.to_datetime = _to_datetime
