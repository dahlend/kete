# SPDX-FileCopyrightText: 2025 California Institute of Technology
# SPDX-License-Identifier: BSD-3-Clause

"""
Physical Constants
"""

K_SUN = 0.01720209894846
"""AU^(3/2) per (Day sqrt(Solar Mass))"""

SPEED_OF_LIGHT_AUDAY = 173.14463267424034
"""AU per day"""

SPEED_OF_LIGHT = 299_792_458.0
"""Meters per second"""

PLANCK = 6.62618e-34
"""J * s"""

BOLTZMANN = 1.381e-23
"""J / K"""

STEFAN_BOLTZMANN = 5.670374419e-8
"""J per K**4 m**2 s"""

SUN_TEMP = 5778
"""Sun's effective black body temp in Kelvin."""

SUN_GM = 0.00029591220828411956
"""Standard gravitational parameter, from DE440. au^3 / day^2"""

AU_M = 149_597_870_700
"""AU in meters by definition."""

AU_KM = AU_M / 1000
"""AU in km."""

EARTH_MAJOR_AXIS_M = 6_378_137.0
"""Earth semi major axis in meters."""

EARTH_MINOR_AXIS_M = 6_356_752.3
"""Earth minor axis in meters."""

EARTH_RADIUS_M = EARTH_MAJOR_AXIS_M
"""Earth radius in meters."""

EARTH_RADIUS_AU = EARTH_RADIUS_M / AU_M
"""Earth radius in AU."""

EARTH_MOON_DIST_KM = 385_000
"""Approximate average distance from the Earth to the Moon in km."""

MOON_RADIUS_AU = 1.16138016662292e-05
"""Radius of the Moon in au."""

SUN_RADIUS_M = 1.392e9 / 2.0
"""Sun radius in meters."""

J2000 = 2451545.0
"""J2000 JD."""

SOLAR_FLUX = 1.3608e3
"""W/m^2 measured at 1 AU."""

C_V = 1329.0
"""Relationship constant for conversion between D, H_V, and p_V"""
