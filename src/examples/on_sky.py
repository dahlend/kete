"""
Orbit computation and on sky positions.
=======================================

Given a set of orbital elements, compute the on-sky position of an object.
"""

import kete

# %% Loading orbital elements
# Define an object through cometary orbital elements:
elements = kete.CometElements(
    desig="Ceres",
    epoch=2460200.5,
    eccentricity=0.0789,
    inclination=10.58688,
    lon_of_ascending=80.25498,
    peri_time=2459919.898,
    peri_arg=73.4218,
    peri_dist=2.5489,
)

# These can be converted to cartesian coordinates
state = elements.state

# States for objects may also be queried from JPL Horizons
# state = kete.HorizonsProperties.fetch("Ceres").state

print("State information from orbital elements:\n")
print(state)

# This state may be viewed in several frames
# These share the exact definitions used by SPICE
# state.as_equatorial, state.as_ecliptic, state.as_galactic, state.as_fk4

# %% Pick an observing time:
# By default, kete uses TDB scaled JD time, the Time object automatically handles
# conversions to and from this scaling:
jd = kete.Time.from_ymd(2025, 6, 1.5).jd

# Currently the state of the object above is not at this date
# lets perform orbit propagation to bring it to this date:
state = kete.propagate_n_body(state, jd)


# %% Observatory Information
# Lets try to make an observation of ceres from the ground
obs_info = kete.mpc.find_obs_code("Palomar Mountain")

# Location of palomar mountain in lat/lon/elevation
print(
    "\nInformation about Palomar, these values are with respect to "
    "WGS84 Earth Coordinates."
)
print("lat, lon, elevation, Obs code, and name")
print(obs_info)

# Load the position of the observer in the solar system:
observer = kete.spice.mpc_code_to_ecliptic("Palomar Mountain", jd)


# %% Propagation and Orbit calculations
# We can now compare the position of ceres with respect to the observatory:
observer_to_object = state.pos - observer.pos

print("\nVector from the observatory to the object (no light delay):")
print(observer_to_object)

# Unfortunately, light takes a measurable amount of time to move, so we need to add a
# correction for it. In this case we can just use the two-body approximation for motion.
# Note that in general the two-body approximation will result in incorrect
# position calculations when propagation time exceeds a few days.
# But its an excellent approximation over the minutes time scale.
light_delay_state = kete.propagate_two_body(state, state.jd, observer.pos)

print("\nThe light delayed position of the object vs the instantaneous:")
print(light_delay_state.pos)
print(state.pos)
# it is a small difference, but it is measurable.

# lets compute the vector from the observer to this light delayed state
# and convert it to the equatorial frame:
observer_to_object = (light_delay_state.pos - observer.pos).as_equatorial


# %% RA/Dec calculations
print(f"\nAt {kete.Time(jd).iso} UTC from the {obs_info[-1]} observatory")
print(
    f"{state.desig} is visible at:\n"
    f"{observer_to_object.ra_hms} RA  {observer_to_object.dec_dms} DEC"
)
