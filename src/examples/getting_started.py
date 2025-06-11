"""
Getting Started
===============

kete has a few foundational concepts which are important to understand. The most
important concepts are related to geometry:

- Frames - Reference frame, such as Ecliptic, Equatorial, Galactic.
- Vectors - Cartesian vectors with an associated coordinate frame.
- States - The name, position, velocity, and epoch for a specific object.
- Field of View (FOV) - A patch of sky along with the state of the observer.
- Cometary Elements - Orbital elements of an object.

Units of position are in AU and velocity is AU / Day.
"""

# %%
# Basic Geometry
# --------------

import matplotlib.pyplot as plt
import numpy as np

import kete

# Lets construct an object circular orbit
# Recall that an object in circular orbit has a speed of sqrt(GM / r) where r is the
# radius of the orbit.
r = 1
epoch = kete.Time.j2000()
speed = np.sqrt(kete.constants.SUN_GM / r)
pos = kete.Vector([0, r, 0], frame=kete.Frames.Equatorial)
vel = kete.Vector([-speed, 0, 0], frame=kete.Frames.Equatorial)

# We now make a State, representing the full information for an object in orbit of the
# sun.
state = kete.State("Circular", epoch, pos, vel)
state

# %%
# Vectors have convenient conversions to and from common reference frames:

pos.as_equatorial, pos.as_ecliptic

# %%
# Creation of vectors also have a few convenient helper functions:

# On sky position of the andromeda galaxy
ra = kete.conversion.ra_hms_to_degrees("0 42 44")
dec = kete.conversion.dec_dms_to_degrees("+41 16 9")
vec = kete.Vector.from_ra_dec(ra, dec)

# Change to galactic frame, and print the elevation and azimuth
vec = vec.change_frame(kete.Frames.Galactic)
vec.el, vec.az

# %%
# Now that we have seen how to make a state, lets look at much easier ways to get this
# data so that it is not necessary to hand write the position and velocities.
#
# Getting states
# --------------
# There are multiple ways to retrieve state information:
#
# - Manual entry, either from orbit elements of direct values.
# - From Spice (local)
# - From JPL Horizons (internet)
# - From the Minor Planet Center (internet)
#
# We have just seen an example of the first way, lets explore the other ways.

# Grabbing the state of the Moon.
# kete includes spice kernels for all of the planets, the Moon, Pluto and 5 asteroids:
# Ceres, Pallas, Vesta, Hygiea, Interamnia
jd = kete.Time.from_ymd(2020, 1, 1).jd
moon = kete.spice.get_state("Moon", jd)

# Grabbing a state from JPL Horizons
# Note that the epoch of this state is the epoch of fit from Horizons.
eros = kete.HorizonsProperties.fetch("Eros").state

# Grabbing a state from the MPC
# This is more expensive, as the MPC lookup involves downloading a full copy of the MPC
# database. This database contains more than a million asteroids. This gets returned as
# a pandas dataframe.
orb_file = kete.mpc.fetch_known_orbit_data()
subset = orb_file[orb_file["name"] == "Eros"]

# Note that in this case the tool converts all objects in the provided table into states
# which means it will always return a list of states, so below we get a list containing
# a single state.
states = kete.conversion.table_to_states(subset)
states

# %%
# Moving states
# -------------
#
# The state(s) which are constructed above usually do not have an epoch which is useful.
# Luckily, kete has tools for propagating the states forward and backward in time to
# whichever epoch is desired. There are 2 workhorse functions related to orbit
# propagation:
#
# - `propagate_n_body`
#       Highest accuracy, propagate a list of States to the desired Time using full
#       orbital mechanics including all planets, and optionally the 5 largest asteroids.
#       This includes effects due to General Relativity, and the non-spherical J2 term
#       of Jupiter, the Sun, and Earth. Optionally supports non-gravitational forces
#       such as Solar Radiation Pressure (SRP) and Poynting-Robertson.
# - `propagate_two_body`
#       Lower accuracy but faster, propagate the orbit using Keplerian mechanics, only
#       including the force due to the Sun.
#
# Both of these functions work on many states at once, and are designed to propagate
# the entire MPC catalog simultaneously.

# Lets grab the state of a few asteroids:
names = ["Eros", "Bennu", "Ryugu"]

orb_file = kete.mpc.fetch_known_orbit_data()
subset = orb_file[[name in names for name in orb_file["name"]]]

# This now contains 3 states:
states = kete.conversion.table_to_states(subset)

# Lets propagate these states to the current epoch
now = kete.Time.now().jd

states_now_exact = kete.propagate_n_body(states, now)
states_now_kepler = kete.propagate_two_body(states, now)
states_now_exact

# %%
# Field of Views
# --------------
# Knowing the state of an object in the solar system is interesting by itself, but
# unfortunately not directly useful when actually observing. To start with, we have to
# define the location of the observer. For ground observers, there are 2 simple ways to
# retrieve this position:
#
# - Using an MPC Observatory code
# - Using Earth based latitude/longitude and altitude as one would find on a map.

now = kete.Time.now().jd

# lookup the observers position in the solar system from a given observatory code/name.
observer_a = kete.spice.mpc_code_to_ecliptic("Palomar Mountain", now)

# lookup the observer position from an earth location, this is IPAC, at 100m altitude.
observer_b = kete.spice.earth_pos_to_ecliptic(now, 33.133278, 241.87206, 0.1, "IPAC")
observer_b

# %%
# Now that we have the observer location, we can define a Field of View for that
# observer. There are many different possible FOVs, the conceptually simplest one is an
# omni-directional field of view:

fov = kete.OmniDirectionalFOV(observer_a)
fov

# %%
# We can now compute the observed position of objects in this FOV:

state = kete.HorizonsProperties.fetch("Eros").state

# This accepts many states and many FOVs at once, it is the most efficient when all the
# requested states and FOVs are provided at the same time.
visible = kete.fov_state_check([state], [fov])[0]
visible

# %%
# The returned object requires some discussion, as it has not been covered yet.
# `fov_state_check` returns a `SimultaneousState`, which is a collection of `States`
# which are at the same epoch, along with an optional FOV. If the FOV is present, it is
# assumed that the states are the location as seen from the observers location along
# with the effects of light delay.
#
# If the FOV is provided, as it would be after the `fov_state_check` call will do, then
# there is a function on the `SimultaneousState` which provides the vectors from the
# observer to the objects.

obs_vecs = visible.obs_vecs[0]
obs_vecs = obs_vecs.as_equatorial
obs_vecs.ra, obs_vecs.dec

# %%
# Plotting Planets
# ----------------
#
# Here is a basic example of using kete to plot an orbit.

plt.figure(dpi=150)
jd = kete.Time.now().jd
plt.gca().set_aspect("equal", "box")

# plot the planets orbits
for i, planet in enumerate(["Mercury", "Venus", "Earth", "Mars", "Jupiter"]):
    plan = kete.spice.get_state(planet, jd)
    plt.scatter(plan.pos.x, plan.pos.y, color=f"C{i}", s=10)
    jds = np.linspace(jd - plan.elements.orbital_period, jd, 100)
    pos = np.array([kete.spice.get_state(planet, jd).pos for jd in jds]).T
    plt.plot(pos[0], pos[1], color="black", alpha=0.2)

# plot the orbit of eros, using 2 body mechanics to plot the previous orbit
eros = kete.HorizonsProperties.fetch("Eros").state
eros = kete.propagate_two_body(eros, jd)
plt.scatter(eros.pos.x, eros.pos.y, c="black", s=10)
jds = np.linspace(jd - eros.elements.orbital_period, jd, 100)
pos = np.array([kete.propagate_two_body(eros, jd).pos for jd in jds]).T
plt.plot(pos[0], pos[1], c="C0", alpha=0.2)

plt.xlabel("Ecliptic X (au)")
plt.ylabel("Ecliptic Y (au)")
plt.scatter(0, 0, color="red")
plt.tight_layout()
plt.show()
