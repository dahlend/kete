"""
Plot states from the MPC
========================

This will open an interactive 3d plot of the solar system.

This plots all of the Hildas from the MPC, along with the last 90 days of their orbit.
"""

import kete
import matplotlib.pyplot as plt
import numpy as np

# Set the X/Y/Z scale
zoom = 4.0

# Fetch all known asteroid orbits
orb = kete.mpc.fetch_known_orbit_data()

# Subset to the Hildas
orb = orb[orb["group_name"] == "Hilda"]

# convert the table of data to State objects
states = kete.conversion.table_to_states(orb)

# Every object in the MPC orbit file has its own epoch of fit.
# This means that some objects epochs may be months or years away from
# others. In order to bring all of these objects to the same epoch, the
# function below will propagate the states to the first state's epoch.
states = kete.propagate_two_body(states, states[0].jd)

# Grab the positions from this subset
pos = np.array([p.pos for p in states])

plt.figure(dpi=250)
ax = plt.gcf().add_subplot(projection="3d")

# Plot the objects
ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=0.1, alpha=0.5)

# Plot the planets for 1 orbit each
for i, planet in enumerate(["Mercury", "Venus", "Earth", "Mars", "Jupiter"]):
    jd = states[0].jd
    plan = kete.spice.get_state(planet, jd)
    ax.scatter(plan.pos.x, plan.pos.y, color=f"C{i}", s=10)
    jds = np.linspace(jd - plan.elements.orbital_period, jd, 100)
    pos = np.array([kete.spice.get_state(planet, jd).pos for jd in jds]).T
    ax.plot(pos[0], pos[1], pos[2], color="black", alpha=0.2)

for state in states:
    jd = states[0].jd
    jds = np.linspace(jd - 90, jd, 100)
    pos = np.array([kete.propagate_two_body([state], jd)[0].pos for jd in jds]).T
    ax.plot(pos[0], pos[1], pos[2], color="black", alpha=0.1, lw=0.2)

ax.scatter(0, 0, 0, color="red")
ax.set_xticks([-zoom, 0, zoom])
ax.set_yticks([-zoom, 0, zoom])
ax.set_zticks([-zoom, 0, zoom])
ax.set_xlim(-zoom, zoom)
ax.set_ylim(-zoom, zoom)
ax.set_zlim(-zoom, zoom)
plt.tight_layout()
plt.show()
