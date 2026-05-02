"""
Orbiting a Contact Binary: Multi-Orbit Trajectory
==================================================

Demonstrates :mod:`kete.extended_body` with a two-component (contact
binary) body.  The body is represented by two icosahedral lobes of
different radii joined along the x-axis; the combined centre of mass
is at the origin.  The tidal integrator is anchored to Ceres' shipped
SPK ephemeris (NAIF id 20000001) purely to provide a realistic
heliocentric position; the shape model has nothing to do with Ceres.

A test particle is started in a low circular orbit (800 km from the
COM — only ~1.6x the bounding radius of the larger lobe) and
propagated for five full orbital periods.  The asymmetric bimodal
gravity field causes measurable radial oscillations even at this
altitude.

Replace the icosahedral mesh with a published shape model for real
work.
"""

import math

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

import kete
from kete.extended_body import (
    ExtendedBody,
    Polyhedron,
    RotationModel,
    propagate_near_body,
)

AU_KM = kete.constants.AU_KM


# %%
# Helper: icosahedron mesh of given radius, optionally offset along x, y, z.
def _icosphere(radius_km, cx_km=0.0, cy_km=0.0, cz_km=0.0):
    r = radius_km / AU_KM
    cx = cx_km / AU_KM
    cy = cy_km / AU_KM
    cz = cz_km / AU_KM
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    raw = [
        (-1, phi, 0),
        (1, phi, 0),
        (-1, -phi, 0),
        (1, -phi, 0),
        (0, -1, phi),
        (0, 1, phi),
        (0, -1, -phi),
        (0, 1, -phi),
        (phi, 0, -1),
        (phi, 0, 1),
        (-phi, 0, -1),
        (-phi, 0, 1),
    ]
    s = r / math.sqrt(1.0 + phi * phi)
    verts = [(cx + c[0] * s, cy + c[1] * s, cz + c[2] * s) for c in raw]
    faces = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]
    return verts, faces


# %%
# Contact-binary geometry.
#
# Lobe A (large): radius 250 km.  Lobe B (small): radius 150 km.
# The two lobes touch (distance between centres = 250 + 150 = 400 km).
# For equal density, mass is proportional to radius^3, so the centre
# of mass (chosen as the origin) lies at:
#
#   x_com = (m_A * x_A + m_B * x_B) / (m_A + m_B)  = 0
#   => x_A = -r_A^3 / (r_A^3 + r_B^3) * sep_km
#   => x_B = +r_B^3 / ... * ... (complementary)
r_a_km = 250.0  # large lobe radius
r_b_km = 150.0  # small lobe radius
sep_km = r_a_km + r_b_km  # touching: 400 km
m_a = r_a_km**3
m_b = r_b_km**3
cx_a_km = -m_b / (m_a + m_b) * sep_km  # negative: large lobe is -x
cx_b_km = +m_a / (m_a + m_b) * sep_km  # positive: small lobe is +x

# Total GM split in proportion to volume (equal density).
total_gm_km3_s2 = 62.6  # same scale as Ceres for a realistic orbit
total_gm = total_gm_km3_s2 * (86400.0**2) / (AU_KM**3)
gm_a = total_gm * m_a / (m_a + m_b)
gm_b = total_gm * m_b / (m_a + m_b)

verts_a, faces_a = _icosphere(r_a_km, cx_a_km)
verts_b, faces_b = _icosphere(r_b_km, cx_b_km, cz_km=20.0)
poly_a = Polyhedron(verts_a, faces_a, gm=gm_a)
poly_b = Polyhedron(verts_b, faces_b, gm=gm_b)

# Bounding radius of the whole body (centre of mass to far tip of lobe A).
bounding_km = abs(cx_a_km) + r_a_km  # ~ 311 km

body = ExtendedBody(
    components=[poly_a, poly_b],
    rotation=RotationModel.fixed(),
    length_au=bounding_km / AU_KM,
    proximity_radius_au=15.0 * bounding_km / AU_KM,
)

# %%
# Orbital parameters.  The particle starts 800 km from the COM along
# +y (perpendicular to the body long axis), with the circular-orbit
# speed in the -x direction.  This is the equatorial plane viewed
# edge-on to the binary's long axis — the orbit crosses both lobe
# gravity wells each revolution.
jd0 = kete.Time(2_460_000.5)
orbit_r_km = 2000.0
orbit_r_au = orbit_r_km / AU_KM
v_circ_au_day = math.sqrt(total_gm / orbit_r_au) * 1.2
t_orbit_days = 2.0 * math.pi * math.sqrt(orbit_r_au**3 / total_gm)

particle0 = kete.State(
    desig="probe",
    jd=jd0,
    pos=[0.0, orbit_r_au, 0.0],
    vel=[-v_circ_au_day, 0.0, 0.0],
    center_id=20_000_001,
)

# %%
# Propagate by chaining: each call advances the *previous* output by
# one step, so the integrator always works on a single short arc rather
# than accumulating from the initial epoch.  This keeps run time O(n)
# in the number of steps regardless of how many orbits are requested.
n_orbits = 500
steps_per_orbit = 100
n_steps = n_orbits * steps_per_orbit

step_days = t_orbit_days / steps_per_orbit

xs_km = [0.0]
ys_km = [orbit_r_km]
ts_orbit = [0.0]

current = particle0
for i in range(1, n_steps + 1):
    jdf = kete.Time(current.jd + step_days)
    try:
        current = propagate_near_body(
            body=body,
            body_naif_id=20_000_001,
            particle=current,
            jd_final=jdf,
        )
        xs_km.append(current.pos[0] * AU_KM)
        ys_km.append(current.pos[1] * AU_KM)
        ts_orbit.append(i / steps_per_orbit)
    except Exception as e:
        break
xs_km = np.asarray(xs_km)
ys_km = np.asarray(ys_km)
ts_orbit = np.asarray(ts_orbit)

# %%
# Reference circle (pure Keplerian, no body gravity perturbation).
theta = np.linspace(0.0, 2.0 * math.pi, 500)
ref_x = orbit_r_km * np.cos(theta)
ref_y = orbit_r_km * np.sin(theta)

# %%
# Plot.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: x-y trajectory coloured by orbit number, with lobe outlines.
ax = axes[0]
sc = ax.scatter(xs_km, ys_km, c=ts_orbit, cmap="viridis", s=6, zorder=4)
ax.plot(ref_x, ref_y, "k--", linewidth=0.8, alpha=0.5, label="Keplerian circle")

# Draw lobe cross-sections in the x-y plane.
for cx, r, _label, col in [
    (cx_a_km, r_a_km, "Lobe A", "peru"),
    (cx_b_km, r_b_km, "Lobe B", "tan"),
]:
    circle = mpatches.Circle((cx, 0), r, color=col, alpha=0.35, zorder=2)
    ax.add_patch(circle)
    ax.plot(cx, 0, ".", color=col, markersize=5, zorder=3)

ax.plot(
    0,
    0,
    "+",
    color="k",
    markersize=8,
    markeredgewidth=1.5,
    label="Centre of mass",
    zorder=5,
)
ax.set_aspect("equal")
ax.set_xlabel("x (km, body-centric)")
ax.set_ylabel("y (km, body-centric)")
ax.set_title("Orbit trajectory — body-centric frame")
ax.legend(
    fontsize=8,
    handles=[
        mpatches.Patch(color="peru", alpha=0.5, label="Lobe A (large)"),
        mpatches.Patch(color="tan", alpha=0.5, label="Lobe B (small)"),
        plt.Line2D([0], [0], color="k", linestyle="--", label="Keplerian circle"),
        plt.Line2D(
            [0],
            [0],
            marker="+",
            color="k",
            linestyle="none",
            markersize=8,
            label="Centre of mass",
        ),
    ],
)
fig.colorbar(sc, ax=ax, label="Orbit number")

# Right: radial distance vs orbit number — shows gravity-field
# oscillation from the bimodal lobe structure.
radii_km = np.sqrt(xs_km**2 + ys_km**2)
ax2 = axes[1]
ax2.plot(ts_orbit, radii_km, linewidth=1.0)
ax2.axhline(
    orbit_r_km,
    color="k",
    linestyle="--",
    linewidth=0.8,
    label=f"Initial radius ({orbit_r_km:.0f} km)",
)
ax2.axhline(
    bounding_km,
    color="salmon",
    linestyle=":",
    linewidth=0.8,
    label=f"Bounding radius ({bounding_km:.0f} km)",
)
for k in range(1, n_orbits + 1):
    ax2.axvline(k, color="grey", linestyle=":", linewidth=0.5)
ax2.set_xlabel("Orbit number")
ax2.set_ylabel("Distance from COM (km)")
ax2.set_title("Radial distance vs time")
ax2.legend(fontsize=8)
ax2.set_yscale("log")

plt.suptitle(
    f"Contact binary: lobes {r_a_km:.0f} km + {r_b_km:.0f} km, "
    f"orbit radius {orbit_r_km:.0f} km",
    fontsize=10,
)
plt.tight_layout()
plt.show()
