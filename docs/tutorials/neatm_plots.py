import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import kete

# Compute the temperatures of each facet of an object and plot it in 3d
# Set the physical parameters used for the simulation
g_phase = 0.15
emissivity = 0.9
vis_albedo = 0.05

# WISE W3
wavelength = 11560.8


# Beaming is only used by NEATM
beaming = 1.4

# Define the geometry
geom = kete.shape.TriangleEllipsoid(40)
obj2sun = np.array([1, 0, 0])

# Compute the temperature at the subsolar point on the object.
subsolar_temp = kete.flux.sub_solar_temperature(
    -obj2sun, vis_albedo, g_phase, emissivity, beaming
)

# Compute the NEATM facet temperatures for the object
facet_temps = kete.flux.neatm_facet_temps(
    geom.normals,
    subsolar_temp,
    obj2sun,
)

# Plot the results
plt.figure(dpi=150, figsize=(6, 5))
ax = plt.subplot(111, projection="3d")

norm = mpl.colors.Normalize(vmin=0, vmax=max(facet_temps))
m = cm.ScalarMappable(norm=norm, cmap=mpl.colormaps["inferno"])
colors = m.to_rgba(facet_temps, alpha=2)
polygons = Poly3DCollection(geom.facets, edgecolor="None", color=colors)
ax.add_collection3d(polygons)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

bg_color = (0.1, 0.1, 0.1, 1.0)
ax.xaxis.set_pane_color(bg_color)
ax.yaxis.set_pane_color(bg_color)
ax.zaxis.set_pane_color(bg_color)

# Now set color to white (or whatever is "invisible")
ax.xaxis.pane.set_edgecolor("w")
ax.yaxis.pane.set_edgecolor("w")
ax.zaxis.pane.set_edgecolor("w")

ax.quiver(1, -2, 0, 1, 0, 0, length=1, normalize=True, color="w")
ax.text(1.4, -2.6, 0, "Sun", c="w")


plt.title("NEATM")

cbar = plt.colorbar(m, ax=ax, label="Temp (K)", shrink=0.5)
cbar.ax.tick_params(labelsize=10)
plt.tight_layout()
plt.savefig("data/NEATM_heating.png")


# Plot of the relative visible flux from the observer position
# =============================================================

plt.figure(dpi=150, figsize=(6, 5))
ax = plt.subplot(111, projection="3d")

facet_flux = [kete.flux.black_body_flux(t, wavelength) for t in facet_temps]
obs2obj = -kete.Vector.from_lat_lon(ax.elev, ax.azim)

facet_flux = np.array(facet_flux) / np.max(facet_flux)

norm = mpl.colors.Normalize(vmin=0, vmax=max(facet_flux))
m = cm.ScalarMappable(norm=norm, cmap=mpl.colormaps["Greys_r"])
colors = m.to_rgba(facet_flux, alpha=1)
polygons = Poly3DCollection(geom.facets, edgecolor="None", color=colors)
ax.add_collection3d(polygons)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
# First remove fill
bg_color = (0.1, 0.1, 0.1, 1.0)
ax.xaxis.set_pane_color(bg_color)
ax.yaxis.set_pane_color(bg_color)
ax.zaxis.set_pane_color(bg_color)

# Now set color to white (or whatever is "invisible")
ax.xaxis.pane.set_edgecolor("w")
ax.yaxis.pane.set_edgecolor("w")
ax.zaxis.pane.set_edgecolor("w")

ax.quiver(1, -2, 0, 1, 0, 0, length=1, normalize=True, color="w")
ax.text(1.4, -2.6, 0, "Sun", color="w")

plt.title("NEATM Emitted IR Flux")

cbar = plt.colorbar(m, ax=ax, label="Flux (Relative)", shrink=0.5)
cbar.ax.tick_params(labelsize=10)
plt.tight_layout()
plt.savefig("data/NEATM_flux.png")
