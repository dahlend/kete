"""
MCMC Posterior for a Near-Miss Asteroid
=======================================

Create a synthetic near-Earth asteroid on a close-approach trajectory,
observe it over three consecutive nights from Palomar Mountain, then
recover the full non-Gaussian posterior using NUTS MCMC sampling.

The posterior from such a short arc is strongly non-Gaussian -- exactly
the case where MCMC matters for impact probability assessment.

This demonstrates the workflow:

1. Build an Apollo-type NEO orbit that approaches Earth.
2. Observe it from Palomar over 3 nights (2 obs per night, 6 total).
3. Run Gauss IOD to get candidate states.
4. Run :func:`kete.fitting.nuts_sample` directly on the IOD candidates
   to sample the posterior (no differential correction needed).
5. Visualize the distribution in orbital elements and the
   close-approach distance spread.
"""

import matplotlib.pyplot as plt
import numpy as np

import kete

np.random.seed(42)

# %%
# 1. Build a Near-Miss Orbit
# ---------------------------
# Apollo-type orbit: a > 1 AU, q < 1 AU, low inclination.
# The orbital orientation is chosen so perihelion falls near Earth's
# ecliptic longitude ~60 days after the observations, producing a
# realistic discovery scenario at ~0.3 AU geocentric distance with
# a genuine close approach (~0.04 AU) weeks later.

epoch = kete.Time.from_ymd(2028, 9, 15)

elements = kete.CometElements(
    desig="NearMiss",
    epoch=epoch,
    eccentricity=0.45,
    inclination=5.0,
    peri_arg=15.0,
    lon_of_ascending=40.0,
    peri_time=kete.Time(epoch.jd + 60),  # perihelion 60 days after epoch
    peri_dist=0.97,  # AU -- just inside Earth's orbit
)
true_state = elements.state
print(f"True state at epoch: {true_state}")
print(f"  a = {elements.semi_major:.4f} AU,  e = {elements.eccentricity:.4f}")
print(f"  i = {elements.inclination:.2f} deg,  q = {elements.peri_dist:.4f} AU")

# %%
# Verify the close approach: propagate and find the minimum
# distance to Earth over the next 120 days.

jd_start = epoch.jd
jd_end = epoch.jd + 120
step = 0.25  # 6-hour steps
state = true_state
distances = []
jds = np.arange(jd_start, jd_end, step)

for jd in jds:
    state_at_jd = kete.propagate_two_body(true_state, jd)
    earth = kete.spice.get_state("Earth", jd)
    dist_au = (state_at_jd.pos - earth.pos).r
    distances.append(dist_au)

min_dist = min(distances)
min_jd = jds[np.argmin(distances)]
print(
    f"\nClosest approach: {min_dist:.6f} AU "
    f"({min_dist * kete.constants.AU_KM:.0f} km) "
    f"on JD {min_jd:.2f}"
)

# %%
# 2. Generate Synthetic Observations
# ----------------------------------
# Observe over three consecutive nights from Palomar Mountain (MPC 675).
# Two observations per night, ~1.5 hours apart.

# Print geocentric distance at the obs epoch.
earth_obs = kete.spice.get_state("Earth", epoch.jd)
obj_obs = kete.propagate_two_body(true_state, epoch.jd)
geo_dist = (obj_obs.pos - earth_obs.pos).r
print(f"\nGeocentric distance at obs epoch: {geo_dist:.3f} AU")

obs_night_start = epoch.jd
obs_times = np.concatenate(
    [
        obs_night_start + np.array([0.0, 1.5 / 24]),  # night 1
        obs_night_start + 1 + np.array([0.0, 1.5 / 24]),  # night 2
        obs_night_start + 2 + np.array([0.0, 1.5 / 24]),  # night 3
    ]
)

arc_hours = (obs_times[-1] - obs_times[0]) * 24

fovs = []
for jd in obs_times:
    observer = kete.spice.mpc_code_to_ecliptic("675", jd)
    fovs.append(kete.OmniDirectionalFOV(observer))

# Check visibility (applies light-time correction)
visible = kete.fov_state_check([true_state], fovs)

# Convert to fitting Observations
observations = []
for vis in visible:
    observer = vis.fov.observer.as_equatorial.change_center(0)
    ra, dec, _, _ = vis.ra_dec_with_rates[0]

    # Add realistic astrometric noise: 0.3 arcsec
    sigma = 0.3
    obs = kete.fitting.Observation.optical(
        observer=observer,
        ra=ra + np.random.normal(0, sigma / 3600) / max(np.cos(np.radians(dec)), 0.1),
        dec=dec + np.random.normal(0, sigma / 3600),
        sigma_ra=sigma / max(np.cos(np.radians(dec)), 0.1),
        sigma_dec=sigma,
    )
    observations.append(obs)

print(f"\nGenerated {len(observations)} observations over {arc_hours:.1f} hours:")
for i, obs in enumerate(observations):
    print(f"  [{i}] JD {obs.epoch.jd:.4f}  RA={obs.ra:.5f}  Dec={obs.dec:.5f}")

# %%
# 3. Initial Orbit Determination
# --------------------------------
# Use the unified IOD which works on any arc length from single-night
# tracklets to multi-year arcs.  These raw IOD states are passed
# directly to MCMC -- no differential correction is needed.

candidates = []

try:
    iod_cands = kete.fitting.initial_orbit_determination(observations)
    print(f"\nIOD returned {len(iod_cands)} candidate(s)")
    candidates.extend(iod_cands)
except Exception as ex:
    print(f"\nIOD failed: {ex}")

if not candidates:
    raise RuntimeError("No IOD candidates found -- try different observations")

print(f"\nTotal IOD candidates: {len(candidates)}")
for i, cand in enumerate(candidates):
    e = cand.elements
    print(f"  Candidate {i}: a={e.semi_major:.4f} AU, e={e.eccentricity:.4f}")

# %%
# 4. NUTS MCMC Sampling
# ---------------------
# Run the sampler directly on the IOD candidates -- no differential
# correction required.  Using a Student-t likelihood (nu=5) makes the
# sampler robust to occasional outlier steps in a poorly-constrained
# posterior from a short arc.

samples = kete.fitting.nuts_sample(
    seeds=candidates,
    observations=observations,
    num_draws=2000,
    num_tune=500,
    student_nu=5,
)

n_div = sum(samples.divergent)
print(f"\nNUTS sampling complete:")
print(f"  Total draws: {len(samples)}")
print(f"  Chains: {len(set(samples.chain_id))}")
print(f"  Divergent: {n_div} ({100 * n_div / max(len(samples), 1):.1f}%)")

# %%
# 5. Visualize the Posterior
# --------------------------
# Convert draws to orbital elements and plot distributions.

# samples.draws returns Sun-centered Ecliptic States directly.
all_draws = samples.draws
divergent = np.array(samples.divergent)
good = ~divergent  # keep only non-divergent draws

n_good = good.sum()
n_total = len(all_draws)
print(f"  Non-divergent draws: {n_good} / {n_total}")
if n_good == 0:
    raise RuntimeError(
        "All draws were divergent -- the sampler could not explore the posterior. "
        "This usually means the MAP orbit is poor or the likelihood surface is too "
        "steep. Try using student_nu=5 or increasing num_draws."
    )

chain_ids = np.array(samples.chain_id)[good]
draw_states = [s for s, g in zip(all_draws, good) if g]

semi_majors = np.array([s.elements.semi_major for s in draw_states])
eccentricities = np.array([s.elements.eccentricity for s in draw_states])
inclinations = np.array([s.elements.inclination for s in draw_states])
peri_dists = np.array([s.elements.peri_dist for s in draw_states])

# Filter to physically reasonable bound orbits for plotting.
# Short-arc posteriors can include near-parabolic / hyperbolic tails.
bound = (semi_majors > 0) & (semi_majors < 10) & (eccentricities < 1)
print(f"\n{bound.sum()} / {len(draw_states)} draws are bound orbits with a < 10 AU")

if bound.sum() == 0:
    raise RuntimeError(
        f"No bound draws (a in [{semi_majors.min():.2f}, {semi_majors.max():.2f}], "
        f"e in [{eccentricities.min():.4f}, {eccentricities.max():.4f}])."
    )

# True values for comparison
true_a = elements.semi_major
true_e = elements.eccentricity
true_i = elements.inclination
true_q = elements.peri_dist

# %%
# Corner-style plot of orbital elements
# Color-code by chain to show the distinct orbital families.

n_chains = len(set(samples.chain_id))
chain_colors = [f"C{cid % 10}" for cid in chain_ids[bound]]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle(
    f"Posterior Distribution -- {n_chains} chain(s) from 3-night arc",
    fontsize=13,
)

ax = axes[0, 0]
for cid in sorted(set(chain_ids[bound])):
    mask = chain_ids[bound] == cid
    ax.hist(
        semi_majors[bound][mask],
        bins=30,
        density=True,
        alpha=0.5,
        color=f"C{cid % 10}",
        label=f"Chain {cid}",
    )
ax.axvline(true_a, color="red", ls="--", lw=1.5, label="Truth")
ax.set_xlabel("Semi-major axis (AU)")
ax.set_ylabel("Density")
ax.legend(fontsize=8)

ax = axes[0, 1]
ax.scatter(semi_majors[bound], eccentricities[bound], s=1, alpha=0.3, c=chain_colors)
ax.scatter(true_a, true_e, c="red", s=40, marker="x", zorder=5, label="Truth")
ax.set_xlabel("Semi-major axis (AU)")
ax.set_ylabel("Eccentricity")
ax.legend(fontsize=9)

ax = axes[1, 0]
for cid in sorted(set(chain_ids[bound])):
    mask = chain_ids[bound] == cid
    ax.hist(
        eccentricities[bound][mask],
        bins=30,
        density=True,
        alpha=0.5,
        color=f"C{cid % 10}",
    )
ax.axvline(true_e, color="red", ls="--", lw=1.5)
ax.set_xlabel("Eccentricity")
ax.set_ylabel("Density")

ax = axes[1, 1]
for cid in sorted(set(chain_ids[bound])):
    mask = chain_ids[bound] == cid
    ax.hist(
        inclinations[bound][mask],
        bins=30,
        density=True,
        alpha=0.5,
        color=f"C{cid % 10}",
    )
ax.axvline(true_i, color="red", ls="--", lw=1.5)
ax.set_xlabel("Inclination (deg)")
ax.set_ylabel("Density")

plt.tight_layout()
plt.show()

# %%
# Close-Approach Distance Distribution
# --------------------------------------
# Propagate each posterior sample to the close-approach epoch and
# compute the miss distance.  This is the key output for impact
# probability assessment.

print(f"\nPropagating {bound.sum()} bound samples to close-approach epoch...")

# Use only bound draws for propagation (unbound orbits diverge under two-body).
bound_states = [s for s, b in zip(draw_states, bound) if b]
bound_chain_ids = chain_ids[bound]

miss_distances_km = []
for s in bound_states:
    s_ca = kete.propagate_two_body(s, min_jd)
    earth = kete.spice.get_state("Earth", min_jd)
    dist_km = (s_ca.pos - earth.pos).r * kete.constants.AU_KM
    miss_distances_km.append(dist_km)

miss_distances_km = np.array(miss_distances_km)

fig, ax = plt.subplots(figsize=(8, 4))
for cid in sorted(set(bound_chain_ids)):
    mask = bound_chain_ids == cid
    ax.hist(
        miss_distances_km[mask],
        bins=30,
        density=True,
        alpha=0.5,
        color=f"C{cid % 10}",
        label=f"Chain {cid}",
    )
ax.axvline(
    min_dist * kete.constants.AU_KM,
    color="red",
    ls="--",
    lw=1.5,
    label="True miss distance",
)
ax.set_xlabel("Close-approach distance (km)")
ax.set_ylabel("Density")
ax.set_title("Miss Distance Distribution from MCMC Posterior")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

pct_5, pct_50, pct_95 = np.percentile(miss_distances_km, [5, 50, 95])
print(f"Miss distance (km):  5th={pct_5:.0f},  median={pct_50:.0f},  95th={pct_95:.0f}")
print(f"True miss distance:  {min_dist * kete.constants.AU_KM:.0f} km")

# %%
# On-Sky Uncertainty at Close Approach
# --------------------------------------
# Show the on-sky scatter of posterior samples, illustrating the
# non-Gaussian (banana-shaped) uncertainty from a short arc.

earth_ca = kete.spice.get_state("Earth", min_jd)
ras_ca = []
decs_ca = []
for s in bound_states:
    s_ca = kete.propagate_two_body(s, min_jd)
    obs2obj = kete.Vector(s_ca.pos - earth_ca.pos).as_equatorial
    ras_ca.append(obs2obj.ra)
    decs_ca.append(obs2obj.dec)

ras_ca = np.array(ras_ca)
decs_ca = np.array(decs_ca)

# Unwrap RA to handle the 0/360 boundary
ras_ca = np.unwrap(ras_ca, period=360)

bound_colors = [f"C{cid % 10}" for cid in bound_chain_ids]

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(ras_ca, decs_ca, s=1, alpha=0.3, c=bound_colors, label="MCMC draws")

# True position at close approach
true_ca = kete.propagate_two_body(true_state, min_jd)
true_vec = kete.Vector(true_ca.pos - earth_ca.pos).as_equatorial
ax.scatter(
    true_vec.ra, true_vec.dec, c="red", s=60, marker="x", zorder=5, label="Truth"
)

ax.set_xlabel("RA (deg)")
ax.set_ylabel("Dec (deg)")
ax.set_title("On-Sky Uncertainty at Close Approach")
ax.legend()
ax.invert_xaxis()
plt.tight_layout()
plt.show()
