import matplotlib.pyplot as plt
import numpy as np

import kete

kete.spice.kernel_reload(["./data/20000042.bsp"])

jd_start = kete.Time.from_ymd(1920, 1, 1).jd
jd_end = kete.Time.from_ymd(2020, 1, 1).jd

state = kete.spice.get_state("42", jd_end)

jds = np.logspace(np.log10(jd_end), np.log10(jd_end - 10), 1000)
jds = np.concatenate(
    [jds, np.logspace(np.log10(jds[-1]), np.log10(jd_start + 1), 10000)]
)

n_body_state = state
n_body_state_ast = state

error_line = []
error_2body = []
n_body_no_asteroids = []
n_body_ast = []
state = kete.spice.get_state("42", jd_end)
for jd in jds:
    jpl_pos = kete.spice.get_state("42", jd).pos

    line = state.pos + state.vel * (jd - state.jd)
    error_line.append((jpl_pos - line).r * kete.constants.AU_KM)

    two_body = kete.propagate_two_body(state, jd).pos
    error_2body.append((jpl_pos - two_body).r * kete.constants.AU_KM)

    n_body_state = kete.propagate_n_body(n_body_state, jd)
    n_body_no_asteroids.append((jpl_pos - n_body_state.pos).r * kete.constants.AU_KM)

    n_body_state_ast = kete.propagate_n_body(
        n_body_state_ast, jd, include_asteroids=True
    )
    n_body_ast.append((jpl_pos - n_body_state_ast.pos).r * kete.constants.AU_KM)

plt.figure(dpi=150)
plt.plot(-(jds[0] - jds) / 365, error_2body, label="Two-Body Approximation")
plt.plot(-(jds[0] - jds) / 365, n_body_no_asteroids, label="N-Body Planets Only")
plt.plot(-(jds[0] - jds) / 365, n_body_ast, label="N-Body Including 5 Asteroids")
plt.yscale("log")
plt.ylim(1.1e-1, 1e8)
plt.ylabel("Position Error (km)")
plt.legend(loc=3, framealpha=1)
plt.xlabel("Time (Years)")
plt.title("42 Isis (A856 KB)")
plt.savefig("./data/orbit_models.png")
plt.close()

plt.figure(dpi=150)
plt.plot(jds[0] - jds, error_line, label="Linear Motion")
plt.plot(jds[0] - jds, error_2body, label="Two-Body Approximation")
plt.plot(jds[0] - jds, n_body_no_asteroids, label="N-Body Planets Only")
plt.plot(jds[0] - jds, n_body_ast, label="N-Body Including 5 Asteroids")
plt.ylim(0, 115)
plt.xscale("log")
plt.ylabel("Position Error (km)")
plt.axhline(74, ls="--", c="k", label="0.1 arcsecond at 1 au")
plt.xlim(2e-3, 3e4)
plt.legend(loc=2, framealpha=1)
plt.xlabel("Time (days)")
plt.title("42 Isis (A856 KB)")
plt.savefig("./data/orbit_models_short.png")
