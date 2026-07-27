import subprocess
import sys

import numpy as np
import pytest

from kete import Frames, NonGravModel, State, SymplecticSim, Time, Vector
from kete.conversion import compute_diameter
from kete.propagation import a_over_m_from_physical, lambda_0_from_physical


@pytest.fixture
def jd0():
    return Time.from_ymd(2020, 1, 1).jd


def test_from_spice_energy_conservation(jd0):
    sim = SymplecticSim.from_spice(jd0, dt=4.0)
    assert sim.jd == jd0
    assert sim.include_gr
    assert sim.include_j2
    assert sim.use_correctors
    # Sun + 8 planets + 5 asteroids
    assert len(sim.massive_states) == 14
    # Mercury sets the shortest period
    assert sim.shortest_period == pytest.approx(88.0, abs=1.0)

    e0 = sim.energy
    sim.integrate_n_steps(10_000)
    assert sim.jd == jd0 + 40_000
    # correctors are on by default, so the energy band is far below the
    # kernel-level 1e-6 budget
    assert abs((sim.energy - e0) / e0) < 1e-8

    # corrector off runs remain within the kernel budget
    kernel = SymplecticSim.from_spice(jd0, dt=4.0, use_correctors=False)
    e0 = kernel.energy
    kernel.integrate_n_steps(10_000)
    assert abs((kernel.energy - e0) / e0) < 1e-6


def test_integrate_to_rounds_to_whole_steps(jd0):
    sim = SymplecticSim.from_spice(jd0, dt=10.0, include_registered=False)
    assert len(sim.massive_states) == 9
    sim.integrate_to(Time(jd0 + 1004.0))
    assert sim.steps_taken == 100
    assert sim.jd == jd0 + 1000.0
    # cannot integrate backwards with a positive dt
    with pytest.raises(ValueError):
        sim.integrate_to(Time(jd0))


def test_backwards_integration(jd0):
    sim = SymplecticSim.from_spice(jd0, dt=-4.0, include_registered=False)
    sim.integrate_n_steps(100)
    assert sim.jd == jd0 - 400.0


def test_test_particle_reversibility(jd0):
    # A main-belt-like test particle propagated 40 years forward, then
    # integrated backwards from the evolved system; it must return to its
    # starting position at the roundoff level. The backward simulation is
    # built through the explicit constructor from the evolved states.
    from kete._core import known_masses

    tp = State(
        "probe",
        jd0,
        Vector([2.5, 0.4, 0.1], Frames.Ecliptic),
        Vector([-0.002, 0.010, 0.0002], Frames.Ecliptic),
        center_id=0,
    )
    sim = SymplecticSim.from_spice(
        jd0, dt=4.0, include_registered=False, test_particles=[tp]
    )
    assert sim.n_test_particles == 1
    sim.integrate_n_steps(3650)
    mid = sim.test_particle_states[0]
    assert mid.desig == "probe"

    mass_by_id = {naif: mass for (_, naif, mass, _) in known_masses()}
    masses = [mass_by_id[naif] for naif in [10, 1, 2, 3, 4, 5, 6, 7, 8]]
    back = SymplecticSim(
        sim.massive_states, masses, test_particles=[mid], dt=-4.0
    )
    back.integrate_n_steps(3650)
    final = back.test_particle_states[0]
    diff = np.linalg.norm(np.array(final.pos) - np.array(tp.pos))
    assert diff < 1e-9
    assert sim.lost_particles == []


def test_sun_impact_recorded(jd0):
    tp = State(
        "plunger",
        jd0,
        Vector([0.1, 0.0, 0.05], Frames.Ecliptic),
        Vector([0.0, 1e-4, 0.0], Frames.Ecliptic),
        center_id=10,
    )
    sim = SymplecticSim.from_spice(
        jd0, dt=0.05, include_registered=False, test_particles=[tp]
    )
    sim.integrate_n_steps(200)
    assert sim.n_test_particles == 0
    (desig, jd_lost, reason) = sim.lost_particles[0]
    assert desig == "plunger"
    assert reason == "sun_impact"
    assert jd0 < jd_lost < jd0 + 10.0


def test_constructor_validation(jd0):
    sun = State("sun", jd0, Vector([0, 0, 0]), Vector([0, 0, 0]), center_id=0)
    # mismatched lengths
    with pytest.raises(ValueError):
        SymplecticSim([sun], [1.0, 2.0])
    # first body must be the sun
    with pytest.raises(ValueError):
        SymplecticSim([sun], [0.5])
    # non-empty non_gravs must have one entry per test particle
    grain = State(
        "g", jd0, Vector([2, 0, 0]), Vector([0, 0.011, 0]), center_id=10
    )
    with pytest.raises(ValueError):
        SymplecticSim(
            [sun],
            [1.0],
            test_particles=[grain],
            non_gravs=[NonGravModel.new_dust(0.1), NonGravModel.new_dust(0.2)],
        )
    # beta >= 1 is unbound and rejected
    with pytest.raises(ValueError):
        SymplecticSim(
            [sun],
            [1.0],
            test_particles=[grain],
            non_gravs=[NonGravModel.new_dust(1.0)],
        )
    # the A1/A2/A3 model is accepted in its un-lagged form, but the
    # time-lagged outgassing variant is rejected
    ok = SymplecticSim(
        [sun],
        [1.0],
        test_particles=[grain],
        non_gravs=[NonGravModel.new_asteroid(1e-9, 0.0, 0.0)],
    )
    assert ok.n_test_particles == 1
    lagged = NonGravModel.new_comet(1e-9, 0.0, 0.0, dt=30.0)
    with pytest.raises(ValueError):
        SymplecticSim(
            [sun], [1.0], test_particles=[grain], non_gravs=[lagged]
        )
    # NaN (fit-free) parameters follow the documented propagation convention
    # and are treated as 0.0, so this constructs as a force-free particle.
    free = NonGravModel.new_farnocchia(
        a_over_m=float("nan"),
        lambda_0=0.5,
        albedo=0.15,
        absorptivity=0.9,
        flattening=1.0,
        spin_pole=[0, 0, 1],
    )
    sim = SymplecticSim([sun], [1.0], test_particles=[grain], non_gravs=[free])
    assert sim.n_test_particles == 1


def test_dust_pr_inspiral(jd0):
    # Poynting-Robertson inspiral of a circular grain through the Python API,
    # Sun only. da/dt = -2 beta GMS / (c a), so a^2 decays linearly at
    # d(a^2)/dt = -4 beta GMS / c. This is the physical benchmark (Wyatt &
    # Whipple 1950), and the same result the Rust suite certifies.
    from kete.constants import SPEED_OF_LIGHT_AUDAY, SUN_GM

    beta, a0 = 0.02, 2.5
    mu_eff = (1.0 - beta) * SUN_GM
    v0 = np.sqrt(mu_eff / a0)
    zero = Vector([0, 0, 0], Frames.Ecliptic)
    sun = State("sun", jd0, zero, zero, center_id=0)
    grain = State(
        "grain",
        jd0,
        Vector([a0, 0, 0], Frames.Ecliptic),
        Vector([0, v0, 0], Frames.Ecliptic),
        center_id=0,
    )
    period = 2 * np.pi * np.sqrt(a0**3 / mu_eff)
    sim = SymplecticSim(
        [sun],
        [1.0],
        test_particles=[grain],
        dt=period / 40,
        include_gr=False,
        include_j2=False,
        use_correctors=False,
        non_gravs=[NonGravModel.new_dust(beta)],
    )
    assert sim.n_test_particles == 1

    def a_squared():
        s = sim.massive_states[0]
        g = sim.test_particle_states[0]
        pos = np.array(g.pos) - np.array(s.pos)
        vel = np.array(g.vel) - np.array(s.vel)
        a = 1.0 / (2.0 / np.linalg.norm(pos) - vel.dot(vel) / mu_eff)
        return a * a

    times, aa = [], []
    for _ in range(200):
        times.append(sim.jd - jd0)
        aa.append(a_squared())
        sim.integrate_n_steps(40)
    slope = np.polyfit(times, aa, 1)[0]
    predicted = -4.0 * beta * SUN_GM / SPEED_OF_LIGHT_AUDAY
    assert slope == pytest.approx(predicted, rel=1e-3)

    # A grain with beta = 0 must not drift at all (no radiation force).
    grain0 = State(
        "grain0",
        jd0,
        Vector([a0, 0, 0], Frames.Ecliptic),
        Vector([0, np.sqrt(SUN_GM / a0), 0], Frames.Ecliptic),
        center_id=0,
    )
    plain = SymplecticSim(
        [sun], [1.0], test_particles=[grain0], dt=period / 40,
        include_gr=False, include_j2=False, use_correctors=False,
        non_gravs=[NonGravModel.new_dust(0.0)],
    )
    a_start = plain.test_particle_states[0]
    plain.integrate_n_steps(2000)
    end = plain.test_particle_states[0]
    # Circular gravity-only orbit conserves the radius to the roundoff floor.
    r0 = np.linalg.norm(np.array(a_start.pos))
    r1 = np.linalg.norm(np.array(end.pos))
    assert abs(r1 - r0) < 1e-9


def test_yarkovsky_from_h_mag_chain():
    # H -> diameter -> area-to-mass (the radiation pressure coupling).
    # Each step must agree with the standalone helper it composes.
    h_mag, albedo, density = 17.0, 0.15, 2500.0
    y = NonGravModel.new_farnocchia_from_h_mag(
        h_mag=h_mag,
        spin_pole=[0, 0, 1],
        albedo=albedo,
        density=density,
        thermal_inertia=200.0,
        rotation_period=6.0,
    )

    diameter = compute_diameter(albedo, h_mag)
    assert y.a_over_m == pytest.approx(
        a_over_m_from_physical(density, diameter, 1.0), rel=1e-12
    )
    assert y.lambda_0 == pytest.approx(
        lambda_0_from_physical(200.0, 0.9, 0.9, 1.0, 6.0), rel=1e-12
    )

    # A/M = 3 / (4 * density * radius) by hand.
    radius_m = diameter * 500.0
    assert y.a_over_m == pytest.approx(3.0 / (4.0 * density * radius_m), rel=1e-12)

    # The stored A/M inverts back to the assumed density.
    assert y.bulk_density(diameter) == pytest.approx(density, rel=1e-12)

    # An explicit a_over_m is stored as given.
    plain = NonGravModel.new_farnocchia(
        a_over_m=1e-6,
        lambda_0=0.5,
        albedo=0.15,
        absorptivity=0.9,
        flattening=1.0,
        spin_pole=[0, 0, 1],
    )
    assert plain.a_over_m == 1e-6


def test_yarkovsky_h_mag_scaling():
    # Five magnitudes fainter is ten times smaller, hence ten times the
    # area-to-mass and ten times the drift. This is the link that carries an
    # absolute magnitude through to a drift rate.
    bright = NonGravModel.new_farnocchia_from_h_mag(h_mag=15.0, spin_pole=[0, 0, 1])
    faint = NonGravModel.new_farnocchia_from_h_mag(h_mag=20.0, spin_pole=[0, 0, 1])
    assert faint.a_over_m == pytest.approx(bright.a_over_m * 10.0, rel=1e-12)

    # The thermal lag is a surface property, independent of size.
    assert faint.lambda_0 == pytest.approx(bright.lambda_0, rel=1e-12)

    with pytest.raises(ValueError):
        NonGravModel.new_farnocchia_from_h_mag(
            h_mag=17.0, spin_pole=[0, 0, 1], albedo=0.0
        )


def test_yarkovsky_validation(jd0):
    tp = State(
        "probe",
        jd0,
        Vector([2.5, 0.0, 0.0], Frames.Ecliptic),
        Vector([0.0, 0.0109, 0.0], Frames.Ecliptic),
        center_id=0,
    )
    y = NonGravModel.new_farnocchia_from_h_mag(h_mag=17.0, spin_pole=[0, 0, 1])
    # one non-grav entry per test particle
    with pytest.raises(ValueError):
        SymplecticSim.from_spice(jd0, test_particles=[tp], non_gravs=[y, y])
    # negative area-to-mass is unphysical; it is rejected when handed to the
    # simulation (the model itself allows it to stay fittable-friendly)
    sun = State("sun", jd0, Vector([0, 0, 0]), Vector([0, 0, 0]), center_id=0)
    bad = NonGravModel.new_farnocchia(
        a_over_m=-1.0,
        lambda_0=0.5,
        albedo=0.15,
        absorptivity=0.9,
        flattening=1.0,
        spin_pole=[0, 0, 1],
    )
    with pytest.raises(ValueError):
        SymplecticSim([sun], [1.0], test_particles=[tp], non_gravs=[bad])
    # a zero spin pole has no direction
    with pytest.raises(ValueError):
        NonGravModel.new_farnocchia(
            a_over_m=1e-6,
            lambda_0=0.5,
            albedo=0.15,
            absorptivity=0.9,
            flattening=1.0,
            spin_pole=[0, 0, 0],
        )


def _semi_major(sim, idx=0):
    """Osculating heliocentric semi-major axis of a test particle, in AU."""
    from kete.constants import SUN_GM

    sun = sim.massive_states[0]
    tp = sim.test_particle_states[idx]
    pos = np.array(tp.pos) - np.array(sun.pos)
    vel = np.array(tp.vel) - np.array(sun.vel)
    energy = 0.5 * vel.dot(vel) - SUN_GM / np.linalg.norm(pos)
    return -SUN_GM / (2.0 * energy)


def test_yarkovsky_family_spreads_by_obliquity(jd0):
    # The science this exists for: a population with randomly oriented spin
    # poles drifts in both directions, which is what spreads a collisional
    # family in semi-major axis. Poles near the orbit normal drift outward,
    # near-antiparallel poles drift inward, and the rate follows cos(obliquity).
    #
    # The Sun alone keeps the measurement clean. With planets included the
    # osculating semi-major axis oscillates by ~1e-3 AU, thousands of times the
    # drift measured here, and a drifting particle also slides in orbital phase
    # relative to a non-drifting one, which reads back as a further change in
    # osculating `a`. Both are measurement artifacts rather than physics, so
    # the planets are left out and the drift is fit as a slope over whole
    # orbits. `test_yarkovsky_runs_with_planets` covers the full-dynamics path.
    from kete.constants import SUN_GM

    rng = np.random.default_rng(42)
    n = 12
    poles = rng.normal(size=(n, 3))
    poles /= np.linalg.norm(poles, axis=1)[:, None]

    a = 2.4
    sun = State(
        "sun",
        jd0,
        Vector([0, 0, 0], Frames.Ecliptic),
        Vector([0, 0, 0], Frames.Ecliptic),
        center_id=0,
    )
    # Same circular orbit for every member, so any spread comes from the spin
    # pole alone rather than from the initial conditions.
    states = [
        State(
            f"member{idx}",
            jd0,
            Vector([a, 0.0, 0.0], Frames.Ecliptic),
            Vector([0.0, np.sqrt(SUN_GM / a), 0.0], Frames.Ecliptic),
            center_id=0,
        )
        for idx in range(n)
    ]
    # H = 17 with the default albedo is a ~1.4 km body.
    yarks = [
        NonGravModel.new_farnocchia_from_h_mag(
            h_mag=17.0, spin_pole=Vector(list(pole), Frames.Ecliptic)
        )
        for pole in poles
    ]

    period = 2.0 * np.pi * np.sqrt(a**3 / SUN_GM)
    sim = SymplecticSim(
        [sun],
        [1.0],
        test_particles=states,
        non_gravs=yarks,
        dt=period / 50,
        include_gr=False,
        include_j2=False,
        use_correctors=False,
    )
    # Sample once per orbit so the osculating oscillation enters every sample at
    # the same phase and drops out of the slope.
    times, axes = [], []
    for _ in range(100):
        times.append(sim.jd - jd0)
        axes.append([_semi_major(sim, i) for i in range(n)])
        sim.integrate_n_steps(50)
    drift = np.polyfit(times, np.array(axes), 1)[0]  # AU/day, per member

    # Both signs must appear: the family spreads rather than translating.
    assert (drift > 0).any(), "no member drifted outward"
    assert (drift < 0).any(), "no member drifted inward"

    # The rate follows the projection of the pole onto the orbit normal (+z
    # here), so drift / cos(obliquity) is one shared constant.
    # The along-track term works out to exactly cos(obliquity) times a constant,
    # independent of where the pole points within the orbit plane. The in-plane
    # components do contribute a term, but one that averages to zero over a full
    # orbit, leaving only a small residual at finite sampling (measured 3.3e-6
    # relative, since the orbital period creeps as `a` drifts).
    cos_obliquity = poles[:, 2]
    rate = drift / cos_obliquity
    assert np.ptp(rate) / np.abs(rate).mean() < 1e-5, (
        "drift must be proportional to cos(obliquity)"
    )

    # That constant is the zero-obliquity drift rate certified in the Rust
    # suite for these parameters (8.449037e-13 AU/day at a_over_m = 6e-7 and
    # lambda_0 = 0.188, scaled here to H = 17 and the default lag).
    assert rate.mean() == pytest.approx(4.5479e-13, rel=1e-4)


def test_yarkovsky_runs_with_planets(jd0):
    # The full-dynamics path: radiation forces alongside the planets, through
    # `from_spice`. This checks the wiring and the sign of the effect, not the
    # rate; measuring the rate cleanly under planetary perturbations needs the
    # isolation used in `test_yarkovsky_family_spreads_by_obliquity`.
    tp = State(
        "member",
        jd0,
        Vector([2.4, 0.0, 0.0], Frames.Ecliptic),
        Vector([0.0, 0.011106, 0.0], Frames.Ecliptic),
        center_id=10,
    )
    prograde = NonGravModel.new_farnocchia_from_h_mag(
        h_mag=17.0, spin_pole=Vector([0, 0, 1], Frames.Ecliptic)
    )
    retrograde = NonGravModel.new_farnocchia_from_h_mag(
        h_mag=17.0, spin_pole=Vector([0, 0, -1], Frames.Ecliptic)
    )

    def final_a(non_gravs):
        sim = SymplecticSim.from_spice(
            jd0,
            dt=27.0,
            include_registered=False,
            test_particles=[tp],
            non_gravs=non_gravs,
            use_correctors=False,
        )
        assert sim.n_test_particles == 1
        sim.integrate_n_steps(20_000)  # ~1.5 kyr
        return _semi_major(sim)

    gravity_only = final_a(None)
    out = final_a([prograde]) - gravity_only
    inward = final_a([retrograde]) - gravity_only

    # A prograde pole must push the orbit outward relative to gravity alone,
    # a retrograde pole inward, and the two must be near mirror images.
    assert out > 0.0, "prograde spin must drift outward under full dynamics"
    assert inward < 0.0, "retrograde spin must drift inward under full dynamics"
    assert out == pytest.approx(-inward, rel=0.05)

    # Passing no non-grav at all must leave the trajectory untouched.
    assert final_a([None]) == gravity_only


def test_registered_mass_flows_into_from_spice():
    # Registered masses beyond the planets (here the Pluto system, which is
    # in the known-mass table) must be picked up by `from_spice`.
    # Registration is process-global state, so run it in a subprocess to keep
    # this test hermetic.
    code = (
        "import kete\n"
        "jd = kete.Time.from_ymd(2020, 1, 1).jd\n"
        "kete.register_mass(9)\n"
        "sim = kete.SymplecticSim.from_spice(jd)\n"
        "names = [s.desig for s in sim.massive_states]\n"
        "assert len(names) == 15, names\n"
        "assert any('pluto' in n.lower() for n in names), names\n"
        "plain = kete.SymplecticSim.from_spice(jd, include_registered=False)\n"
        "assert len(plain.massive_states) == 9\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
