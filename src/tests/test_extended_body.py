"""Tests for the kete.extended_body bindings (Phase 7)."""

import math

import numpy as np
import pytest

import kete
from kete.extended_body import (
    ExtendedBody,
    Polyhedron,
    RotationModel,
    body_acceleration,
    is_inside_proximity,
    propagate_near_body,
)


def _icosphere(radius=1.0):
    """Tiny hand-built icosahedron mesh suitable for the gravity tests."""
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
    verts = [tuple(c * radius / math.sqrt(1.0 + phi * phi) for c in v) for v in raw]
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


def test_polyhedron_basic():
    verts, faces = _icosphere(1.0e-8)  # ~1.5 km in AU
    poly = Polyhedron(verts, faces, gm=1.0e-18)
    assert poly.n_vertices == 12
    assert poly.n_faces == 20
    assert poly.n_edges == 30
    assert poly.volume > 0.0
    assert poly.bounding_radius == pytest.approx(1.0e-8, rel=1e-3)
    assert poly.gm == 1.0e-18

    # Surface impact -> ValueError
    with pytest.raises(ValueError):
        poly.acceleration(verts[0])

    # Exterior point gives finite acceleration pointing inward.
    a = poly.acceleration([1.0e-7, 0.0, 0.0])
    assert np.isfinite(a).all()
    assert a[0] < 0.0


def test_polyhedron_from_density():
    verts, faces = _icosphere(1.0e-8)
    poly_dens = Polyhedron.from_density(verts, faces, density_kg_m3=2000.0)
    assert poly_dens.gm > 0.0


def test_extended_body_acceleration_outside():
    verts, faces = _icosphere(1.0e-8)
    poly = Polyhedron(verts, faces, gm=4.0e-20)
    body = ExtendedBody(
        components=[poly],
        rotation=RotationModel.fixed(),
        length_au=1.0e-8,
        proximity_radius_au=1.0e-6,
    )
    assert body.n_components == 1
    assert body.gm == 4.0e-20
    assert body.length_au == 1.0e-8
    assert body.proximity_radius_au == 1.0e-6

    # body_acceleration helper through the proximity-regime evaluator.
    a = body_acceleration(body, [5.0e-8, 0.0, 0.0], jd_tdb=2_451_545.0)
    assert np.isfinite(a).all()
    assert a[0] < 0.0


def test_rotation_model_constant_spin():
    rot = RotationModel.constant_spin(
        pole_ra=86.6,
        pole_dec=-65.5,
        w0=42.0,
        w_dot=2000.0,
        epoch=kete.Time(2_451_545.0),
    )
    assert "constant_spin" in repr(rot)


def test_is_inside_proximity_smoke():
    # Build a body roughly the size of Bennu's bounding radius,
    # and check the predicate around it.  We use an SSB-centered
    # synthetic state so no SPK lookup is required for the body
    # NAIF id (we pass the body's own id as the center to short-
    # circuit the SPK call inside the predicate).
    verts, faces = _icosphere(1.0e-7)
    poly = Polyhedron(verts, faces, gm=4.0e-20)
    body = ExtendedBody(
        components=[poly],
        rotation=RotationModel.fixed(),
        length_au=1.0e-7,
        proximity_radius_au=1.0e-5,
    )
    # Use Sun (10) as the center so the predicate path doesn't hit
    # the SPK; we just want to exercise the wrapper end-to-end.
    state = kete.State(
        desig="probe",
        jd=kete.Time(2_451_545.0),
        pos=[1.0e-6, 0.0, 0.0],
        vel=[0.0, 0.0, 0.0],
        center_id=10,
    )
    inside = is_inside_proximity(state, body, body_naif_id=10, factor=10.0)
    assert isinstance(inside, bool)


def test_seam_continuity_against_ceres():
    """Seam-continuity check using Ceres' SPK-registered ephemeris.

    With the body's own gravity set to negligible and only the Sun in
    the perturber list, ``propagate_near_body`` reduces to integrating
    the particle in a Sun-only field expressed in a frame attached to
    Ceres.  The result must match a heliocentric two-body Kepler
    propagation of the same particle to within a few parts in 1e8 over
    one day.  This exercises the full helio -> body-relative ->
    integrate -> body-relative -> helio path through real SPK lookups.
    """
    jd0 = kete.Time(2_460_000.5)
    jd1 = kete.Time(2_460_001.5)

    ceres = kete.spice.get_state("ceres", jd0)

    # Particle: 30000 km offset, with Ceres' own velocity plus a
    # 1 m/s relative kick.  Heliocentric, Sun-centered.
    offset_au = 30_000.0 / 1.495978707e8  # 30000 km in AU
    rel_v_au_day = 1e-3 / (1.495978707e8 / 86400.0)  # 1 m/s in AU/day
    pos = [ceres.pos[0] + offset_au, ceres.pos[1], ceres.pos[2]]
    vel = [ceres.vel[0], ceres.vel[1] + rel_v_au_day, ceres.vel[2]]
    particle = kete.State(desig="probe", jd=jd0, pos=pos, vel=vel, center_id=10)

    # Tiny synthetic Ceres mesh; gm intentionally negligible so the
    # only acceleration is the Sun perturber.
    verts, faces = _icosphere(1.0e-6)
    poly = Polyhedron(verts, faces, gm=1.0e-30)
    body = ExtendedBody(
        components=[poly],
        rotation=RotationModel.fixed(),
        length_au=1.0e-6,
        proximity_radius_au=1.0e-3,
    )

    near = propagate_near_body(
        body=body,
        body_naif_id=20_000_001,  # Ceres' SPK id
        particle=particle,
        jd_final=jd1,
    )

    kepler = kete.propagation.propagate_two_body(particle, jd1)

    # near is centered on Ceres; bring it back to the Sun for the
    # comparison.
    near_helio = near.change_center(10)

    dpos = np.array(near_helio.pos) - np.array(kepler.pos)
    dvel = np.array(near_helio.vel) - np.array(kepler.vel)
    # 1 day, no other planets: agreement to a few hundred meters /
    # mm-per-second is plenty to confirm the seam math is consistent.
    assert np.linalg.norm(dpos) < 1e-8  # ~1500 m
    assert np.linalg.norm(dvel) < 1e-8


def test_contact_binary_two_lobes():
    """Two offset polyhedron components linearly add their fields.

    Build a contact binary from two icosahedral lobes separated
    along x.  The acceleration at a far-field point on the +x axis
    should be very close to (gm_a + gm_b) / r^2, with small
    corrections from the offset geometry, and components should
    superpose linearly.
    """
    verts, faces = _icosphere(1.0e-8)

    # Two equal-density lobes 4e-8 AU apart (along x).
    sep = 4.0e-8
    verts_a = [(v[0] - sep / 2.0, v[1], v[2]) for v in verts]
    verts_b = [(v[0] + sep / 2.0, v[1], v[2]) for v in verts]
    gm_each = 1.0e-19
    poly_a = Polyhedron(verts_a, faces, gm=gm_each)
    poly_b = Polyhedron(verts_b, faces, gm=gm_each)

    body = ExtendedBody(
        components=[poly_a, poly_b],
        rotation=RotationModel.fixed(),
        length_au=sep,
        proximity_radius_au=1.0e-5,
    )
    assert body.n_components == 2
    assert body.gm == pytest.approx(2.0 * gm_each)

    # Far-field probe at 1e-6 AU on +x: dominated by the total mass.
    r = 1.0e-6
    a = body.acceleration([r, 0.0, 0.0])
    a_point = -2.0 * gm_each / (r * r)
    # Expect a within a few percent of the point-mass estimate.
    assert a[0] == pytest.approx(a_point, rel=2.0e-2)
    assert abs(a[1]) < abs(a[0]) * 1.0e-6
    assert abs(a[2]) < abs(a[0]) * 1.0e-6

    # Linearity check: sum of per-lobe accelerations equals the
    # ExtendedBody aggregate.
    a_a = poly_a.acceleration([r, 0.0, 0.0])
    a_b = poly_b.acceleration([r, 0.0, 0.0])
    np.testing.assert_allclose(a, np.array(a_a) + np.array(a_b), rtol=1.0e-12, atol=0.0)


def test_nongrav_parity_against_heliocentric():
    """propagate_near_body with a NonGravModel must produce the same
    non-grav-induced trajectory delta as the heliocentric N-body
    propagator over a short arc when the body's own gravity is
    negligible.

    Construction:
      - Place a particle near Ceres (negligible body gravity, fixed
        rotation so omega = 0) with the Sun as the only perturber.
      - In each regime, run twice — once with no nongrav, once with
        a JPL comet model — and look at the difference.
      - The two deltas must agree, which isolates the new nongrav
        coupling in the proximity force function from the (different)
        underlying gravity model used by the heliocentric N-body
        propagator (which always includes all planets).
    """
    verts, faces = _icosphere(1.0e-9)  # tiny radius
    poly = Polyhedron(verts, faces, gm=1.0e-30)
    body = ExtendedBody(
        components=[poly],
        rotation=RotationModel.fixed(),
        length_au=1.0e-9,
        proximity_radius_au=1.0e-3,
    )

    jd0 = kete.Time(2_460_000.5)
    jd1 = kete.Time(jd0.jd + 1.0)
    ceres = kete.spice.get_state("ceres", jd0)

    offset_au = 30.0 / kete.constants.AU_KM
    kick_au_day = 1.0e-3 / kete.constants.AU_KM * 86400.0
    pos = [ceres.pos[0] + offset_au, ceres.pos[1], ceres.pos[2]]
    vel = [ceres.vel[0], ceres.vel[1] + kick_au_day, ceres.vel[2]]
    particle = kete.State(desig="probe", jd=jd0, pos=pos, vel=vel, center_id=10)

    # Pick a nongrav strength large enough that the induced delta is
    # well above the integrator's float64 round-off floor (gravity-
    # only parity is ~3e-12 AU over a 1-day arc); a 100x JPL-comet
    # scale puts |delta| at ~1e-9 AU so the SNR is ~1000:1.
    ng = kete.propagation.NonGravModel.new_comet(a1=1.0e-7, a2=5.0e-8, a3=0.0)

    helio_off = kete.propagation.propagate_n_body([particle], jd1, non_gravs=[None])[0]
    helio_on = kete.propagation.propagate_n_body([particle], jd1, non_gravs=[ng])[0]

    common = dict(
        body=body,
        body_naif_id=20_000_001,
        particle=particle,
        jd_final=jd1,
    )
    near_off = propagate_near_body(non_grav=None, **common).change_center(10)
    near_on = propagate_near_body(non_grav=ng, **common).change_center(10)

    dpos_helio = np.array(helio_on.pos) - np.array(helio_off.pos)
    dpos_near = np.array(near_on.pos) - np.array(near_off.pos)
    dvel_helio = np.array(helio_on.vel) - np.array(helio_off.vel)
    dvel_near = np.array(near_on.vel) - np.array(near_off.vel)

    # With the signal well above the integrator floor, the heliocentric
    # and proximity regimes must reproduce the same nongrav-induced
    # delta to better than 0.5%.
    rel_pos = np.linalg.norm(dpos_helio - dpos_near) / max(
        np.linalg.norm(dpos_helio), 1.0e-15
    )
    rel_vel = np.linalg.norm(dvel_helio - dvel_near) / max(
        np.linalg.norm(dvel_helio), 1.0e-15
    )
    assert rel_pos < 5.0e-3, (rel_pos, dpos_helio, dpos_near)
    assert rel_vel < 5.0e-3, (rel_vel, dvel_helio, dvel_near)


def test_nongrav_changes_trajectory():
    """Sanity: turning a non-grav model on must perturb the trajectory
    relative to the gravity-only proximity-regime run.
    """
    verts, faces = _icosphere(1.0e-9)
    poly = Polyhedron(verts, faces, gm=1.0e-30)
    body = ExtendedBody(
        components=[poly],
        rotation=RotationModel.fixed(),
        length_au=1.0e-9,
        proximity_radius_au=1.0e-3,
    )

    jd0 = kete.Time(2_460_000.5)
    jd1 = kete.Time(jd0.jd + 1.0)
    ceres = kete.spice.get_state("ceres", jd0)
    offset_au = 30.0 / kete.constants.AU_KM
    pos = [ceres.pos[0] + offset_au, ceres.pos[1], ceres.pos[2]]
    vel = list(ceres.vel)
    particle = kete.State(desig="probe", jd=jd0, pos=pos, vel=vel, center_id=10)

    common = dict(
        body=body,
        body_naif_id=20_000_001,
        particle=particle,
        jd_final=jd1,
    )

    no_ng = propagate_near_body(non_grav=None, **common).change_center(10)
    with_ng = propagate_near_body(
        non_grav=kete.propagation.NonGravModel.new_comet(a1=1.0e-7, a2=0.0, a3=0.0),
        **common,
    ).change_center(10)

    sep_au = np.linalg.norm(np.array(with_ng.pos) - np.array(no_ng.pos))
    # 1e-7 A1 over a 1-day arc at ~3 AU is comfortably above noise.
    assert sep_au > 1.0e-12


def test_propagate_near_body_impact():
    """A particle dropped onto the surface must raise ValueError tagged
    with the central body's NAIF id, and at a TDB time that lies inside
    the requested propagation window.
    """
    # 100 km icosahedron with Ceres-like density (~2160 kg/m^3) so the
    # body's self-gravity is large enough to reel a slowly-moving
    # particle in within minutes of integrator time.
    radius_au = 100.0 / kete.constants.AU_KM
    verts, faces = _icosphere(radius_au)
    poly = Polyhedron.from_density(verts, faces, density_kg_m3=2160.0)
    body = ExtendedBody(
        components=[poly],
        rotation=RotationModel.fixed(),
        length_au=radius_au,
        proximity_radius_au=10.0 * radius_au,
    )

    jd0 = kete.Time(2_460_000.5)
    jd1 = kete.Time(jd0.jd + 1.0)
    ceres = kete.spice.get_state("ceres", jd0)

    # Particle 1.5 R from the body center on +x with a 50 m/s inward
    # radial kick; gravity does the rest.  Heliocentric, Sun-centered.
    offset_au = 1.5 * radius_au
    v_in_au_day = -0.05 / kete.constants.AU_KM * 86400.0  # m/s -> AU/day
    pos = [ceres.pos[0] + offset_au, ceres.pos[1], ceres.pos[2]]
    vel = [ceres.vel[0] + v_in_au_day, ceres.vel[1], ceres.vel[2]]
    particle = kete.State(desig="impactor", jd=jd0, pos=pos, vel=vel, center_id=10)

    with pytest.raises(ValueError, match="impact with 20000001") as excinfo:
        propagate_near_body(
            body=body,
            body_naif_id=20_000_001,
            particle=particle,
            jd_final=jd1,
        )

    # The error message embeds the TDB time of impact; sanity-check
    # that it falls inside the requested integration window.
    msg = str(excinfo.value)
    t_str = msg.split("at time", 1)[1].strip()
    t_impact = float(t_str)
    assert jd0.jd <= t_impact <= jd1.jd, (jd0.jd, t_impact, jd1.jd)


def test_propagate_near_body_skim_no_impact():
    """A grazing trajectory that stays just outside the body must not
    spuriously trigger an impact.  Companion to
    :func:`test_propagate_near_body_impact`.
    """
    radius_au = 100.0 / kete.constants.AU_KM
    verts, faces = _icosphere(radius_au)
    poly = Polyhedron.from_density(verts, faces, density_kg_m3=2160.0)
    body = ExtendedBody(
        components=[poly],
        rotation=RotationModel.fixed(),
        length_au=radius_au,
        proximity_radius_au=10.0 * radius_au,
    )

    jd0 = kete.Time(2_460_000.5)
    jd1 = kete.Time(jd0.jd + 0.1)
    ceres = kete.spice.get_state("ceres", jd0)

    # Park 5 R out on +x with circular-orbit tangential velocity in +y
    # (no radial component); the orbit stays well above the surface.
    r0_au = 5.0 * radius_au
    v_circ_au_day = math.sqrt(poly.gm / r0_au)
    pos = [ceres.pos[0] + r0_au, ceres.pos[1], ceres.pos[2]]
    vel = [ceres.vel[0], ceres.vel[1] + v_circ_au_day, ceres.vel[2]]
    particle = kete.State(desig="orbiter", jd=jd0, pos=pos, vel=vel, center_id=10)

    out = propagate_near_body(
        body=body,
        body_naif_id=20_000_001,
        particle=particle,
        jd_final=jd1,
    )
    # Output is body-centered; require the final radius to still be
    # comfortably outside the body surface.
    r_final = np.linalg.norm(np.array(out.pos))
    assert r_final > 2.0 * radius_au, (r_final, radius_au)
