import numpy as np
import pytest

import kete
from kete import mpc, orbit_fitting


@pytest.mark.parametrize(
    "packed, unpacked",
    [
        ["J95X00A", "1995 XA"],
        ["J95X01L", "1995 XL1"],
        ["J95F13B", "1995 FB13"],
        ["J98SA8Q", "1998 SQ108"],
        ["J98SC7V", "1998 SV127"],
        ["J98SG2S", "1998 SS162"],
        ["K99AJ3Z", "2099 AZ193"],
        ["K08Aa0A", "2008 AA360"],
        ["K07Tf8A", "2007 TA418"],
        ["K16J01B", "2016 JB1"],
        # surveys
        ["PLS2040", "2040 P-L"],
        ["T1S3138", "3138 T-1"],
        ["T2S1010", "1010 T-2"],
        ["T3S4101", "4101 T-3"],
        # pre 1925
        ["I01A00A", "A801 AA"],
        # comets
        ["J96N020", "1996 N2"],
        ["J95A010", "1995 A1"],
        ["J94P01b", "1994 P1-B"],
        ["J94P010", "1994 P1"],
        ["K48X130", "2048 X13"],
        ["K33L89c", "2033 L89-C"],
        ["K88AA30", "2088 A103"],
        ["CK20F030", "C/2020 F3"],
        ["PK16S00V", "P/2016 SV"],
        ["PK05SL6B", "P/2005 SB216"],
        ["DJ18W010", "D/1918 W1"],
        ["XJ51G020", "X/1951 G2"],
        ["CK20F030", "C/2020 F3"],
        ["PK16S00V", "P/2016 SV"],
        ["PK05SL6B", "P/2005 SB216"],
        ["DJ18W010", "D/1918 W1"],
        ["XJ51G020", "X/1951 G2"],
        ["K20F030", "2020 F3"],
        ["K16S00V", "2016 SV"],
        ["K05SL6B", "2005 SB216"],
        ["J18W010", "1918 W1"],
        ["J51G020", "1951 G2"],
        ["K16J01b", "2016 J1-B"],
        # planet sats
        ["J005S", "Jupiter V"],
        ["S019S", "Saturn XIX"],
        ["U004S", "Uranus IV"],
        ["N011S", "Neptune XI"],
    ],
)
def test_provisional(packed, unpacked):
    assert mpc.unpack_designation(packed) == unpacked
    assert mpc.pack_designation(unpacked) == packed


@pytest.mark.parametrize(
    "unpacked, packed",
    [
        ["50", "00050"],
        ["619999", "z9999"],
        ["620000", "~0000"],
        ["620025", "~000P"],
        ["203289", "K3289"],
        ["15396335", "~zzzz"],
        ["2I", "0002I"],
        ["212P", "0212P"],
    ],
)
def test_permanent(unpacked, packed):
    assert mpc.unpack_designation(packed) == unpacked
    assert mpc.pack_designation(unpacked) == packed


def test_MPCObservation():
    MPC_OBS = [
        "01566         S2010 09 12.65630 "
        "17 32 56.69 -65 49 50.3                L~0MylC51",
        "01566         s2010 09 12.65630 "
        "1 +  238.2318 - 2934.1497 - 6253.4539   ~0MylC51",
    ]
    obs_list = orbit_fitting.MPCObservation.from_lines(MPC_OBS)
    assert len(obs_list) == 1
    obs = obs_list[0]
    assert obs.desig == "1566"
    assert obs.discovery is False
    assert obs.note1 == ""
    assert obs.note2 == "S"
    assert obs.jd == 2455452.157066019
    _ = obs.sc2obj


def test_mpc_obs_to_observations_ground():
    """Ground-based MPC observations convert correctly."""

    # Palomar Mountain (675), note2 = "C" (CCD).
    # RA = 17h 32m 56.69s, Dec = -65 49 50.3
    lines = [
        "01566         C2010 09 12.65630 "
        "17 32 56.69 -65 49 50.3                L~0Myl675",
    ]
    mpc_obs = orbit_fitting.MPCObservation.from_lines(lines)
    assert len(mpc_obs) == 1

    obs_list = orbit_fitting.mpc_obs_to_observations(mpc_obs)
    assert len(obs_list) == 1
    obs = obs_list[0]

    # RA/Dec may be bias-corrected; check they are within 1 arcsec of the input.
    assert abs(obs.ra - mpc_obs[0].ra) < 1.0 / 3600.0
    assert abs(obs.dec - mpc_obs[0].dec) < 1.0 / 3600.0

    # The observer getter returns Sun-centered Ecliptic state.
    assert obs.observer.center_id == 10
    # Sigma should be positive.
    assert obs.sigma_dec > 0.0
    assert obs.sigma_ra > 0.0


def test_mpc_obs_to_observations_spacecraft():
    """Spacecraft MPC observations (note2 == S) convert correctly."""

    lines = [
        "01566         S2010 09 12.65630 "
        "17 32 56.69 -65 49 50.3                L~0MylC51",
        "01566         s2010 09 12.65630 "
        "1 +  238.2318 - 2934.1497 - 6253.4539   ~0MylC51",
    ]
    mpc_obs = orbit_fitting.MPCObservation.from_lines(lines)
    assert len(mpc_obs) == 1
    assert mpc_obs[0].note2 == "S"

    obs_list = orbit_fitting.mpc_obs_to_observations(mpc_obs)
    assert len(obs_list) == 1
    obs = obs_list[0]

    assert abs(obs.ra - mpc_obs[0].ra) < 1.0 / 3600.0
    assert abs(obs.dec - mpc_obs[0].dec) < 1.0 / 3600.0
    assert obs.observer.center_id == 10


# SWAN26Q NEOCP lines: STEREO-A (C49) with geocentric offsets in AU (units flag 2),
# and a roving observer (247) giving longitude, latitude, and altitude.
SC_AU_LINES = [
    "     SWAN26Q 2S2026 08 11.95713016 13 38.424-39 50 06.00               vNEOCPC49",
    "     SWAN26Q 2s2026 08 11.9571302 +0.10231295 +0.99336454 +0.43090773   NEOCPC49",
]
ROVING_LINES = [
    "     SWAN26Q 4V2026 09 03.12877411 43 34.872+20 38 23.03         13.2 gWNEOCP247",
    "     SWAN26Q 4v2026 09 03.1287741 247.16759   37.63058   3190          WNEOCP247",
]


def test_spacecraft_offset_units_flag():
    """Column 33 selects km (1) or AU (2) for the geocentric offset."""
    obs = orbit_fitting.MPCObservation.from_lines(SC_AU_LINES)[0]
    earth = kete.spice.get_state("Earth", obs.jd).pos
    offset = np.array(obs.sun2sc) - np.array(list(earth))
    expected = np.linalg.norm([0.10231295, 0.99336454, 0.43090773])
    assert np.isclose(np.linalg.norm(offset), expected, rtol=1e-9)

    km_line = SC_AU_LINES[1][:32] + "1" + SC_AU_LINES[1][33:]
    obs_km = orbit_fitting.MPCObservation.from_lines([SC_AU_LINES[0], km_line])[0]
    offset_km = np.array(obs_km.sun2sc) - np.array(list(earth))
    assert np.isclose(
        np.linalg.norm(offset_km), expected / kete.constants.AU_KM, rtol=1e-6
    )

    bad_line = SC_AU_LINES[1][:32] + " " + SC_AU_LINES[1][33:]
    with pytest.raises(SyntaxError, match="units flag"):
        orbit_fitting.MPCObservation.from_lines([SC_AU_LINES[0], bad_line])


def test_roving_observer():
    """Roving observer lines are parsed and placed at their stated location."""
    mpc_obs = orbit_fitting.MPCObservation.from_lines(ROVING_LINES)
    assert len(mpc_obs) == 1
    obs = mpc_obs[0]
    assert obs.note2 == "V"
    assert obs.geodetic == (37.63058, 247.16759, 3.19)

    converted = orbit_fitting.mpc_obs_to_observations(mpc_obs, debias=False)[0]
    expected = kete.spice.earth_pos_to_ecliptic(obs.jd, 37.63058, 247.16759, 3.19)
    assert np.allclose(converted.observer.pos, expected.pos, atol=1e-12)


def test_unsupported_lines_warn(caplog):
    """Skipped observation types are reported rather than dropped silently."""
    radar = (
        "01566         R2010 09 12.65630 "
        "17 32 56.69 -65 49 50.3                L~0Myl251"
    )
    with caplog.at_level("WARNING"):
        found = orbit_fitting.MPCObservation.from_lines([radar])
    assert found == []
    assert "Skipped 1 MPC lines" in caplog.text


def test_roving_sites_reweight_separately():
    """Roving observers at different sites are not grouped as one station."""
    first = ROVING_LINES[1]
    moved = first[:36] + "8" + first[37:]
    lines = []
    for _ in range(4):
        lines += [ROVING_LINES[0], first, ROVING_LINES[0], moved]
    mpc_obs = orbit_fitting.MPCObservation.from_lines(lines)
    assert len({o.geodetic for o in mpc_obs}) == 2
    observations = orbit_fitting.mpc_obs_to_observations(mpc_obs, debias=False)
    single = orbit_fitting.mpc_obs_to_observations(mpc_obs[:1], debias=False)[0]
    for obs in observations:
        assert np.isclose(obs.sigma_ra, single.sigma_ra)


def test_ades_wgs84_altitude_meters():
    """ADES WGS84 altitude is in meters."""
    from kete.orbit_fitting.mpc_api import _build_observer

    rec = {"sys": "WGS84", "pos1": "247.16759", "pos2": "37.63058", "pos3": "3190"}
    jd = 2461000.5
    observer = _build_observer("247", jd, rec)
    expected = kete.spice.earth_pos_to_ecliptic(
        jd, 37.63058, 247.16759, 3.19, center=10
    ).as_equatorial
    assert np.allclose(observer.pos, expected.pos, atol=1e-12)
