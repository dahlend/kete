import os
import tempfile

import pytest
from kete._core import SimultaneousStates

import kete
from kete.spitzer import parse_poly, resolve_artifact_url
from kete.vector import State, Vector


def _fake_observer(jd=2451545.0):
    return State("Spitzer", jd, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0])


def test_parse_poly_roundtrip():
    # EWKB polygon from a real IRSA caom.plane row (spitzer_sha, IRAC1, calib_level=2).
    poly_hex = (
        "0103000020E610000001000000050000009DD66D50FBA463C0710168942E554140F1"
        "B913EC3F9E63C04DD9E907756F4140DF5339ED299963C062DA37F75761414049DA8D"
        "3EE69F63C08787307E1A4741409DD66D50FBA463C0710168942E554140"
    )
    corners = parse_poly(poly_hex)
    assert len(corners) == 4
    # All corners should have valid RA [0, 360) and Dec [-90, 90]
    for v in corners:
        eq = v.as_equatorial
        assert 0 <= eq.ra < 360
        assert -90 <= eq.dec <= 90


def test_parse_poly_normalises_negative_ra():
    # Longitude in [-180, 0) must be normalized to [180, 360)
    import struct

    def encode_double(v):
        return struct.pack("<d", v)

    # Build a minimal EWKB with 5 points, lon=-10 (should become 350)
    lon, lat = -10.0, 5.0
    pts = [
        (lon, lat),
        (lon + 0.1, lat),
        (lon + 0.1, lat + 0.1),
        (lon, lat + 0.1),
        (lon, lat),
    ]
    ring_data = struct.pack("<I", len(pts))
    for x, y in pts:
        ring_data += encode_double(x) + encode_double(y)
    header = b"\x01"  # little-endian
    header += struct.pack("<I", 0x20000003)  # EWKB polygon type
    header += struct.pack("<I", 4326)  # SRID
    header += struct.pack("<I", 1)  # 1 ring
    data = header + ring_data
    corners = parse_poly(data.hex())
    assert corners[0].as_equatorial.ra == pytest.approx(350.0, abs=1e-6)


def test_spitzer_frame_from_corners():
    observer = _fake_observer()
    corners = [
        Vector.from_ra_dec(10.0, 20.0),
        Vector.from_ra_dec(10.1, 20.0),
        Vector.from_ra_dec(10.1, 20.1),
        Vector.from_ra_dec(10.0, 20.1),
    ]
    fov = kete.SpitzerFrame(
        corners,
        observer,
        "ivo://test/obs1",
        "IRAC1",
        "ibe://data/spitzer/sha/archive/proc/IRAC000400/r6409728/ch1/bcd/SPITZER_I1_6409728_0000_0000_1_bcd.fits",
        12.0,
    )
    assert fov.band == "IRAC1"
    assert fov.obs_id == "ivo://test/obs1"
    assert len(fov.corners) == 4
    assert fov.duration == pytest.approx(12.0)


def test_spitzer_frame_from_corners_mips():
    observer = _fake_observer()
    corners = [
        Vector.from_ra_dec(10.0, 20.0),
        Vector.from_ra_dec(10.1, 20.0),
        Vector.from_ra_dec(10.1, 20.1),
        Vector.from_ra_dec(10.0, 20.1),
    ]
    fov = kete.SpitzerFrame(corners, observer, "ivo://test/mips", "MIPS24", "", 10.0)
    assert fov.band == "MIPS24"


def test_spitzer_frame_from_pointing():
    observer = _fake_observer()
    pointing = Vector.from_ra_dec(10.05, 20.05)
    fov = kete.SpitzerFrame.from_pointing(
        pointing, 0.0, observer, "ivo://test/obs2", "IRAC2", "", 0.087, 0.087, 12.0
    )
    assert fov.band == "IRAC2"
    assert fov.obs_id == "ivo://test/obs2"


def test_spitzer_frame_jd():
    jd = 2451545.0
    observer = _fake_observer(jd)
    corners = [
        Vector.from_ra_dec(10.0, 20.0),
        Vector.from_ra_dec(10.1, 20.0),
        Vector.from_ra_dec(10.1, 20.1),
        Vector.from_ra_dec(10.0, 20.1),
    ]
    fov = kete.SpitzerFrame(corners, observer, "ivo://test/obs3", "IRAC3", "", 0.0)
    assert abs(fov.jd - jd) < 1e-6


def test_spitzer_frame_repr():
    observer = _fake_observer()
    pointing = Vector.from_ra_dec(45.0, 10.0)
    fov = kete.SpitzerFrame.from_pointing(
        pointing, 0.0, observer, "ivo://test", "IRAC4", "", 0.087, 0.087, 12.0
    )
    r = repr(fov)
    assert "SpitzerFrame" in r
    assert "ivo://test" in r
    assert "IRAC4" in r


def test_spitzer_frame_bad_band():
    observer = _fake_observer()
    corners = [
        Vector.from_ra_dec(10.0, 20.0),
        Vector.from_ra_dec(10.1, 20.0),
        Vector.from_ra_dec(10.1, 20.1),
        Vector.from_ra_dec(10.0, 20.1),
    ]
    with pytest.raises(ValueError, match="SpitzerBand"):
        kete.SpitzerFrame(corners, observer, "ivo://test", "BADBAND", "", 0.0)


def test_spitzer_frame_in_kete_namespace():
    assert hasattr(kete, "SpitzerFrame")
    assert hasattr(kete, "spitzer")


def test_spitzer_frame_binary_roundtrip():
    observer = _fake_observer()
    corners = [
        Vector.from_ra_dec(10.0, 20.0),
        Vector.from_ra_dec(10.1, 20.0),
        Vector.from_ra_dec(10.1, 20.1),
        Vector.from_ra_dec(10.0, 20.1),
    ]
    obs_id = "ivo://irsa.ipac/spitzer_sha?99999/bcd_1"
    artifact_uri = "ibe://data/spitzer/sha/archive/proc/IRAC000400/r99999/ch2/bcd/SPITZER_I2_99999_0000_0000_1_bcd.fits"
    fov = kete.SpitzerFrame(corners, observer, obs_id, "IRAC2", artifact_uri, 12.0)
    ss = SimultaneousStates([observer], fov)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "spitzer_test.bin")
        SimultaneousStates.save_list([ss], path)
        loaded = SimultaneousStates.load_list(path)

    fov2 = loaded[0].fov
    assert type(fov2).__name__ == "SpitzerFrame"
    assert fov2.obs_id == obs_id
    assert fov2.artifact_uri == artifact_uri
    assert fov2.band == "IRAC2"
    assert abs(fov2.jd - fov.jd) < 1e-9
    assert fov2.duration == pytest.approx(12.0)


# --- resolve_artifact_url tests ---


def test_resolve_caom_https_uri():
    uri = (
        "https://irsa.ipac.caltech.edu/data/SPITZER/SHA/archive/proc/"
        "IRAC000400/r6409728/ch1/bcd/SPITZER_I1_6409728_0000_0000_1_bcd.fits"
    )
    url = resolve_artifact_url(uri)
    assert url.startswith("https://irsa.ipac.caltech.edu/ibe/data/spitzer/sha/")
    assert url.endswith("_bcd.fits")


def test_resolve_ibe_uri():
    uri = (
        "ibe://data/spitzer/sha/archive/proc/IRAC000400/r6409728/ch1/bcd/"
        "SPITZER_I1_6409728_0000_0000_1_bcd.fits"
    )
    url = resolve_artifact_url(uri)
    assert url.startswith("https://irsa.ipac.caltech.edu/ibe/data/")
    assert url.endswith("_bcd.fits")


def test_resolve_passthrough_https():
    uri = "https://example.com/some/path_bcd.fits"
    assert resolve_artifact_url(uri) == uri


def test_resolve_unknown_scheme():
    with pytest.raises(ValueError, match="Unrecognised"):
        resolve_artifact_url("ftp://example.com/file.fits")


def test_resolve_bunc_suffix_irac():
    uri = (
        "ibe://data/spitzer/sha/archive/proc/IRAC000400/r6409728/ch1/bcd/"
        "SPITZER_I1_6409728_0000_0000_1_bcd.fits"
    )
    url = resolve_artifact_url(uri, file_type="bunc")
    assert url.endswith("_bunc.fits")
    assert "_bcd.fits" not in url


def test_resolve_bunc_suffix_irs_blue():
    uri = "https://example.com/SPITZER_S0_12345_0000_1_bcdb.fits"
    url = resolve_artifact_url(uri, file_type="bunc")
    assert url.endswith("_buncb.fits")


def test_resolve_bunc_suffix_irs_red():
    uri = "https://example.com/SPITZER_S0_12345_0000_1_bcdr.fits"
    url = resolve_artifact_url(uri, file_type="bunc")
    assert url.endswith("_buncr.fits")


def test_resolve_bad_suffix_for_non_bcd():
    uri = "https://example.com/some_random.fits"
    with pytest.raises(ValueError, match="Cannot derive"):
        resolve_artifact_url(uri, file_type="bunc")


def test_spitzer_frame_irs_peakup():
    observer = _fake_observer()
    corners = [
        Vector.from_ra_dec(10.0, 20.0),
        Vector.from_ra_dec(10.1, 20.0),
        Vector.from_ra_dec(10.1, 20.1),
        Vector.from_ra_dec(10.0, 20.1),
    ]
    fov = kete.SpitzerFrame(
        corners, observer, "ivo://test/irs", "IRS Peak-Up Blue", "", 5.0
    )
    assert fov.band == "IRS Peak-Up Blue"
    assert fov.duration == pytest.approx(5.0)

    # Binary roundtrip
    ss = SimultaneousStates([observer], fov)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "irs_test.bin")
        SimultaneousStates.save_list([ss], path)
        loaded = SimultaneousStates.load_list(path)
    fov2 = loaded[0].fov
    assert fov2.band == "IRS Peak-Up Blue"
    assert fov2.duration == pytest.approx(5.0)
