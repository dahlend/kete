# SPDX-FileCopyrightText: 2026 Dar Dahlen
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for kete.orbit_fitting.fetch_gaia_observations."""

import math

import pandas as pd
import pytest

from kete import orbit_fitting
from kete.orbit_fitting import gaia as gaia_mod


def _make_row(
    epoch=0.0,
    ra=10.0,
    dec=5.0,
    ra_error_random=2.0,
    dec_error_random=2.0,
    ra_error_systematic=1.0,
    dec_error_systematic=1.0,
    x_gaia_geocentric=0.009,
    y_gaia_geocentric=0.003,
    z_gaia_geocentric=0.001,
    vx_gaia_geocentric=0.0001,
    vy_gaia_geocentric=0.0002,
    vz_gaia_geocentric=0.0,
    ra_dec_correlation_random=0.0,
    g_mag=15.0,
    astrometric_outcome_transit=1,
):
    """Return a one-row DataFrame with the minimal columns expected."""
    return pd.DataFrame(
        [
            {
                "epoch": epoch,
                "ra": ra,
                "dec": dec,
                "ra_error_random": ra_error_random,
                "dec_error_random": dec_error_random,
                "ra_error_systematic": ra_error_systematic,
                "dec_error_systematic": dec_error_systematic,
                "ra_dec_correlation_random": ra_dec_correlation_random,
                "x_gaia_geocentric": x_gaia_geocentric,
                "y_gaia_geocentric": y_gaia_geocentric,
                "z_gaia_geocentric": z_gaia_geocentric,
                "vx_gaia_geocentric": vx_gaia_geocentric,
                "vy_gaia_geocentric": vy_gaia_geocentric,
                "vz_gaia_geocentric": vz_gaia_geocentric,
                "g_mag": g_mag,
                "astrometric_outcome_transit": astrometric_outcome_transit,
            }
        ]
    )


@pytest.fixture()
def mock_query_tap(monkeypatch):
    """Monkeypatch kete.orbit_fitting.gaia.query_tap and return a factory."""

    def factory(df):
        monkeypatch.setattr(gaia_mod, "query_tap", lambda *a, **kw: df)

    return factory


def test_basic_conversion(mock_query_tap):
    """A single well-formed row should produce one Observation."""
    mock_query_tap(_make_row())
    obs = orbit_fitting.fetch_gaia_observations("99942")
    assert len(obs) == 1
    o = obs[0]
    # Within the solar deflection removed by the loader (a few mas).
    assert o.ra == pytest.approx(10.0, abs=1e-5)
    assert o.dec == pytest.approx(5.0, abs=1e-5)
    assert o.band == "G"
    assert math.isfinite(o.mag)


def test_epoch_jd(mock_query_tap):
    """Epoch offset of 0 should give TDB JD corresponding to TCB 2455197.5."""
    from kete import Time

    mock_query_tap(_make_row(epoch=0.0))
    obs = orbit_fitting.fetch_gaia_observations("433")
    assert len(obs) == 1
    expected_jd = Time(2455197.5, scaling="tcb").jd
    assert obs[0].epoch == pytest.approx(expected_jd, abs=1e-5)


def test_sigma_ra_includes_cosdec(mock_query_tap):
    """
    sigma_ra getter returns sky-plane arcsec (combined error, already includes
    cos(dec)).
    """
    dec = 30.0
    ra_rand = 2.0  # mas
    ra_sys = 1.0  # mas
    mock_query_tap(
        _make_row(dec=dec, ra_error_random=ra_rand, ra_error_systematic=ra_sys)
    )
    obs = orbit_fitting.fetch_gaia_observations("433")
    assert len(obs) == 1
    # Gaia ra_error_* fields are sky-plane (already include cos(dec)).
    # Observation.optical stores the pure-RA direction internally but the
    # sigma_ra getter converts back to sky-plane arcsec for the caller.
    expected_sigma_ra = math.hypot(ra_rand, ra_sys) / 1000.0  # arcsec, sky-plane
    assert obs[0].sigma_ra == pytest.approx(expected_sigma_ra, rel=1e-6)


def test_bad_outcome_rejected(mock_query_tap):
    """Observations with astrometric_outcome_transit != 1 are excluded."""
    mock_query_tap(_make_row(astrometric_outcome_transit=2))
    obs = orbit_fitting.fetch_gaia_observations("433")
    assert obs == []


def test_nan_position_rejected(mock_query_tap):
    """Rows with NaN Gaia position are skipped."""
    mock_query_tap(_make_row(x_gaia_geocentric=float("nan")))
    obs = orbit_fitting.fetch_gaia_observations("433")
    assert obs == []


def test_zero_sigma_rejected(mock_query_tap):
    """Rows where both random and systematic errors are zero are skipped."""
    mock_query_tap(
        _make_row(
            ra_error_random=0.0,
            ra_error_systematic=0.0,
            dec_error_random=0.0,
            dec_error_systematic=0.0,
        )
    )
    obs = orbit_fitting.fetch_gaia_observations("433")
    assert obs == []


def test_empty_table(mock_query_tap):
    """An empty TAP result returns an empty list."""
    mock_query_tap(pd.DataFrame())
    obs = orbit_fitting.fetch_gaia_observations("433")
    assert obs == []


def test_none_table(mock_query_tap):
    """A None TAP result returns an empty list."""
    mock_query_tap(None)
    obs = orbit_fitting.fetch_gaia_observations("433")
    assert obs == []


def test_named_object_query(monkeypatch):
    """A non-numeric designation queries by denomination, not number_mp."""
    captured = {}

    def fake_tap(query, **kw):
        captured["query"] = query
        return pd.DataFrame()

    monkeypatch.setattr(gaia_mod, "query_tap", fake_tap)
    orbit_fitting.fetch_gaia_observations("Apophis")
    assert "denomination" in captured["query"]
    assert "number_mp" not in captured["query"]


def test_numeric_object_query(monkeypatch):
    """A numeric designation queries by number_mp."""
    captured = {}

    def fake_tap(query, **kw):
        captured["query"] = query
        return pd.DataFrame()

    monkeypatch.setattr(gaia_mod, "query_tap", fake_tap)
    orbit_fitting.fetch_gaia_observations("99942")
    assert "number_mp" in captured["query"]
    assert "denomination" not in captured["query"]


def test_missing_g_mag(mock_query_tap):
    """A missing g_mag column produces an Observation with NaN magnitude."""
    df = _make_row()
    df["g_mag"] = None
    mock_query_tap(df)
    obs = orbit_fitting.fetch_gaia_observations("433")
    assert len(obs) == 1
    assert math.isnan(obs[0].mag)


def test_observer_is_earth_plus_geocentric(mock_query_tap):
    """The Gaia position is the Earth's state plus the table's geocentric vector."""
    import numpy as np

    import kete

    mock_query_tap(_make_row())
    obs = orbit_fitting.fetch_gaia_observations("433")
    jd = obs[0].epoch.jd
    earth = kete.spice.get_state("Earth", jd, center=0).as_equatorial
    observer = obs[0].observer.as_equatorial
    if observer.center_id != 0:
        sun = kete.spice.get_state("Sun", jd, center=0).as_equatorial
        pos = np.array(list(observer.pos)) + np.array(list(sun.pos))
    else:
        pos = np.array(list(observer.pos))
    expected = np.array(list(earth.pos)) + np.array([0.009, 0.003, 0.001])
    assert np.allclose(pos, expected, atol=1e-12)


def test_high_correlation_kept(mock_query_tap):
    """An along-scan error ellipse with correlation near 1 is not limited to 0.999."""
    mock_query_tap(
        _make_row(
            ra_error_random=500.0,
            dec_error_random=500.0,
            ra_error_systematic=0.0,
            dec_error_systematic=0.0,
            ra_dec_correlation_random=0.99999,
        )
    )
    obs = orbit_fitting.fetch_gaia_observations("433")
    assert obs[0].sigma_corr == pytest.approx(0.99999, abs=1e-9)


def test_remove_star_deflection():
    """At 90 degrees elongation the direction moves 2 GM / c^2 toward the Sun."""
    import numpy as np

    from kete.constants import SPEED_OF_LIGHT_AUDAY, SUN_GM

    k = 2.0 * SUN_GM / SPEED_OF_LIGHT_AUDAY**2
    # Observer 1 AU along +x; the Sun is toward -x, the source toward +y.
    ra, dec = gaia_mod._remove_star_deflection(90.0, 0.0, np.array([1.0, 0.0, 0.0]))
    assert ra == pytest.approx(90.0 + np.degrees(k), abs=1e-12)
    assert dec == pytest.approx(0.0, abs=1e-12)
