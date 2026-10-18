# SPDX-FileCopyrightText: 2026 Dar Dahlen
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import numpy as np
import pytest

import kete
from kete import linking
from kete.orbit_fitting import Observation

JD0 = 2460400.5
NIGHTS = [0, 1, 2, 4, 6, 9, 12, 14]
SIGMA = 0.1  # arcsec


def _objects():
    """Main belt objects near opposition at JD0, on circular orbits."""
    earth = kete.spice.get_state("Earth", JD0)
    lon0 = np.arctan2(earth.pos[1], earth.pos[0])
    out = []
    for k, (r, dlon, inc) in enumerate(
        [(2.3, 2.0, 3.0), (2.6, -4.0, 8.0), (2.9, 6.0, -5.0), (3.1, -1.0, 15.0)]
    ):
        lon, inc = lon0 + np.radians(dlon), np.radians(inc)
        v = np.sqrt(kete.constants.SUN_GM / r)
        pos = [r * np.cos(lon), r * np.sin(lon), 0.0]
        vel = [
            -v * np.sin(lon) * np.cos(inc),
            v * np.cos(lon) * np.cos(inc),
            v * np.sin(inc),
        ]
        out.append(kete.State(f"obj{k}", JD0, pos, vel))
    return out


@pytest.fixture(scope="module")
def scene():
    """Two detections half an hour apart from the geocenter on each night, of every
    object, with noise, shuffled. Returns the observations and the object index of
    each."""
    times = [JD0 + n + dt for n in NIGHTS for dt in (0.0, 0.02)]
    fovs = [kete.OmniDirectionalFOV(kete.spice.get_state("Earth", jd)) for jd in times]
    rng = np.random.default_rng(42)
    obs, labels = [], []
    for visible in kete.fov_state_check(_objects(), fovs):
        observer = visible.fov.observer
        for state in visible.states:
            seen = state.change_center(0).pos - observer.change_center(0).pos
            noise = rng.normal(0.0, SIGMA / 3600, 2)
            dec = seen.dec + noise[1]
            ra = seen.ra + noise[0] / np.cos(np.radians(dec))
            obs.append(
                Observation.optical(observer, ra, dec, SIGMA, SIGMA, time_sigma=0.0)
            )
            labels.append(int(state.desig[3:]))
    order = rng.permutation(len(obs))
    return [obs[i] for i in order], np.array(labels)[order]


class TestEndToEnd:
    def test_link_and_confirm(self, scene):
        obs, labels = scene
        assert len(obs) == 4 * 2 * len(NIGHTS)
        sights = linking.Sights.from_observations(obs)
        assert len(sights) == len(obs)
        # The two detections of each night are a tracklet, and the tracklets'
        # ends pair.
        units = linking.form_units(sights, 0.0, 3.0, 1.0, 0.1)
        tracklets = [
            u for u in units.indices if len(set(labels[u])) == 1 and len(u) == 2
        ]
        assert len(tracklets) > 0
        origin = kete.spice.get_state("Earth", JD0 + 7)
        grid = linking.hypothesis_grid(origin, 0.8, 2.5)
        assert all(h.origin.jd == origin.jd for h in grid)
        candidates = linking.link(sights, units, grid)

        whole = {
            labels[c.sights[0]]
            for c in candidates
            if len(c.sights) == 2 * len(NIGHTS) and len(set(labels[c.sights])) == 1
        }
        assert whole == {0, 1, 2, 3}

        found = linking.confirm(obs, candidates, min_sights=8, max_rms=3.0)
        assert len(found) == 4
        off = linking.confirm(
            obs, candidates, min_sights=8, max_rms=3.0, min_sights_per_side=1
        )
        assert [f.sights for f in off] == [f.sights for f in found]
        # Every object is seen on eight nights, eight time groups, so a gap
        # limit of zero changes nothing. Time groups up to twenty days long
        # make each object one group, which confirms nothing.
        tight = linking.confirm(obs, candidates, min_sights=8, max_rms=3.0, max_gap=0.0)
        assert [f.sights for f in tight] == [f.sights for f in found]
        lone = linking.confirm(
            obs, candidates, min_sights=8, max_rms=3.0, group_span=20.0
        )
        assert lone == []
        for f in found:
            assert len(f.sights) == 2 * len(NIGHTS)
            assert len(set(labels[f.sights])) == 1
            assert f.fit.rms < 3.0

    def test_candidate_state(self, scene):
        # The state of a candidate is the Sun-centered state of its object at
        # the reference time. Along the line of sight it is only as good as
        # the grid, but seen from the origin it points at the object.
        obs, labels = scene
        sights = linking.Sights.from_observations(obs)
        units = linking.form_units(sights, 0.0, 3.0, 1.0, 0.1)
        origin = kete.spice.get_state("Earth", JD0 + 7).change_center(10)
        grid = linking.hypothesis_grid(origin, 0.8, 2.5)
        objects = _objects()
        checked = set()
        for candidate in linking.link(sights, units, grid):
            label = set(labels[candidate.sights])
            if len(label) != 1 or len(candidate) < 2 * len(NIGHTS):
                continue
            label = label.pop()
            state = candidate.state
            assert state.center_id == 10
            assert state.jd == origin.jd
            true = kete.propagate_two_body([objects[label]], state.jd)[0]
            assert true.center_id == 10
            at = np.array(list(origin.pos.as_equatorial))
            seen = np.array(list(true.pos.as_equatorial)) - at
            found = np.array(list(state.pos.as_equatorial)) - at
            cos = seen @ found / np.linalg.norm(seen) / np.linalg.norm(found)
            assert np.degrees(np.arccos(min(cos, 1.0))) * 3600 < 120
            assert 0.5 < np.linalg.norm(found) / np.linalg.norm(seen) < 1.5
            checked.add(label)
        assert checked == {0, 1, 2, 3}

    def test_sights_from_arrays_match_observations(self, scene):
        obs, _ = scene
        sights = linking.Sights.from_observations(obs, frame=kete.Frames.Equatorial)
        again = linking.Sights(
            sights.time,
            sights.observer,
            sights.direction,
            sights.sigma,
            frame=kete.Frames.Equatorial,
        )
        assert np.allclose(again.observer, sights.observer, rtol=0, atol=1e-15)
        assert np.allclose(again.direction, sights.direction, rtol=0, atol=1e-15)
        assert np.allclose(again.sigma, SIGMA)
        assert np.allclose(again.time, [o.epoch.jd for o in obs], rtol=0, atol=0)
        # Times may be given as Time.
        timed = linking.Sights(
            [kete.Time(t) for t in sights.time],
            sights.observer,
            sights.direction,
            sights.sigma,
            frame=kete.Frames.Equatorial,
        )
        assert timed.time == sights.time

        # The direction is the observed RA and Dec, the observer is Sun-centered.
        o = obs[0]
        want = kete.Vector.from_ra_dec(o.ra, o.dec).as_equatorial
        assert np.allclose(sights.direction[0], list(want), atol=1e-12)
        assert np.allclose(sights.observer[0], list(o.observer.pos.as_equatorial))

        # Ecliptic by default.
        ecliptic = linking.Sights.from_observations(obs[:1])
        assert ecliptic.frame == kete.Frames.Ecliptic
        assert np.allclose(ecliptic.direction[0], list(want.as_ecliptic), atol=1e-12)

    def test_merge_pieces(self, scene):
        obs, labels = scene
        sights = linking.Sights.from_observations(obs)
        units = linking.form_units(sights, 0.0, 3.0, 1.0, 0.1)
        grid = linking.hypothesis_grid(kete.spice.get_state("Earth", JD0 + 7), 0.8, 2.5)
        nights = np.array([o.epoch.jd for o in obs]) - JD0

        # Each object confirmed in two pieces, its first four nights and its
        # last four, from the units within each.
        def piece(first, last):
            own = [
                u
                for u in units.indices
                if all((first <= nights[u]) & (nights[u] < last))
            ]
            candidates = linking.link(sights, linking.Units(own), grid)
            return linking.confirm(
                obs, candidates, min_sights=6, max_rms=3.0, max_gap=3.0
            )

        pieces = piece(-1, 5) + piece(5, 15)
        assert len(pieces) == 8
        merged = linking.merge_pieces(
            obs, pieces, min_sights=6, max_rms=3.0, max_gap=3.0
        )
        assert len(merged) == 4
        for f in merged:
            assert len(f.sights) == 2 * len(NIGHTS)
            assert len(set(labels[f.sights])) == 1
        # Whole objects are left as they are.
        again = linking.merge_pieces(
            obs, pieces=merged, min_sights=6, max_rms=3.0, max_gap=3.0
        )
        assert [f.sights for f in again] == [f.sights for f in merged]
        with pytest.raises(ValueError, match="at least 4"):
            linking.merge_pieces(obs, pieces, min_sights=3)
        with pytest.raises(ValueError, match="past"):
            linking.merge_pieces(obs[:5], pieces)


class TestSearch:
    SETTINGS = dict(min_gap=0.0, max_gap=3.0, min_depth=0.8, max_depth=2.5)

    def test_search(self, scene):
        obs, labels = scene
        found = linking.search(obs, min_sights=8, max_rms=3.0, **self.SETTINGS)
        assert len(found) == 4
        assert {labels[f.sights[0]] for f in found} == {0, 1, 2, 3}
        for f in found:
            assert len(f.sights) == 2 * len(NIGHTS)
            assert len(set(labels[f.sights])) == 1
            assert f.fit.rms < 3.0

        # The first pass confirms every object, so a second adds nothing.
        one = linking.search(
            obs, passes=[(1.0, 1.0)], min_sights=8, max_rms=3.0, **self.SETTINGS
        )
        assert [f.sights for f in one] == [f.sights for f in found]
        # Every object has two detections on each side of its largest gap, so
        # turning the rule off changes nothing.
        off = linking.search(
            obs, min_sights=8, max_rms=3.0, min_sights_per_side=1, **self.SETTINGS
        )
        assert [f.sights for f in off] == [f.sights for f in found]
        # Every object is seen twice a night, in tracklets of two, so its units
        # all hold a tracklet, and leaving out the pairs of lone detections
        # changes nothing. Without tracklets of two there are no units.
        tracklets_only = linking.search(
            obs, lone_pairs=False, min_sights=8, max_rms=3.0, **self.SETTINGS
        )
        assert [f.sights for f in tracklets_only] == [f.sights for f in found]
        none = linking.search(
            obs, min_tracklet=3, lone_pairs=False, min_sights=8, **self.SETTINGS
        )
        assert none == []

    def test_search_stations(self, scene):
        obs, _ = scene
        settings = dict(min_sights=8, max_rms=3.0, **self.SETTINGS)
        stated = {
            "search": linking.search(obs, **settings),
            "tiles": linking.search_windows(obs, **settings),
            "windows": linking.search_windows(obs, window=30.0, **settings),
        }
        # Observatory codes, one hashable label per observation, in a list or
        # an array. Four objects are too few to estimate a factor from, so
        # every observation keeps its stated uncertainty, as without stations.
        codes = ["F51" if k % 3 else 568 for k in range(len(obs))]
        for stations in [codes, np.array(["F51", "G96"] * (len(obs) // 2))]:
            labeled = {
                "search": linking.search(obs, stations=stations, **settings),
                "tiles": linking.search_windows(obs, stations, **settings),
                "windows": linking.search_windows(obs, stations, 30.0, **settings),
            }
            for name, found in labeled.items():
                assert [f.sights for f in found] == [f.sights for f in stated[name]]
                assert [f.fit.rms for f in found] == [f.fit.rms for f in stated[name]]
        for short in [codes[1:], codes + ["F51"]]:
            with pytest.raises(ValueError, match="stations"):
                linking.search(obs, stations=short, **settings)
            with pytest.raises(ValueError, match="stations"):
                linking.search_windows(obs, stations=short, **settings)
            with pytest.raises(ValueError, match="stations"):
                linking.search_windows(obs, stations=short, window=30.0, **settings)
        with pytest.raises(TypeError):
            linking.search(obs, stations=[[k] for k in range(len(obs))], **settings)

    def test_search_keywords(self, scene):
        obs, _ = scene
        with pytest.raises(ValueError, match="at least one pass"):
            linking.search(obs, passes=[], **self.SETTINGS)
        # No two detections move as little as a hundredth of a degree a day.
        assert linking.search(obs, passes=[(1.5, 0.01)], **self.SETTINGS) == []
        # A later pass with a larger rate limit finds the objects.
        found = linking.search(
            obs,
            passes=[(1.5, 0.01), (1.5, 1.0)],
            min_sights=8,
            max_rms=3.0,
            **self.SETTINGS,
        )
        assert len(found) == 4
        # A confirmation which keeps more detections than any object has.
        assert linking.search(obs, min_sights=17, **self.SETTINGS) == []
        # Time groups longer than the whole span confirm nothing.
        assert linking.search(obs, group_span=20.0, **self.SETTINGS) == []
        for name in ["tracklet_span", "group_span"]:
            with pytest.raises(ValueError, match=name):
                linking.search(obs, **{name: -1.0}, **self.SETTINGS)
        with pytest.raises(ValueError, match="min_tracklet must be at least 2"):
            linking.search(obs, min_tracklet=1, **self.SETTINGS)
        with pytest.raises(ValueError, match="min_tracklet must not be negative"):
            linking.search(obs, min_tracklet=-1, **self.SETTINGS)
        with pytest.raises(ValueError, match="at least 4"):
            linking.search(obs, min_sights=3, **self.SETTINGS)
        with pytest.raises(ValueError, match="max_rms"):
            linking.search(obs, max_rms=np.nan, **self.SETTINGS)
        for factor in [-1.0, 0.0, np.nan, np.inf]:
            with pytest.raises(ValueError, match="grid_factor of pass 1"):
                linking.search(obs, passes=[(1.5, 1.0), (factor, 1.0)], **self.SETTINGS)
        # Rates are reported in degrees per day, as given.
        with pytest.raises(ValueError, match="max_rate of pass 0.*got -2.5$"):
            linking.search(obs, passes=[(1.5, -2.5)], **self.SETTINGS)
        for name in ["min_sights", "min_sights_per_side"]:
            with pytest.raises(ValueError, match="must not be negative"):
                linking.search(obs, **{name: -1}, **self.SETTINGS)
        with pytest.raises(ValueError, match="settings"):
            linking.search(obs, threshold=-1.0, **self.SETTINGS)
        with pytest.raises(ValueError, match="min_gap"):
            linking.search(obs, **dict(self.SETTINGS, min_gap=4.0))


class TestTiles:
    SETTINGS = dict(TestSearch.SETTINGS, passes=[(1.5, 1.0)], min_sights=8, max_rms=3.0)

    def test_search_windows_tiled(self, scene):
        obs, labels = scene
        found = sorted(f.sights for f in linking.search(obs, **self.SETTINGS))
        assert len(found) == 4
        # The objects are a few degrees apart, so at nside 16 they fall in
        # different cores, and each is whole in more than one tile.
        sights = linking.Sights.from_observations(obs)
        tiles = linking.tiles(sights, self.SETTINGS["passes"][0][1], nside=16)
        assert len(tiles) > 4
        for nside in [None, 16]:
            tiled = linking.search_windows(obs, nside=nside, **self.SETTINGS)
            assert sorted(f.sights for f in tiled) == found
        with pytest.raises(ValueError, match="max_rms"):
            linking.search_windows(obs, **dict(self.SETTINGS, max_rms=-1.0))
        for nside in [3, -4]:
            with pytest.raises(ValueError, match="power of two"):
                linking.search_windows(obs, nside=nside, **self.SETTINGS)
        with pytest.raises(ValueError, match="parallax_margin"):
            linking.search_windows(obs, parallax_margin=-1.0, **self.SETTINGS)

    def test_search_windows(self, scene):
        obs, labels = scene
        settings = dict(self.SETTINGS, min_sights=6)
        found = sorted(f.sights for f in linking.search_windows(obs, **settings))
        assert [len(f) for f in found] == [16] * 4
        # Windows of eight days starting every four, from the first
        # observation: days 0 to 8 hold nights 0 to 6, then days 4 to 12 hold
        # what the first left of nights 4 to 9, night 9 alone, too few to
        # confirm, then the last, from day 8 on, holds nights 9 to 14. No
        # window holds an object whole; the pieces of the first and last
        # merge into each object.
        windowed = linking.search_windows(obs, window=4.0, **settings)
        assert sorted(f.sights for f in windowed) == found
        nights = sorted({int(o.epoch.jd - JD0 + 0.5) for o in obs})
        first = [n for n in nights if n < 8]
        last = [n for n in nights if n >= 8]
        assert first == [0, 1, 2, 4, 6] and last == [9, 12, 14]
        for f in found:
            # Each piece alone confirms, so each window found one.
            for part in [first, last]:
                own = [i for i in f if int(obs[i].epoch.jd - JD0 + 0.5) in part]
                assert len(own) >= settings["min_sights"]
        # A window longer than the span is one window over all of it.
        one = linking.search_windows(obs, window=30.0, **settings)
        assert sorted(f.sights for f in one) == found
        # The window is at least the max_gap of 3 days.
        for window in [0.0, -1.0, np.nan, np.inf, 2.9]:
            with pytest.raises(ValueError, match="window"):
                linking.search_windows(obs, window=window, **settings)
        short = dict(settings, max_gap=0.0, tracklet_span=0.0)
        with pytest.raises(ValueError, match="more than"):
            linking.search_windows(obs, window=1e-5, **short)
        with pytest.raises(ValueError, match="max_rms"):
            linking.search_windows(obs, window=4.0, **dict(settings, max_rms=-1.0))
        with pytest.raises(ValueError, match="min_tracklet"):
            linking.search_windows(obs, **dict(settings, min_tracklet=1))
        tracklets_only = linking.search_windows(obs, lone_pairs=False, **settings)
        assert sorted(f.sights for f in tracklets_only) == found

    def test_tiles_units(self, scene):
        obs, labels = scene
        sights = linking.Sights.from_observations(obs)
        tiles = linking.tiles(sights, 1.0)
        assert all(t == sorted(t) for t in tiles)
        assert set().union(*map(set, tiles)) == set(range(len(obs)))
        # Each object is whole in some tile.
        for label in range(4):
            own = set(np.flatnonzero(labels == label))
            assert any(own <= set(t) for t in tiles)
        # A degree a day for five days reaches as far as a margin of five
        # degrees at the reference time, where any rate adds nothing.
        sight = sights.direction[0]
        two = linking.Sights(
            [JD0, JD0 + 5], [[1.0, 0, 0]] * 2, [sight, sight], [SIGMA, SIGMA]
        )
        late = linking.tiles(two, 1.0, nside=16, reference=JD0)
        assert linking.tiles(two, 1.0, nside=16, reference=kete.Time(JD0)) == late
        margin = linking.tiles(two, 1e-9, parallax_margin=5.0, nside=16, reference=JD0)
        count = sum(1 in t for t in late)
        assert count > sum(0 in t for t in late)
        assert count == sum(0 in t for t in margin)

    def test_parallax_margin(self):
        assert linking.parallax_margin(0.0, 0.5) == 0.0
        assert np.isclose(
            linking.parallax_margin(0.01, 0.5), np.degrees(np.arcsin(0.02))
        )
        assert linking.parallax_margin(1.0, 0.5) == 180.0
        with pytest.raises(ValueError, match="separation"):
            linking.parallax_margin(-1.0, 0.5)
        with pytest.raises(ValueError, match="min_distance"):
            linking.parallax_margin(0.01, 0.0)

    def test_tiles_validation(self):
        sights = linking.Sights(
            [JD0, JD0 + 1], [[1.0, 0, 0]] * 2, [[0, 1.0, 0]] * 2, [SIGMA, SIGMA]
        )
        # Angles are reported in degrees, as given.
        with pytest.raises(ValueError, match="max_rate.*got -1.5$"):
            linking.tiles(sights, -1.5)
        with pytest.raises(ValueError, match="parallax_margin"):
            linking.tiles(sights, 1.0, parallax_margin=np.inf)
        for nside in [3, -4, 2**40]:
            with pytest.raises(ValueError, match="power of two"):
                linking.tiles(sights, 1.0, nside=nside)
        for reference in [np.nan, -np.inf, 1e16]:
            with pytest.raises(ValueError, match="reference time"):
                linking.tiles(sights, 1.0, reference=reference)
        with pytest.raises(ValueError, match="smaller nside"):
            linking.tiles(sights, 30.0, nside=1024)


class TestValidation:
    def _arrays(self, n=3):
        return (
            [JD0 + i for i in range(n)],
            [[1.0, 0.0, 0.0]] * n,
            [[0.0, 2.0, 0.0]] * n,
            [0.1] * n,
        )

    def test_sights_scales_direction(self):
        sights = linking.Sights(*self._arrays())
        assert np.allclose(sights.direction, [[0.0, 1.0, 0.0]] * 3)
        assert len(sights) == 3

    def test_sights_length_mismatch(self):
        time, observer, direction, sigma = self._arrays()
        with pytest.raises(ValueError, match="same length"):
            linking.Sights(time[:2], observer, direction, sigma)

    @pytest.mark.parametrize(
        "column, value",
        [
            (0, np.nan),
            (0, 1e16),
            (1, [np.inf, 0.0, 0.0]),
            (2, [0.0, 0.0, 0.0]),
            (2, [np.nan, 1.0, 0.0]),
            (3, -1.0),
            (3, 0.0),
            (3, np.inf),
        ],
    )
    def test_sights_bad_value(self, column, value):
        arrays = [list(a) for a in self._arrays()]
        arrays[column][1] = value
        with pytest.raises(ValueError, match="detection 1"):
            linking.Sights(*arrays)

    def test_radar_has_no_line_of_sight(self, scene):
        obs, _ = scene
        radar = Observation.radar_range(0, 0, 0, 0, 0, 0, kete.Time(JD0), 0.1, 1e-8)
        with pytest.raises(ValueError, match="observation 2: .*optical"):
            linking.Sights.from_observations(obs[:2] + [radar])

    def test_sigma_near_pole(self):
        # Unequal uncertainties near the celestial pole: sigma_ra is on the
        # sky, so the sigma of the sight is the root mean square of the two
        # whatever the declination.
        observer = kete.spice.get_state("Earth", JD0)
        for dec in [89.99, -89.9, 0.0]:
            o = Observation.optical(observer, 30.0, dec, 0.3, 0.4, time_sigma=0.0)
            sights = linking.Sights.from_observations([o])
            assert np.isclose(sights.sigma[0], np.sqrt((0.3**2 + 0.4**2) / 2))

    def test_form_units(self):
        # Three detections a day apart in one direction: without tracklets, the
        # two pairs a day apart. With tracklets up to a day and a half, the
        # first two are a tracklet, and the second also pairs with the third.
        sights = linking.Sights(*self._arrays())
        assert sorted(
            linking.form_units(sights, 0.0, 1.5, 1.0, 0.0, lone_pairs=True).indices
        ) == [
            [0, 1],
            [1, 2],
        ]
        units = linking.form_units(sights, 0.0, 1.5, 1.0, tracklet_span=1.5)
        assert len(units) == 2
        assert sorted(units.indices) == [[0, 1], [1, 2]]
        assert linking.form_units(sights, 0.0, 0.5, 1.0, tracklet_span=1.5).indices == [
            [0, 1]
        ]
        with pytest.raises(ValueError):
            linking.form_units(sights, 2.0, 1.0, 1.0, 0.0)
        # The rate is reported in degrees per day, as given.
        with pytest.raises(ValueError, match="max_rate.*got -1.5$"):
            linking.form_units(sights, 0.0, 1.0, -1.5, 0.0)
        for max_rate in [0.0, np.nan, np.inf]:
            with pytest.raises(ValueError, match="max_rate"):
                linking.form_units(sights, 0.0, 1.0, max_rate, 0.0)
        for tracklet_span in [-1.0, np.inf, np.nan]:
            with pytest.raises(ValueError, match="tracklet_span"):
                linking.form_units(sights, 0.0, 1.0, 1.0, tracklet_span)

    def test_form_units_min_tracklet_and_lone_pairs(self):
        # The detections of test_form_units: with tracklets up to a day and a
        # half, the first two are a tracklet, whose end pairs with the third.
        sights = linking.Sights(*self._arrays())
        default = sorted(linking.form_units(sights, 0.0, 1.5, 1.0, 1.5).indices)
        assert default == [[0, 1], [1, 2]]
        # Every pair holds the end of the tracklet, so leaving out the pairs of
        # two lone detections changes nothing.
        units = linking.form_units(sights, 0.0, 1.5, 1.0, 1.5, lone_pairs=False)
        assert sorted(units.indices) == default
        # Below min_tracklet the tracklet is two lone detections, which pair
        # when lone_pairs allows.
        units = linking.form_units(
            sights, 0.0, 1.5, 1.0, 1.5, min_tracklet=3, lone_pairs=True
        )
        assert sorted(units.indices) == [[0, 1], [1, 2]]
        # With no tracklet left, no two detections pair without lone_pairs.
        units = linking.form_units(
            sights, 0.0, 1.5, 1.0, 1.5, min_tracklet=3, lone_pairs=False
        )
        assert units.indices == []
        for min_tracklet in [0, 1]:
            with pytest.raises(ValueError, match="min_tracklet must be at least 2"):
                linking.form_units(
                    sights, 0.0, 1.5, 1.0, 1.5, min_tracklet=min_tracklet
                )
        with pytest.raises(ValueError, match="min_tracklet must not be negative"):
            linking.form_units(sights, 0.0, 1.5, 1.0, 1.5, min_tracklet=-1)

    def test_form_units_track(self):
        # A tracklet of three detections moving a tenth of a degree a day, and
        # a day later one detection on its track and one a quarter of a degree
        # off it. The tracklet's ends pair only with the first, and so do those
        # of a tracklet of two over the same span.
        rate = np.radians(0.1)
        times = [0.0, 10 / 1440, 20 / 1440, 1.0, 1.0]
        directions = [[1.0, rate * t, 0.0] for t in times[:4]]
        directions.append([1.0, rate, np.radians(0.25)])
        directions = [list(np.array(d) / np.linalg.norm(d)) for d in directions]
        n = len(times)
        sights = linking.Sights(
            [JD0 + t for t in times], [[1.0, 0.0, 0.0]] * n, directions, [0.1] * n
        )
        units = linking.form_units(sights, 0.5, 1.5, 1.0, 0.1)
        assert sorted(units.indices) == [[0, 1, 2], [0, 3], [2, 3]]
        keep = [0, 2, 3, 4]
        sights = linking.Sights(
            [JD0 + times[k] for k in keep],
            [[1.0, 0.0, 0.0]] * 4,
            [directions[k] for k in keep],
            [0.1] * 4,
        )
        units = linking.form_units(sights, 0.5, 1.5, 1.0, 0.1)
        assert sorted(units.indices) == [[0, 1], [0, 2], [1, 2]]
        # With the middle detection of the three 20 arcseconds off their line,
        # within the rate limit of both neighbors, the third is off the track
        # of the first two, so there is no tracklet of three.
        bent = [list(d) for d in directions]
        bent[1][2] = np.radians(20 / 3600)
        bent = [list(np.array(d) / np.linalg.norm(d)) for d in bent]
        sights = linking.Sights(
            [JD0 + t for t in times], [[1.0, 0.0, 0.0]] * n, bent, [0.1] * n
        )
        units = linking.form_units(sights, 0.5, 1.5, 1.0, 0.1).indices
        assert [0, 1] in units
        assert all(len(u) < 3 for u in units)

    def test_units(self):
        units = linking.Units([[0, 1], [1, 2, 3]])
        assert len(units) == 2
        assert units.indices == [[0, 1], [1, 2, 3]]
        with pytest.raises(ValueError, match="unit 1 has the negative index -1"):
            linking.Units([[0, 1], [-1, 2]])

    def test_hypothesis_grid(self):
        origin = kete.spice.get_state("Earth", JD0)
        grid = linking.hypothesis_grid(origin, 1.0, 2.0)
        assert all(1.0 <= h.depth <= 2.0 for h in grid)
        assert {tuple(h.direction.as_equatorial) for h in grid} == {
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        }
        # The origin is moved to the Sun.
        moved = linking.hypothesis_grid(origin.change_center(399), 1.0, 2.0)
        assert moved[0].origin.center_id == 10
        assert np.allclose(list(moved[0].origin.pos), list(origin.pos))
        with pytest.raises(ValueError):
            linking.hypothesis_grid(origin, 0.0, 2.0)
        with pytest.raises(ValueError):
            linking.hypothesis_grid(origin, 2.0, 1.0)
        with pytest.raises(ValueError, match="more than"):
            linking.hypothesis_grid(origin, 1e-6, 2.0)
        with pytest.raises(ValueError, match="max_depth"):
            linking.hypothesis_grid(origin, 1.0, 1e300)
        # The default steps are the constants of the module.
        steps = linking.hypothesis_grid(
            origin, 1.0, 2.0, linking.DEPTH_STEP, linking.RATE_STEP
        )
        assert len(steps) == len(grid) < linking.MAX_HYPOTHESES
        assert linking.MAX_UNITS > 0
        assert 0.0 < linking.MIN_ALIGNMENT < 1.0

    def test_link_validation(self):
        sights = linking.Sights(*self._arrays())
        grid = linking.hypothesis_grid(kete.spice.get_state("Earth", JD0), 1.0, 1.1)
        for unit in [[0, 5], [1, 1], [1], [2, 1], [0, 2, 1]]:
            with pytest.raises(ValueError, match="not two or more sights"):
                linking.link(sights, linking.Units([unit]), grid)
        for setting in ["threshold", "grid_allowance", "max_uncertainty"]:
            for value in [-1.0, np.inf, np.nan]:
                with pytest.raises(ValueError, match="settings"):
                    linking.link(sights, linking.Units([]), grid, **{setting: value})
        with pytest.raises(ValueError, match="both zero"):
            linking.link(
                sights, linking.Units([]), grid, threshold=0.0, grid_allowance=0.0
            )
        with pytest.raises(ValueError, match="max_uncertainty is zero"):
            linking.link(sights, linking.Units([]), grid, max_uncertainty=0.0)
        with pytest.raises(ValueError, match="min_sights must not be negative"):
            linking.link(sights, linking.Units([]), grid, min_sights=-1)
        assert linking.link(sights, linking.Units([]), grid) == []

        # The unit spans one day, so a reference time more than a day outside
        # it is refused, and one less than a day outside is not.
        pairs = linking.Units([[0, 1]])
        for days, refused in [(-1.5, True), (-0.9, False), (1.9, False), (2.5, True)]:
            origin = kete.spice.get_state("Earth", JD0 + days)
            grid = linking.hypothesis_grid(origin, 1.0, 1.1)
            if refused:
                with pytest.raises(ValueError, match="outside the span"):
                    linking.link(sights, pairs, grid)
            else:
                linking.link(sights, pairs, grid)

    def test_confirm_validation(self, scene):
        obs, _ = scene
        sights = linking.Sights.from_observations(obs)
        units = linking.form_units(sights, 0.0, 3.0, 1.0, 0.1)
        grid = linking.hypothesis_grid(kete.spice.get_state("Earth", JD0 + 7), 0.8, 2.5)
        candidates = linking.link(sights, units, grid)
        with pytest.raises(ValueError, match="indexes past"):
            linking.confirm(obs[:10], candidates, 4, 3.0)
        with pytest.raises(ValueError, match="at least 4"):
            linking.confirm(obs, candidates, 3, 3.0)
        for max_rms in [0.0, -1.0, np.inf, np.nan]:
            with pytest.raises(ValueError, match="max_rms"):
                linking.confirm(obs, candidates, 4, max_rms)
        for group_span in [-1.0, np.inf, np.nan]:
            with pytest.raises(ValueError, match="group_span"):
                linking.confirm(obs, candidates, 4, 3.0, group_span=group_span)
        for max_gap in [-1.0, np.nan]:
            with pytest.raises(ValueError, match="max_gap"):
                linking.confirm(obs, candidates, 4, 3.0, max_gap=max_gap)
        with pytest.raises(ValueError, match="min_sights_per_side must be at least 1"):
            linking.confirm(obs, candidates, min_sights_per_side=0)
        for name in ["min_sights", "min_sights_per_side"]:
            with pytest.raises(ValueError, match=f"{name} must not be negative"):
                linking.confirm(obs, candidates, **{name: -1})
        assert linking.confirm(obs, [], 4, 3.0) == []
        assert linking.confirm(obs, [], 4, 3.0, group_span=0.0, max_gap=np.inf) == []
        # The defaults of confirm are those of search.
        assert linking.confirm(obs, []) == []
        assert [f.sights for f in linking.confirm(obs, candidates)] == [
            f.sights
            for f in linking.confirm(
                obs, candidates, min_sights=4, max_rms=2.0, min_sights_per_side=2
            )
        ]

    def test_hypotheses(self):
        origin = kete.spice.get_state("Earth", JD0)
        grid = linking.hypothesis_grid(origin, 1.0, 1.2)
        assert isinstance(grid, linking.Hypotheses)
        items = list(grid)
        assert len(items) == len(grid) > 10
        assert all(isinstance(h, linking.Hypothesis) for h in items)
        assert grid[-1].depth == items[-1].depth
        assert grid[3].depth_rate == items[3].depth_rate
        with pytest.raises(IndexError):
            grid[len(grid)]
        with pytest.raises(IndexError):
            grid[-len(grid) - 1]

        part = grid[2:9:3]
        assert isinstance(part, linking.Hypotheses)
        assert [h.depth_rate for h in part] == [h.depth_rate for h in items[2:9:3]]
        assert len(grid[::-1]) == len(grid)
        assert grid[::-1][0].depth_rate == items[-1].depth_rate
        assert len(grid[5:2]) == 0


def _circular(radius, days):
    """Sun-centered position and velocity on a circular orbit in the Equatorial
    xy plane, at phase zero at JD0."""
    n = np.sqrt(kete.constants.SUN_GM / radius**3)
    c, s = np.cos(n * days), np.sin(n * days)
    return radius * np.array([c, s, 0.0]), radius * n * np.array([-s, c, 0.0])


def _opposition_sight(days):
    """Observer at 1 AU and the direction to an object at 2.5 AU, both on circular
    orbits and at opposition along the Equatorial x axis at JD0, with light time."""
    observer, _ = _circular(1.0, days)
    light_time = 0.0
    for _ in range(4):
        obj, _ = _circular(2.5, days - light_time)
        light_time = (
            np.linalg.norm(obj - observer) / kete.constants.SPEED_OF_LIGHT_AUDAY
        )
    seen = obj - observer
    return observer, seen / np.linalg.norm(seen)


def _rotate_z(direction, arcsec):
    """A direction turned by an angle in arcseconds about the z axis."""
    angle = np.radians(arcsec / 3600)
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = direction
    return np.array([c * x - s * y, s * x + c * y, z])


@pytest.fixture(scope="module")
def grid():
    """Hypotheses about the observer of `_opposition_sight` at JD0."""
    position, velocity = _circular(1.0, 0.0)
    origin = kete.State("origin", JD0, position, velocity, frame=kete.Frames.Equatorial)
    return linking.hypothesis_grid(origin, 1.0, 2.0)


class TestUnits:
    """Settings whose units decide the outcome, near the values where it flips."""

    @pytest.mark.parametrize("gap", [0.5, 1.0, 3.0])
    def test_max_rate_in_degrees_per_day(self, gap):
        # Two directions 1.5 degrees per day of gap apart.
        angle = np.radians(1.5 * gap)
        sights = linking.Sights(
            [JD0, JD0 + gap],
            [[1.0, 0.0, 0.0]] * 2,
            [[0.0, 1.0, 0.0], [np.sin(angle), np.cos(angle), 0.0]],
            [0.1] * 2,
        )
        assert (
            linking.form_units(sights, 0.0, 5.0, 1.49, 0.0, lone_pairs=True).indices
            == []
        )
        assert linking.form_units(
            sights, 0.0, 5.0, 1.51, 0.0, lone_pairs=True
        ).indices == [[0, 1]]

    def _one_pair(self, sigma):
        """The object seen at JD0 and a day later, with this sigma in arcsec."""
        (o0, d0), (o1, d1) = _opposition_sight(0.0), _opposition_sight(1.0)
        return linking.Sights(
            [JD0, JD0 + 1], [o0, o1], [d0, d1], [sigma] * 2, kete.Frames.Equatorial
        )

    def test_sigma_and_max_uncertainty(self, grid):
        # The uncertainty of a pair is the error of its position at its middle,
        # sigma times its range over the square root of two on the worst axis,
        # plus the error of its velocity, the square root of two times that
        # over its gap of a day, times the half day from the middle to the
        # reference time at the first sight: the square root of two times
        # sigma times the range, within a few percent, as the light of the
        # first sight left the object some minutes before, so the reference
        # time is a little inside the pair. The pair takes part only while
        # that is at most max_uncertainty of the range: at 1e-5, up to a sigma
        # of about 1.46 arcseconds.
        pairs = linking.Units([[0, 1]])

        def found(sigma, **settings):
            got = linking.link(self._one_pair(sigma), pairs, grid, **settings)
            return [c.sights for c in got]

        assert found(1.4, min_sights=2, max_uncertainty=1e-5) == [[0, 1]]
        assert found(1.5, min_sights=2, max_uncertainty=1e-5) == []
        # By default the largest uncertainty is 0.05, up to a sigma of about
        # 7290 arcseconds.
        assert found(7_100, min_sights=2) == [[0, 1]]
        assert found(7_500, min_sights=2) == []
        # Four sights by default.
        assert found(1.4) == []

    def _two_pairs(self, sigma, offset):
        """The pair of `_one_pair`, and a copy of it with the observers moved a
        little along the lines of sight and the second direction turned by
        `offset` arcseconds. The orbits of the two pairs then miss the other's
        second sight by `offset` times its range, under every hypothesis."""
        (o0, d0), (o1, d1) = _opposition_sight(0.0), _opposition_sight(1.0)
        shift = 1e-6
        return linking.Sights(
            [JD0, JD0 + 1, JD0, JD0 + 1],
            [o0, o1, o0 + shift * d0, o1 + shift * d1],
            [d0, d1, d0, _rotate_z(d1, offset)],
            [sigma] * 4,
            kete.Frames.Equatorial,
        )

    def test_grid_allowance_in_arcsec(self, grid):
        # With no threshold the miss allowed is the grid allowance alone.
        sights = self._two_pairs(0.1, 20.0)
        pairs = linking.Units([[0, 1], [2, 3]])

        def found(allowance, **settings):
            got = linking.link(
                sights, pairs, grid, threshold=0.0, grid_allowance=allowance, **settings
            )
            return [c.sights for c in got]

        assert found(18.0) == []
        assert found(22.0) == [[0, 1, 2, 3]]
        assert found(22.0, min_sights=5) == []

    def test_sigma_in_threshold(self, grid):
        # With no grid allowance the miss allowed is the threshold times the
        # error noise gives the miss. A pair knows its position best at its
        # middle, to sigma times the range over the square root of two on
        # each axis, and its velocity to the square root of two times that
        # over its gap. Carried half a day to the second sight of the other
        # pair, with that sight's own error added, the error is the square
        # root of two times sigma times the range, so the 20 arcsecond miss is
        # allowed from sigma = 3.54 arcseconds at a threshold of 4.
        pairs = linking.Units([[0, 1], [2, 3]])

        def found(sigma):
            got = linking.link(
                self._two_pairs(sigma, 20.0),
                pairs,
                grid,
                threshold=4.0,
                grid_allowance=0.0,
            )
            return [c.sights for c in got]

        assert found(3.45) == []
        assert found(3.65) == [[0, 1, 2, 3]]

    def test_hypotheses_as_list(self, grid):
        sights = self._two_pairs(0.1, 20.0)
        pairs = linking.Units([[0, 1], [2, 3]])
        held = linking.link(sights, pairs, grid)
        listed = linking.link(sights, pairs, list(grid))
        assert [c.sights for c in held] == [c.sights for c in listed] == [[0, 1, 2, 3]]
        assert list(held[0].state.pos) == list(listed[0].state.pos)


def _fields(times, radius=15.0, overlap=False):
    """Cones from the geocenter at each time: one of this radius in degrees
    around the direction away from the Sun, one towards the Sun, and optionally one of a
    degree around the first object, inside the first cone."""
    first = _objects()[0]
    out = []
    for jd in times:
        earth = kete.spice.get_state("Earth", jd)
        away = earth.pos
        out.append(kete.fov.ConeFOV(away, radius, earth))
        out.append(kete.fov.ConeFOV(-away, 5.0, earth))
        if overlap:
            at = kete.propagate_two_body([first], jd)[0]
            out.append(kete.fov.ConeFOV(at.pos - earth.pos, 1.0, earth))
    return out


def _far_object():
    """A main belt object a quarter turn ahead of the Earth, in none of the
    fields."""
    earth = kete.spice.get_state("Earth", JD0)
    lon = np.arctan2(earth.pos[1], earth.pos[0]) + np.pi / 2
    v = np.sqrt(kete.constants.SUN_GM / 2.5)
    return kete.State(
        "far",
        JD0,
        [2.5 * np.cos(lon), 2.5 * np.sin(lon), 0.0],
        [-v * np.sin(lon), v * np.cos(lon), 0.0],
    )


def _separation(a, b):
    """Angle in arcseconds between two directions given as RA and Dec."""
    u = np.array(list(kete.Vector.from_ra_dec(*a).as_equatorial))
    w = np.array(list(kete.Vector.from_ra_dec(*b).as_equatorial))
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(u, w)), u @ w)) * 3600


class TestInject:
    TIMES = [JD0 + n + dt for n in [0, 1, 3] for dt in (0.0, 0.02)]

    def _expected(self, objects, fields):
        """The field, object and true RA and Dec of each detection, from
        fov_state_check one field at a time, an object seen in two fields of
        one time from the first of them only."""
        out, taken = [], set()
        for i, fov in enumerate(fields):
            for seen in kete.fov_state_check(objects, [fov]):
                for state, vec in zip(seen.states, seen.obs_vecs):
                    k = [s.desig for s in objects].index(state.desig)
                    if (fov.observer.jd, k) in taken:
                        continue
                    taken.add((fov.observer.jd, k))
                    vec = vec.as_equatorial
                    out.append((i, k, vec.ra, vec.dec))
        return sorted(out)

    def test_directions_and_labels(self):
        objects = _objects() + [_far_object()]
        fields = _fields(self.TIMES, overlap=True)
        expected = self._expected(objects, fields)
        # The four objects at each time, the first once although two fields
        # of that time hold it, and not the far one.
        assert len(expected) == 4 * len(self.TIMES)
        sigma = [0.1 + 0.01 * i for i in range(len(fields))]
        obs, labels = linking.inject(objects, fields, sigma, seed=3)
        assert labels.dtype.kind == "i"
        assert list(labels) == [k for _, k, _, _ in expected]
        offsets = []
        for o, (i, _, ra, dec) in zip(obs, expected):
            assert o.epoch.jd == fields[i].observer.jd
            assert np.isclose(o.sigma_ra, sigma[i]) and np.isclose(
                o.sigma_dec, sigma[i]
            )
            offsets.append(_separation((o.ra, o.dec), (ra, dec)) / sigma[i])
        # Noise of sigma along each of two directions on the sky.
        assert 0.0 < np.max(offsets) < 6.0
        assert 0.5 < np.sqrt(np.mean(np.square(offsets)) / 2) < 1.5

        # The same seed gives the same observations, another seed others.
        again, again_labels = linking.inject(objects, fields, sigma, seed=3)
        assert [(o.ra, o.dec) for o in again] == [(o.ra, o.dec) for o in obs]
        assert list(again_labels) == list(labels)
        other, _ = linking.inject(objects, fields, sigma, seed=4)
        assert [o.ra for o in other] != [o.ra for o in obs]

    def test_detect(self):
        objects = _objects() + [_far_object()]
        fields = _fields(self.TIMES)
        obs, labels = linking.inject(objects, fields, 0.1, detect=0.0, seed=1)
        assert obs == [] and len(labels) == 0 and labels.dtype.kind == "i"
        every, _ = linking.inject(objects, fields, 0.1, seed=1)
        assert len(every) == 4 * len(self.TIMES)
        half, _ = linking.inject(objects, fields, 0.1, detect=0.5, seed=1)
        assert 0 < len(half) < len(every)

        called = []

        def detect(fov, state):
            assert any(fov is f for f in fields)
            assert state.center_id == 10
            # The state at the time light left the object, minutes before.
            assert 0.0 < fov.observer.jd - state.jd < 0.02
            called.append(state.desig)
            return 1.0 if state.desig == "obj2" else 0.0

        obs, labels = linking.inject(objects, fields, 0.1, detect=detect, seed=1)
        assert sorted(set(called)) == ["obj0", "obj1", "obj2", "obj3"]
        assert list(labels) == [2] * len(self.TIMES)
        with pytest.raises(ValueError, match="detect must return"):
            linking.inject(objects, fields, 0.1, detect=lambda fov, state: 2.0)

    def test_validation(self):
        objects, fields = _objects(), _fields(self.TIMES[:2])
        for sigma in [0.0, -1.0, np.nan, np.inf, [0.1, 0.1, 0.1, -0.1]]:
            with pytest.raises(ValueError, match="sigma must be positive"):
                linking.inject(objects, fields, sigma)
        with pytest.raises(ValueError, match="sigma has 3 values for 4"):
            linking.inject(objects, fields, [0.1] * 3)
        for detect in [-0.1, 1.1, np.nan]:
            with pytest.raises(ValueError, match="detect must be from 0 to 1"):
                linking.inject(objects, fields, 0.1, detect=detect)
        with pytest.raises(ValueError, match="non_gravs"):
            linking.inject(objects, fields, 0.1, non_gravs=[None])
        for no_objects, no_fields in [([], fields), (objects, [])]:
            obs, labels = linking.inject(no_objects, no_fields, 0.1)
            assert obs == [] and len(labels) == 0

    def test_search_and_score(self):
        objects = _objects() + [_far_object()]
        times = [JD0 + n + dt for n in NIGHTS for dt in (0.0, 0.02)]
        injected, labels = linking.inject(objects, _fields(times, 30.0), SIGMA, seed=5)
        # Other detections, scattered over the sky at the same times.
        rng = np.random.default_rng(6)
        others = [
            Observation.optical(
                kete.spice.get_state("Earth", jd),
                rng.uniform(0, 360),
                rng.uniform(-60, 60),
                SIGMA,
                SIGMA,
            )
            for jd in times
        ]
        obs = others + injected
        labels = np.concatenate([np.full(len(others), -1), labels])
        found = linking.search(obs, min_sights=8, max_rms=3.0, **TestSearch.SETTINGS)
        table = linking.score(obs, found, labels)
        assert list(table.object) == [0, 1, 2, 3]
        assert list(table.detections) == [2 * len(NIGHTS)] * 4
        assert np.allclose(table.span, times[-1] - times[0])
        assert table.found.all()
        assert list(table.pieces) == [1] * 4
        assert list(table.largest) == [2 * len(NIGHTS)] * 4
        assert list(table.in_mixed) == [0] * 4
        assert table.attrs["mixed"] == 0
        assert linking.score(obs, [], labels).found.sum() == 0


class TestScore:
    def test_pieces_and_mixed(self):
        # Confirmed objects cannot be made by hand; score reads only sights.
        labels = [0, 0, 0, 0, 1, 1, 1, -1, 2, -1]
        days = [0, 1, 2, 3, 0, 1, 2, 5, 4, 6]
        obs = [
            Observation.optical(kete.spice.get_state("Earth", JD0 + d), 0, 0, 1, 1)
            for d in days
        ]
        found = [
            SimpleNamespace(sights=s)
            for s in [[0, 1], [2], [3, 4, 7], [5, 6], [7, 9], [8, 9]]
        ]
        table = linking.score(obs, found, np.array(labels))
        assert list(table.object) == [0, 1, 2]
        assert list(table.detections) == [4, 3, 1]
        assert np.allclose(table.span, [3, 2, 0])
        assert list(table.found) == [True, True, False]
        assert list(table.pieces) == [2, 1, 0]
        assert list(table.largest) == [2, 2, 0]
        assert list(table.in_mixed) == [1, 1, 1]
        # [3, 4, 7] and [8, 9] mix labels with synthetic ones; [7, 9] holds
        # other observations only.
        assert table.attrs["mixed"] == 2

        with pytest.raises(ValueError, match="one label per observation"):
            linking.score(obs, found, labels[1:])
        with pytest.raises(ValueError, match="integers"):
            linking.score(obs, found, np.array(labels, dtype=float))
        with pytest.raises(ValueError, match="confirmed object 1 indexes past"):
            linking.score(obs, [found[0], SimpleNamespace(sights=[10])], labels)
        empty = linking.score([], [], [])
        assert len(empty) == 0 and empty.attrs["mixed"] == 0
