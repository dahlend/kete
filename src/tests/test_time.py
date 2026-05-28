from kete.time import Time


class TestTime:
    def test_init(self):
        t = Time(2460676.5, scaling="utc")
        assert t.jd == 2460676.500800741
        assert t.mjd == 60676.000800740905
        assert t.ymd == (2025, 1, 1)
        assert t.iso == "2025-01-01T00:00:00+00:00"
        assert Time.from_ymd(2025, 1, 1).jd == t.jd

        assert Time.j2000().jd == 2451545
        assert Time.now().jd > Time.j2000().jd

    def test_from_ymd_tt_scaling(self):
        # Midnight TT on any calendar date is an exact half-integer JD.
        # With UTC scaling the UTC→TDB conversion shifts by ~69 s, breaking this.
        jd_tt = Time.from_ymd(2025, 1, 1, scaling='tt').jd
        assert jd_tt == 2460676.5  # exact half-integer

        # Confirm UTC default still applies the ~69 s offset.
        # (37 leap seconds + 32.184 s TT-TAI)
        jd_utc = Time.from_ymd(2025, 1, 1).jd
        assert abs((jd_utc - jd_tt) * 86400 - 69.184) < 1
