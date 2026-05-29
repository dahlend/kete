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
