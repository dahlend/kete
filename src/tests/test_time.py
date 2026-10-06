# SPDX-FileCopyrightText: 2026 Dar Dahlen
# SPDX-FileCopyrightText: 2025 California Institute of Technology
# SPDX-License-Identifier: BSD-3-Clause

from kete.time import Time


class TestTime:
    def test_init(self):
        t = Time(2460676.5, scaling="utc")
        assert t.jd == 2460676.50080074
        assert t.mjd == 60676.0008007399
        assert t.ymd == (2025, 1, 1)
        assert t.iso == "2025-01-01T00:00:00+00:00"
        assert Time.from_ymd(2025, 1, 1).jd == t.jd

        assert Time.j2000().jd == 2451545
        assert Time.now().jd > Time.j2000().jd
