# SPDX-FileCopyrightText: 2025 California Institute of Technology
# SPDX-License-Identifier: BSD-3-Clause

from kete.wise import (
    MISSION_PHASES,
    mission_phase_from_jd,
    mission_phase_from_scan,
)


def test_mission_phase_from_jd():
    for phase in MISSION_PHASES.values():
        p0 = mission_phase_from_jd(phase.jd_start)
        assert p0 == phase, p0.name + "  " + phase.name
        p1 = mission_phase_from_jd(phase.jd_end - 1)
        assert p1 == phase, p1.name + "  " + phase.name

    assert mission_phase_from_jd(1000) is None


def test_mission_phase_from_scan():
    for letter in "rs":
        scan_id = "10000" + letter
        assert mission_phase_from_scan(scan_id) == MISSION_PHASES["Reactivation_2019"]
        scan_id = "64273" + letter
        assert mission_phase_from_scan(scan_id) is None

    scan_id = "01000a"
    assert mission_phase_from_scan(scan_id) == MISSION_PHASES["Cryo"]
    scan_id = "08000b"
    assert mission_phase_from_scan(scan_id) == MISSION_PHASES["3-Band"]
    scan_id = "09000a"
    assert mission_phase_from_scan(scan_id) == MISSION_PHASES["Post-Cryo"]
    scan_id = "49000a"
    assert mission_phase_from_scan(scan_id) == MISSION_PHASES["Reactivation_2014"]
    scan_id = "46000s"
    assert mission_phase_from_scan(scan_id) == MISSION_PHASES["Reactivation_2022"]
