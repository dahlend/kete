import pytest

from kete import mpc


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
    obs_list = mpc.MPCObservation.from_lines(MPC_OBS)
    assert len(obs_list) == 1
    obs = obs_list[0]
    assert obs.desig == "1566"
    assert obs.discovery is False
    assert obs.note1 == ""
    assert obs.note2 == "S"
    assert obs.jd == 2455452.157066019
    _ = obs.sc2obj
