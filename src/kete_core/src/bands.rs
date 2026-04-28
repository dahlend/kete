//! Photometric band definitions used across kete.
//!
//! Two types are provided:
//!
//! - [`Band`] -- a named photometric band, either a known calibrated preset or
//!   an [`Unknown`](Band::Unknown) label from a data source.  `Copy`, 9 bytes.
//!   Use this to identify which filter an observation was taken in.
//!
//! - [`BandInfo`] -- calibration data (wavelength, zero magnitude,
//!   solar correction, optional color correction).  `Copy`, 32 bytes.
//!   Use this for flux and magnitude computations.
//!
//! [`Band::calibration`] bridges the two: it returns the `BandInfo` for any
//! known preset, and `None` for `Band::Unknown`.
//! [`Band::from_name`] is the only constructor needed for both cases.

use crate::constants::{
    V_MAG_ZERO, w1_color_correction, w2_color_correction, w3_color_correction, w4_color_correction,
};

// AB system zero mag in Jy -- shared across all AB bands
const AB_ZERO: f64 = 3631.0;
// WISE (300 K color-corrected wavelengths and zero mags)
const WISE_WL: [f64; 4] = [3352.6, 4602.8, 11098.368, 22640.508];
const WISE_ZERO: [f64; 4] = [306.681, 170.663, 31.368, 7.953];
const WISE_SUN: [f64; 4] = [1.0049, 1.0193, 1.0024, 1.0012];

// ---------------------------------------------------------------------------
// ColorCorrFn
// ---------------------------------------------------------------------------

/// A function that computes a color correction for a single facet temperature.
///
/// Accepts temperature in kelvin; returns a dimensionless scale factor
/// (typically near 1.0) applied to the facet flux. Used by WISE bands
/// to account for the non-stellar SED of warm thermal emitters.
pub type ColorCorrFn = fn(f64) -> f64;

// ---------------------------------------------------------------------------
// BandInfo -- anonymous calibration data
// ---------------------------------------------------------------------------

/// Anonymous photometric band calibration data.
///
/// Carries the physical properties needed for flux and magnitude computations.
/// Has no name field -- use [`Band`] to identify which filter an observation
/// used, and [`Band::calibration`] to obtain the corresponding `BandInfo`.
///
/// `Copy` -- cheap to store in observation vectors.
#[derive(Debug, Clone, Copy)]
pub struct BandInfo {
    /// Effective central wavelength in nm.
    pub wavelength: f64,
    /// Reflected-light flux scale relative to a solar SED. 1.0 = no change.
    pub solar_correction: f64,
    /// Zero-magnitude flux in Jy. `f64::NAN` when unavailable.
    pub zero_mag: f64,
    /// Optional temperature-dependent color correction (WISE only).
    pub color_correction: Option<ColorCorrFn>,
}

impl BandInfo {
    /// Construct a band from explicit calibration data.
    #[must_use]
    pub const fn new(
        wavelength: f64,
        solar_correction: f64,
        zero_mag: f64,
        color_correction: Option<ColorCorrFn>,
    ) -> Self {
        Self {
            wavelength,
            solar_correction,
            zero_mag,
            color_correction,
        }
    }

    // -----------------------------------------------------------------------
    // Preset calibration constants (no names -- use Band:: for named access)
    // -----------------------------------------------------------------------

    /// Johnson V band (551 nm, Vega).
    pub const V: Self = Self::new(551.0, 1.0, V_MAG_ZERO, None);
    /// Johnson U band (366 nm, Vega).
    pub const U: Self = Self::new(366.0, 1.0, 1790.0, None);
    /// Johnson B band (438 nm, Vega).
    pub const B: Self = Self::new(438.0, 1.0, 4063.0, None);
    /// Cousins R band (641 nm, Vega).
    pub const R: Self = Self::new(641.0, 1.0, 3064.0, None);
    /// Cousins I band (798 nm, Vega).
    pub const I: Self = Self::new(798.0, 1.0, 2416.0, None);
    /// 2MASS J band (1235 nm, Vega).
    pub const J: Self = Self::new(1235.0, 1.0, 1594.0, None);
    /// 2MASS H band (1662 nm, Vega).
    pub const H: Self = Self::new(1662.0, 1.0, 1024.0, None);
    /// 2MASS Ks band (2159 nm, Vega).
    pub const KS: Self = Self::new(2159.0, 1.0, 666.7, None);
    /// Near-IR Y band (1035 nm, Vega).
    pub const Y: Self = Self::new(1035.0, 1.0, 2060.0, None);
    /// SDSS u band (361 nm, AB).
    pub const SDSS_U: Self = Self::new(360.8, 1.0, AB_ZERO, None);
    /// SDSS g band (477 nm, AB).
    pub const SDSS_G: Self = Self::new(477.0, 1.0, AB_ZERO, None);
    /// SDSS r band (623 nm, AB).
    pub const SDSS_R: Self = Self::new(623.1, 1.0, AB_ZERO, None);
    /// SDSS i band (763 nm, AB).
    pub const SDSS_I: Self = Self::new(762.5, 1.0, AB_ZERO, None);
    /// SDSS z band (913 nm, AB).
    pub const SDSS_Z: Self = Self::new(913.4, 1.0, AB_ZERO, None);
    /// Pan-STARRS w band (598 nm, AB). Wide band spanning roughly `g` through `i`.
    pub const PAN_STARRS_W: Self = Self::new(598.0, 1.0, AB_ZERO, None);
    /// Pan-STARRS y band (962 nm, AB).
    pub const PAN_STARRS_Y: Self = Self::new(962.0, 1.0, AB_ZERO, None);
    /// ATLAS orange band (663 nm, AB).
    pub const ATLAS_O: Self = Self::new(663.0, 1.0, AB_ZERO, None);
    /// ATLAS cyan band (518 nm, AB).
    pub const ATLAS_C: Self = Self::new(518.0, 1.0, AB_ZERO, None);
    /// Pan-STARRS g band (481 nm, AB).
    pub const PAN_STARRS_G: Self = Self::new(481.0, 1.0, AB_ZERO, None);
    /// Pan-STARRS r band (616 nm, AB).
    pub const PAN_STARRS_R: Self = Self::new(615.5, 1.0, AB_ZERO, None);
    /// Pan-STARRS i band (750 nm, AB).
    pub const PAN_STARRS_I: Self = Self::new(750.3, 1.0, AB_ZERO, None);
    /// Pan-STARRS z band (867 nm, AB).
    pub const PAN_STARRS_Z: Self = Self::new(866.8, 1.0, AB_ZERO, None);
    /// LSST u band (375 nm, AB).
    pub const LSST_U: Self = Self::new(375.1, 1.0, AB_ZERO, None);
    /// LSST g band (474 nm, AB).
    pub const LSST_G: Self = Self::new(474.1, 1.0, AB_ZERO, None);
    /// LSST r band (617 nm, AB).
    pub const LSST_R: Self = Self::new(617.2, 1.0, AB_ZERO, None);
    /// LSST i band (750 nm, AB).
    pub const LSST_I: Self = Self::new(750.1, 1.0, AB_ZERO, None);
    /// LSST z band (868 nm, AB).
    pub const LSST_Z: Self = Self::new(867.9, 1.0, AB_ZERO, None);
    /// LSST y band (971 nm, AB).
    pub const LSST_Y: Self = Self::new(971.2, 1.0, AB_ZERO, None);
    /// Broad VR filter (~620 nm effective, AB). Effective wavelength is source-spectrum dependent.
    pub const DECAM_VR: Self = Self::new(620.0, 1.0, AB_ZERO, None);
    /// Gaia DR3 G band (673 nm).
    pub const GAIA_G: Self = Self::new(673.0, 1.0, 3228.75, None);
    /// Gaia DR3 BP band (532 nm).
    pub const GAIA_BP: Self = Self::new(532.0, 1.0, 3552.01, None);
    /// Gaia DR3 RP band (797 nm).
    pub const GAIA_RP: Self = Self::new(797.0, 1.0, 2554.95, None);

    /// Two NEOS bands (4.7 and 8.0 um).
    pub const NEOS: [Self; 2] = [
        Self::new(4700.0, 1.0, 170.662, None),
        Self::new(8000.0, 1.0, 64.13, None),
    ];

    /// Four WISE bands (W1-W4) with 300 K color corrections.
    pub const WISE: [Self; 4] = [
        Self::new(
            WISE_WL[0],
            WISE_SUN[0],
            WISE_ZERO[0],
            Some(w1_color_correction),
        ),
        Self::new(
            WISE_WL[1],
            WISE_SUN[1],
            WISE_ZERO[1],
            Some(w2_color_correction),
        ),
        Self::new(
            WISE_WL[2],
            WISE_SUN[2],
            WISE_ZERO[2],
            Some(w3_color_correction),
        ),
        Self::new(
            WISE_WL[3],
            WISE_SUN[3],
            WISE_ZERO[3],
            Some(w4_color_correction),
        ),
    ];

    /// Four Spitzer/IRAC channels (3.6, 4.5, 5.8, 8.0 um).
    pub const IRAC: [Self; 4] = [
        Self::new(3600.0, 1.0, 280.9, None),
        Self::new(4500.0, 1.0, 179.7, None),
        Self::new(5800.0, 1.0, 115.0, None),
        Self::new(8000.0, 1.0, 64.9, None),
    ];

    /// Three Spitzer/MIPS bands (24, 70, 160 um).
    pub const MIPS: [Self; 3] = [
        Self::new(23680.0, 1.0, 7.17, None),
        Self::new(71420.0, 1.0, 0.778, None),
        Self::new(155_900.0, 1.0, 0.159, None),
    ];

    /// Two Spitzer/IRS Peak-Up bands (Blue ~15.8 um, Red ~22.3 um).
    pub const IRS_PU: [Self; 2] = [
        Self::new(15800.0, 1.0, 15.6, None),
        Self::new(22300.0, 1.0, 7.80, None),
    ];

    /// Convenience: resolve a band name to calibration data.
    ///
    /// Returns `None` for unrecognised names. Equivalent to
    /// `Band::from_name(s).calibration()`.
    #[must_use]
    pub fn from_name(s: &str) -> Option<Self> {
        Band::from_name(s).calibration()
    }
}

// ---------------------------------------------------------------------------
// Band -- named photometric band identifier
// ---------------------------------------------------------------------------

/// A photometric band identifier.
///
/// Known variants carry no data and are `Copy` unit types.
/// [`Band::Unknown`] stores an unrecognized label inline as up to 8 ASCII
/// bytes (zero-padded).
///
/// Use [`Band::from_name`] to parse a string, [`Band::name`] to recover the
/// canonical label, and [`Band::calibration`] to obtain the physical
/// calibration data for flux computation.
///
/// `Copy` -- fits in a machine word (9 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Band {
    /// Johnson V band.
    V,
    /// Johnson U band.
    U,
    /// Johnson B band.
    B,
    /// Cousins R band.
    R,
    /// Cousins I band.
    I,
    /// 2MASS J band (1235 nm).
    J,
    /// 2MASS H band (1662 nm).
    H,
    /// 2MASS Ks band (2159 nm).
    Ks,
    /// Near-IR Y band (1035 nm).
    Y,
    /// SDSS u band (361 nm, AB).
    SdssU,
    /// SDSS g band (477 nm, AB).
    SdssG,
    /// SDSS r band (623 nm, AB).
    SdssR,
    /// SDSS i band (763 nm, AB).
    SdssI,
    /// SDSS z band (913 nm, AB).
    SdssZ,
    /// Pan-STARRS g band (481 nm, AB).
    PanStarrsG,
    /// Pan-STARRS r band (616 nm, AB).
    PanStarrsR,
    /// Pan-STARRS i band (750 nm, AB).
    PanStarrsI,
    /// Pan-STARRS z band (867 nm, AB).
    PanStarrsZ,
    /// Pan-STARRS w band (598 nm, AB).
    PanStarrsW,
    /// Pan-STARRS y band (962 nm, AB).
    PanStarrsY,
    /// LSST u band (375 nm, AB).
    LsstU,
    /// LSST g band (474 nm, AB).
    LsstG,
    /// LSST r band (617 nm, AB).
    LsstR,
    /// LSST i band (750 nm, AB).
    LsstI,
    /// LSST z band (868 nm, AB).
    LsstZ,
    /// LSST y band (971 nm, AB).
    LsstY,
    /// Broad VR filter (~620 nm effective, AB).
    DecamVr,
    /// ATLAS orange band (663 nm, AB).
    AtlasO,
    /// ATLAS cyan band (518 nm, AB).
    AtlasC,
    /// Gaia DR3 G band (673 nm).
    GaiaG,
    /// Gaia DR3 BP band (532 nm).
    GaiaBP,
    /// Gaia DR3 RP band (797 nm).
    GaiaRP,
    /// WISE W1 band (3.35 um).
    W1,
    /// WISE W2 band (4.60 um).
    W2,
    /// WISE W3 band (11.56 um).
    W3,
    /// WISE W4 band (22.09 um).
    W4,
    /// NEOS band 1 (4.7 um).
    Neos1,
    /// NEOS band 2 (8.0 um).
    Neos2,
    /// Spitzer IRAC channel 1 (3.6 um).
    Irac1,
    /// Spitzer IRAC channel 2 (4.5 um).
    Irac2,
    /// Spitzer IRAC channel 3 (5.8 um).
    Irac3,
    /// Spitzer IRAC channel 4 (8.0 um).
    Irac4,
    /// Spitzer MIPS 24 um band.
    Mips24,
    /// Spitzer MIPS 70 um band.
    Mips70,
    /// Spitzer MIPS 160 um band.
    Mips160,
    /// Spitzer IRS Peak-Up Blue (~15.8 um).
    IrsPuBlue,
    /// Spitzer IRS Peak-Up Red (~22.3 um).
    IrsPuRed,
    /// Band with no standard calibration. Carries the original label as
    /// zero-padded ASCII bytes; use [`Band::name`] to read it back.
    Unknown([u8; 8]),
}

impl Band {
    // -----------------------------------------------------------------------
    // Preset group arrays
    // -----------------------------------------------------------------------

    /// Four WISE bands in order W1-W4.
    pub const WISE: [Self; 4] = [Self::W1, Self::W2, Self::W3, Self::W4];
    /// Two NEOS bands.
    pub const NEOS: [Self; 2] = [Self::Neos1, Self::Neos2];
    /// Four Spitzer IRAC channels.
    pub const IRAC: [Self; 4] = [Self::Irac1, Self::Irac2, Self::Irac3, Self::Irac4];
    /// Three Spitzer MIPS bands.
    pub const MIPS: [Self; 3] = [Self::Mips24, Self::Mips70, Self::Mips160];
    /// Two Spitzer IRS Peak-Up bands.
    pub const IRS_PU: [Self; 2] = [Self::IrsPuBlue, Self::IrsPuRed];

    // -----------------------------------------------------------------------
    // Parsing
    // -----------------------------------------------------------------------

    /// Parse a band name string into a `Band`.
    ///
    /// Matching is **case-sensitive** and trims whitespace.
    /// Case distinguishes photometric systems: `"r"` = SDSS r, `"R"` = Cousins R.
    /// Unrecognised names produce `Band::Unknown(...)`.
    ///
    /// Recognised names -- Johnson-Cousins: `V`, `U`, `B`, `R`, `I`;
    /// 2MASS: `J`, `H`, `Ks` (also `K`); Y band: `Y`;
    /// SDSS: `g`, `r`, `i`, `z`;
    /// Pan-STARRS: `w`, `y`; ATLAS: `o`, `c`;
    /// Gaia: `G` (also `Gaia_G`), `Gb` (also `Gaia_BP`), `Gr` (also `Gaia_RP`);
    /// WISE: `W1`-`W4`; NEOS: `NEOS1`-`NEOS2`;
    /// Spitzer IRAC: `IRAC1`-`IRAC4`; Spitzer MIPS: `MIPS24`/`MIPS70`/`MIPS160`;
    /// Spitzer IRS Peak-Up: `IRS Peak-Up Blue`, `IRS Peak-Up Red`.
    ///
    /// MPC codes `u`, `C`, `L`, `W` have no calibration and resolve to `Unknown`.
    #[must_use]
    pub fn from_name(s: &str) -> Self {
        match s.trim() {
            "V" | "Vj" => Self::V,
            "U" | "Uj" => Self::U,
            "B" | "Bj" => Self::B,
            "R" | "Rc" => Self::R,
            "I" | "Ic" => Self::I,
            "J" => Self::J,
            "H" => Self::H,
            "Ks" | "K" => Self::Ks,
            "Y" => Self::Y,
            "u" | "Su" => Self::SdssU,
            "g" | "Sg" => Self::SdssG,
            "r" | "Sr" => Self::SdssR,
            "i" | "Si" => Self::SdssI,
            "z" | "Sz" => Self::SdssZ,
            "Pg" => Self::PanStarrsG,
            "Pr" => Self::PanStarrsR,
            "Pi" => Self::PanStarrsI,
            "Pz" => Self::PanStarrsZ,
            "w" | "Pw" => Self::PanStarrsW,
            "y" | "Py" => Self::PanStarrsY,
            "Lu" => Self::LsstU,
            "Lg" => Self::LsstG,
            "Lr" => Self::LsstR,
            "Li" => Self::LsstI,
            "Lz" => Self::LsstZ,
            "Ly" => Self::LsstY,
            "VR" => Self::DecamVr,
            "o" | "Ao" => Self::AtlasO,
            "c" | "Ac" => Self::AtlasC,
            "G" | "Gaia_G" => Self::GaiaG,
            "Gb" | "Gaia_BP" => Self::GaiaBP,
            "Gr" | "Gaia_RP" => Self::GaiaRP,
            "W1" => Self::W1,
            "W2" => Self::W2,
            "W3" => Self::W3,
            "W4" => Self::W4,
            "NEOS1" => Self::Neos1,
            "NEOS2" => Self::Neos2,
            "IRAC1" => Self::Irac1,
            "IRAC2" => Self::Irac2,
            "IRAC3" => Self::Irac3,
            "IRAC4" => Self::Irac4,
            "MIPS24" => Self::Mips24,
            "MIPS70" => Self::Mips70,
            "MIPS160" => Self::Mips160,
            "IRS Peak-Up Blue" => Self::IrsPuBlue,
            "IRS Peak-Up Red" => Self::IrsPuRed,
            _ => {
                let bytes = s.as_bytes();
                let mut arr = [0_u8; 8];
                arr[..bytes.len().min(8)].copy_from_slice(&bytes[..bytes.len().min(8)]);
                Self::Unknown(arr)
            }
        }
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Canonical short name for this band.
    ///
    /// For known variants this is the standard label (e.g. `"W1"`, `"V"`).
    /// For `Unknown` it is the stored label, which may be empty.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::V => "V",
            Self::U => "U",
            Self::B => "B",
            Self::R => "R",
            Self::I => "I",
            Self::J => "J",
            Self::H => "H",
            Self::Ks => "Ks",
            Self::Y => "Y",
            Self::SdssU => "u",
            Self::SdssG => "g",
            Self::SdssR => "r",
            Self::SdssI => "i",
            Self::SdssZ => "z",
            Self::PanStarrsG => "Pg",
            Self::PanStarrsR => "Pr",
            Self::PanStarrsI => "Pi",
            Self::PanStarrsZ => "Pz",
            Self::PanStarrsW => "Pw",
            Self::PanStarrsY => "Py",
            Self::LsstU => "Lu",
            Self::LsstG => "Lg",
            Self::LsstR => "Lr",
            Self::LsstI => "Li",
            Self::LsstZ => "Lz",
            Self::LsstY => "Ly",
            Self::DecamVr => "VR",
            Self::AtlasO => "o",
            Self::AtlasC => "c",
            Self::GaiaG => "G",
            Self::GaiaBP => "Gb",
            Self::GaiaRP => "Gr",
            Self::W1 => "W1",
            Self::W2 => "W2",
            Self::W3 => "W3",
            Self::W4 => "W4",
            Self::Neos1 => "NEOS1",
            Self::Neos2 => "NEOS2",
            Self::Irac1 => "IRAC1",
            Self::Irac2 => "IRAC2",
            Self::Irac3 => "IRAC3",
            Self::Irac4 => "IRAC4",
            Self::Mips24 => "MIPS24",
            Self::Mips70 => "MIPS70",
            Self::Mips160 => "MIPS160",
            Self::IrsPuBlue => "IRS Peak-Up Blue",
            Self::IrsPuRed => "IRS Peak-Up Red",
            Self::Unknown(bytes) => {
                let len = bytes.iter().position(|&b| b == 0).unwrap_or(8);
                std::str::from_utf8(&bytes[..len]).unwrap_or("")
            }
        }
    }

    /// Physical calibration data for this band, or `None` for `Unknown`.
    #[must_use]
    pub fn calibration(&self) -> Option<BandInfo> {
        match self {
            Self::V => Some(BandInfo::V),
            Self::U => Some(BandInfo::U),
            Self::B => Some(BandInfo::B),
            Self::R => Some(BandInfo::R),
            Self::I => Some(BandInfo::I),
            Self::J => Some(BandInfo::J),
            Self::H => Some(BandInfo::H),
            Self::Ks => Some(BandInfo::KS),
            Self::Y => Some(BandInfo::Y),
            Self::SdssU => Some(BandInfo::SDSS_U),
            Self::SdssG => Some(BandInfo::SDSS_G),
            Self::SdssR => Some(BandInfo::SDSS_R),
            Self::SdssI => Some(BandInfo::SDSS_I),
            Self::SdssZ => Some(BandInfo::SDSS_Z),
            Self::PanStarrsG => Some(BandInfo::PAN_STARRS_G),
            Self::PanStarrsR => Some(BandInfo::PAN_STARRS_R),
            Self::PanStarrsI => Some(BandInfo::PAN_STARRS_I),
            Self::PanStarrsZ => Some(BandInfo::PAN_STARRS_Z),
            Self::PanStarrsW => Some(BandInfo::PAN_STARRS_W),
            Self::PanStarrsY => Some(BandInfo::PAN_STARRS_Y),
            Self::LsstU => Some(BandInfo::LSST_U),
            Self::LsstG => Some(BandInfo::LSST_G),
            Self::LsstR => Some(BandInfo::LSST_R),
            Self::LsstI => Some(BandInfo::LSST_I),
            Self::LsstZ => Some(BandInfo::LSST_Z),
            Self::LsstY => Some(BandInfo::LSST_Y),
            Self::DecamVr => Some(BandInfo::DECAM_VR),
            Self::AtlasO => Some(BandInfo::ATLAS_O),
            Self::AtlasC => Some(BandInfo::ATLAS_C),
            Self::GaiaG => Some(BandInfo::GAIA_G),
            Self::GaiaBP => Some(BandInfo::GAIA_BP),
            Self::GaiaRP => Some(BandInfo::GAIA_RP),
            Self::W1 => Some(BandInfo::WISE[0]),
            Self::W2 => Some(BandInfo::WISE[1]),
            Self::W3 => Some(BandInfo::WISE[2]),
            Self::W4 => Some(BandInfo::WISE[3]),
            Self::Neos1 => Some(BandInfo::NEOS[0]),
            Self::Neos2 => Some(BandInfo::NEOS[1]),
            Self::Irac1 => Some(BandInfo::IRAC[0]),
            Self::Irac2 => Some(BandInfo::IRAC[1]),
            Self::Irac3 => Some(BandInfo::IRAC[2]),
            Self::Irac4 => Some(BandInfo::IRAC[3]),
            Self::Mips24 => Some(BandInfo::MIPS[0]),
            Self::Mips70 => Some(BandInfo::MIPS[1]),
            Self::Mips160 => Some(BandInfo::MIPS[2]),
            Self::IrsPuBlue => Some(BandInfo::IRS_PU[0]),
            Self::IrsPuRed => Some(BandInfo::IRS_PU[1]),
            Self::Unknown(_) => None,
        }
    }

    /// True if this band has known calibration data.
    #[must_use]
    pub fn is_known(&self) -> bool {
        !matches!(self, Self::Unknown(_))
    }
}
