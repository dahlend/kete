// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! SPICE frames: their definitions, and the resolution of a frame to an
//! inertial frame.
//!
//! [`SpiceFrames`] gives a frame by its SPICE frame ID. Every frame lookup in
//! kete goes through it.
//!
//! # Layout
//!
//! - This module holds the frame model (`FrameDef`), the built-in frames, and
//!   the source of each link of a chain. `kete_core` resolves the chain.
//! - `text::fk` reads the frame definitions of frames kernels.
//! - `text::pck` holds the body models of text PCK kernels.
//! - [`crate::pck`] and [`crate::ck`] read the binary PCK and CK kernels.
//!
//! # Frame classes
//!
//! A frame has a SPICE frame ID and a name. Its class tells where its
//! orientation comes from. Its class ID is the ID that the kernels of that
//! class store.
//!
//! - Inertial (class 1): the built-in frames J2000, B1950, FK4, GALACTIC,
//!   MARSIAU, ECLIPJ2000 and ECLIPB1950, which `kete_core` resolves without a
//!   definition. kete does not support the other built-in inertial frames, the
//!   frames of old JPL ephemerides (DE-96 to DE-143).
//! - PCK (class 2): a binary PCK segment that holds the class ID, or else the
//!   text PCK constants of the body with that ID. ITRF93 (13000, class ID
//!   3000) and the `IAU_*` body frames are built in.
//! - CK (class 3): CK segments that hold the class ID. Their spacecraft clock
//!   is `CK_<class ID>_SCLK`, or the class ID divided by 1000.
//! - TK (class 4): a fixed rotation relative to another frame.
//!
//! # Names
//!
//! A built-in frame takes precedence over a frames kernel definition with the
//! same name or ID. A name is converted to upper case before the lookup. Thus a
//! kernel variable that spells a name in lower case is not found.
use crate::add_context;
use crate::ck::{CkCollection, LOADED_CK, POINTING_TOLERANCE_DAYS};
use crate::pck::{LOADED_PCK, PckCollection};
use crate::text::fk::tk_definition;
use crate::text::sclk::ClockId;
use crate::text::{LOADED_TEXT_KERNELS, TextKernelVars, TextKernels};
use crossbeam::sync::ShardedLockReadGuard;
use kete_core::errors::{Error, KeteResult};
use kete_core::frames::{DynFrameId, FrameSource, InstantFrame, OffsetFrame};
use kete_core::time::{TDB, Time};

/// The loaded kernels as a [`FrameSource`].
///
/// [`FrameSource::try_frame`] gives a frame by its SPICE frame ID, relative to the
/// equatorial frame.
///
/// The definition of a frame comes from the built-in frames or the loaded frames
/// kernels. An ID that neither defines is a CK frame if a loaded CK has pointing
/// for it. Thus CK files work without a frames kernel.
///
/// The orientation of a frame relative to its reference frame comes from one of
/// these sources:
///
/// - the binary PCK segments that cover the time, else the text PCK constants of
///   the body;
/// - the CK segments, which must hold pointing within about 1 ms of the time;
/// - the fixed rotation of a TK frame.
///
/// A frame resolves through its reference frames, until a static inertial frame.
/// The result has a rotation rate only if every frame in the chain has one. A CK
/// segment without angular velocity gives no rate. The result is at the requested
/// time; [`Self::ck_pointing_time`] gives the time of the CK pointing it uses.
///
/// It is the frame counterpart of
/// [`SpiceEphemeris`](crate::ephemeris::SpiceEphemeris), which gives the states. As
/// that does for the SPK files, it holds read guards on the text, PCK and CK kernels
/// for its life, so those kernels cannot load while it exists. Take one for a
/// propagation or a set of lookups and drop it afterward.
///
/// # Errors
/// The lookups give:
/// - [`Error::ValueError`] if a frame in the chain is undefined or of an
///   unsupported class, or if its definition is malformed.
/// - [`Error::Bounds`] if a PCK or CK frame in the chain has no data at the time.
/// - The errors of chain resolution, see [`FrameSource::try_frame`].
pub struct SpiceFrames {
    text: ShardedLockReadGuard<'static, TextKernels>,
    pck: ShardedLockReadGuard<'static, PckCollection>,
    ck: ShardedLockReadGuard<'static, CkCollection>,
}

impl SpiceFrames {
    /// Read guards on the loaded text, PCK and CK kernels.
    ///
    /// # Errors
    /// [`Error::LockFailed`] if a kernel singleton cannot be read.
    pub fn loaded() -> KeteResult<Self> {
        Ok(Self {
            text: LOADED_TEXT_KERNELS.try_read()?,
            pck: LOADED_PCK.try_read()?,
            ck: LOADED_CK.try_read()?,
        })
    }

    /// The time of the CK pointing used for frame `frame_id` at `time`, or `None` if
    /// `frame_id` is not a CK frame.
    ///
    /// A resolved frame is at `time`. A CK frame uses the pointing nearest `time`,
    /// within about 1 ms of it, and this gives the time of that pointing.
    ///
    /// # Errors
    /// - [`Error::ValueError`] if the frame is undefined or of an unsupported class,
    ///   or if its definition is malformed.
    /// - [`Error::Bounds`] if a CK frame has no pointing within about 1 ms of
    ///   `time`.
    pub fn ck_pointing_time(
        &self,
        frame_id: DynFrameId,
        time: Time<TDB>,
    ) -> KeteResult<Option<Time<TDB>>> {
        if frame_id.static_rotation_to_equatorial().is_some() {
            return Ok(None);
        }
        let kernels = self.kernels();
        match kernels.definition(frame_id)? {
            FrameDef::Ck { class_id, clock_id } => Ok(Some(
                kernels
                    .ck_pointing(frame_id, class_id, clock_id, time)?
                    .time,
            )),
            FrameDef::Pck { .. } | FrameDef::Tk(_) => Ok(None),
        }
    }

    /// The held kernels.
    fn kernels(&self) -> Kernels<'_> {
        Kernels {
            text: &self.text,
            pck: &self.pck,
            ck: &self.ck,
        }
    }
}

impl std::fmt::Debug for SpiceFrames {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpiceFrames").finish_non_exhaustive()
    }
}

impl FrameSource for SpiceFrames {
    fn try_link(&self, frame_id: DynFrameId, time: Time<TDB>) -> KeteResult<InstantFrame> {
        self.kernels().try_link(frame_id, time)
    }
}

/// Where the orientation of a frame comes from.
///
/// The static inertial frames have no definition: `kete_core` resolves them.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum FrameDef {
    /// A PCK frame: binary PCK segments holding the class ID, or else the text
    /// PCK constants of the body with that ID.
    Pck {
        /// Class ID of the frame in binary PCK segments, and the body ID of its
        /// text PCK constants.
        class_id: i32,
    },

    /// A CK frame, with the class ID its segments store and its spacecraft
    /// clock.
    Ck {
        /// Class ID of the frame in CK segments.
        class_id: i32,
        /// ID of the spacecraft clock the CK segments use.
        clock_id: ClockId,
    },

    /// A fixed rotation relative to another frame.
    Tk(OffsetFrame),
}

/// The ID of the frame called `name`, matched in upper case.
///
/// # Errors
/// [`Error::ValueError`] if no built-in frame or loaded frames kernel has the
/// name.
pub(crate) fn id_from_name(vars: &TextKernelVars, name: &str) -> KeteResult<DynFrameId> {
    let name = name.trim().to_uppercase();
    if let Some(id) = built_in_id(&name) {
        return Ok(id);
    }
    vars.integer(&format!("FRAME_{name}"))?
        .map(DynFrameId)
        .ok_or_else(|| Error::ValueError(format!("No frame is called {name}.")))
}

/// The name of frame `id`, if it is built in or defined in a loaded frames
/// kernel.
#[must_use]
pub(crate) fn frame_name(vars: &TextKernelVars, id: DynFrameId) -> Option<String> {
    if let Some(name) = built_in_name(id) {
        return Some(name.to_string());
    }
    vars.string(&format!("FRAME_{id}_NAME"))
        .ok()
        .flatten()
        .map(str::to_string)
}

/// The definition of a built-in frame `id`, or `None` if `id` is not built in
/// or is a static inertial frame.
///
/// The built-in `IAU_*` frames are text PCK body frames. `EARTH_FIXED` is a
/// built-in TK frame. Its definition comes from the `TKFRAME_*` variables of a
/// loaded frames kernel.
pub(crate) fn built_in(vars: &TextKernelVars, id: DynFrameId) -> Option<KeteResult<FrameDef>> {
    if id.static_rotation_to_equatorial().is_some() {
        return None;
    }
    if let Some((name, _, def)) = BUILT_IN.iter().find(|(_, i, _)| *i == id) {
        return Some(def.clone().ok_or_else(|| {
            Error::ValueError(format!(
                "Built-in frame {name} ({id}) is not supported. Supported inertial frames \
                 are J2000, B1950, FK4, GALACTIC, MARSIAU, ECLIPJ2000 and ECLIPB1950."
            ))
        }));
    }
    if let Some((_, _, body)) = built_in_iau(id) {
        return Some(Ok(FrameDef::Pck { class_id: *body }));
    }
    (id == EARTH_FIXED).then(|| tk_definition(vars, id, "EARTH_FIXED", Some("EARTH_FIXED")))
}

/// The ID of the built-in frame called `name`, in upper case.
fn built_in_id(name: &str) -> Option<DynFrameId> {
    BUILT_IN
        .iter()
        .find(|(n, _, _)| *n == name)
        .map(|(_, id, _)| *id)
        .or_else(|| {
            BUILT_IN_IAU
                .iter()
                .find(|(n, _, _)| *n == name)
                .map(|(_, id, _)| DynFrameId(*id))
        })
        .or_else(|| (name == "EARTH_FIXED").then_some(EARTH_FIXED))
}

/// The name of the built-in frame `id`.
fn built_in_name(id: DynFrameId) -> Option<&'static str> {
    BUILT_IN
        .iter()
        .find(|(_, i, _)| *i == id)
        .map(|(n, _, _)| *n)
        .or_else(|| built_in_iau(id).map(|(n, _, _)| *n))
        .or_else(|| (id == EARTH_FIXED).then_some("EARTH_FIXED"))
}

/// The entry of [`BUILT_IN_IAU`] for frame `id`, which the table holds sorted
/// by ID.
fn built_in_iau(id: DynFrameId) -> Option<&'static (&'static str, i32, i32)> {
    BUILT_IN_IAU
        .binary_search_by_key(&id.0, |(_, i, _)| *i)
        .ok()
        .map(|idx| &BUILT_IN_IAU[idx])
}

/// The kernels a frame lookup reads, as a [`FrameSource`].
#[derive(Debug, Clone, Copy)]
struct Kernels<'a> {
    /// Text kernels: frame definitions, text PCK constants and clocks.
    text: &'a TextKernels,
    /// Binary PCK kernels.
    pck: &'a PckCollection,
    /// CK kernels.
    ck: &'a CkCollection,
}

impl Kernels<'_> {
    /// The definition of frame `frame_id`: from the built-in frames or the loaded
    /// frames kernels, else a CK frame if a loaded CK has pointing for it.
    ///
    /// # Errors
    /// [`Error::ValueError`] if the frame is undefined or of an unsupported class,
    /// or if its definition is malformed.
    fn definition(&self, frame_id: DynFrameId) -> KeteResult<FrameDef> {
        match self.text.frame_definition(frame_id)? {
            Some(def) => Ok(def),
            None if self.ck.has_instrument(frame_id.0) => Ok(FrameDef::Ck {
                class_id: frame_id.0,
                clock_id: self.text.ck_clock_id(frame_id.0)?,
            }),
            None => Err(Error::ValueError(format!(
                "Frame {frame_id} is not defined: it is not built in, no loaded frames \
                 kernel defines it, and no loaded CK has pointing for it."
            ))),
        }
    }

    /// The name and ID of frame `frame_id`, to label messages.
    fn label(&self, frame_id: DynFrameId) -> String {
        self.text.frame_name(frame_id).map_or_else(
            || frame_id.to_string(),
            |name| format!("{name} ({frame_id})"),
        )
    }

    /// The CK pointing of frame `frame_id` nearest `time`, at the time of the
    /// pointing.
    ///
    /// # Errors
    /// [`Error::Bounds`] if no pointing is within about 1 ms of `time`, and the
    /// errors of the clock and of [`CkCollection::try_get_pointing`].
    fn ck_pointing(
        &self,
        frame_id: DynFrameId,
        class_id: i32,
        clock_id: ClockId,
        time: Time<TDB>,
    ) -> KeteResult<InstantFrame> {
        let context = |err: Error| {
            add_context(
                err,
                &format!("Frame {} at JD {}", self.label(frame_id), time.jd()),
            )
        };
        let clock = self.text.clock(clock_id).map_err(context)?;
        let frame = self
            .ck
            .try_get_pointing(time, class_id, clock)
            .map_err(context)?;
        if (frame.time - time).elapsed.abs() > POINTING_TOLERANCE_DAYS {
            return Err(Error::Bounds(format!(
                "CK frame {} has no pointing at JD {}.",
                self.label(frame_id),
                time.jd()
            )));
        }
        Ok(frame)
    }
}

impl FrameSource for Kernels<'_> {
    /// The frame `frame_id` at `time` relative to the frame its definition
    /// names, see [`SpiceFrames`].
    fn try_link(&self, frame_id: DynFrameId, time: Time<TDB>) -> KeteResult<InstantFrame> {
        let Self { text, pck, .. } = *self;
        let context = |err: Error| {
            add_context(
                err,
                &format!("Frame {} at JD {}", self.label(frame_id), time.jd()),
            )
        };
        match self.definition(frame_id)? {
            // Binary PCK data takes precedence over text PCK data where it covers
            // the time.
            FrameDef::Pck { class_id } => match pck.try_get_orientation(class_id, time) {
                Ok(frame) => Ok(frame),
                Err(Error::Bounds(msg)) => match text.body_rotation(class_id).map_err(context)? {
                    Some(model) => Ok(model.frame(time)),
                    None if pck.has_frame(class_id) => Err(context(Error::Bounds(msg))),
                    None => Err(Error::Bounds(format!(
                        "PCK frame {} has no binary PCK segment with class ID \
                         {class_id}, and no loaded text PCK gives BODY{class_id}_POLE_RA.",
                        self.label(frame_id)
                    ))),
                },
                Err(err) => Err(context(err)),
            },
            FrameDef::Ck { class_id, clock_id } => {
                self.ck_pointing(frame_id, class_id, clock_id, time)
            }
            FrameDef::Tk(tk) => Ok(tk.relative_to_reference(time)),
        }
    }
}

/// The SPICE frame ID of `EARTH_FIXED`, a built-in TK frame defined by a frames
/// kernel.
const EARTH_FIXED: DynFrameId = DynFrameId(10081);

/// Built-in frames: name, SPICE frame ID, and definition (`None` for a static
/// inertial frame, which has none, or a frame kete does not support).
const BUILT_IN: [(&str, DynFrameId, Option<FrameDef>); 22] = [
    ("J2000", DynFrameId::J2000, None),
    ("B1950", DynFrameId::B1950, None),
    ("FK4", DynFrameId::FK4, None),
    ("DE-118", DynFrameId(4), None),
    ("DE-96", DynFrameId(5), None),
    ("DE-102", DynFrameId(6), None),
    ("DE-108", DynFrameId(7), None),
    ("DE-111", DynFrameId(8), None),
    ("DE-114", DynFrameId(9), None),
    ("DE-122", DynFrameId(10), None),
    ("DE-125", DynFrameId(11), None),
    ("DE-130", DynFrameId(12), None),
    ("GALACTIC", DynFrameId::GALACTIC, None),
    ("DE-200", DynFrameId(14), None),
    ("DE-202", DynFrameId(15), None),
    ("MARSIAU", DynFrameId::MARSIAU, None),
    ("ECLIPJ2000", DynFrameId::ECLIPJ2000, None),
    ("ECLIPB1950", DynFrameId::ECLIPB1950, None),
    ("DE-140", DynFrameId(19), None),
    ("DE-142", DynFrameId(20), None),
    ("DE-143", DynFrameId(21), None),
    (
        "ITRF93",
        DynFrameId::ITRF93,
        Some(FrameDef::Pck { class_id: 3000 }),
    ),
];

/// Built-in body-fixed frames from text PCK constants.
///
/// Each entry is the name, the SPICE frame ID, and the NAIF ID of the body
/// whose `BODY<id>_*` constants orient it. The entries are the built-in frames
/// of CSPICE N0067, sorted by frame ID for [`built_in_iau`].
const BUILT_IN_IAU: [(&str, i32, i32); 122] = [
    ("IAU_MERCURY_BARYCENTER", 10001, 1),
    ("IAU_VENUS_BARYCENTER", 10002, 2),
    ("IAU_EARTH_BARYCENTER", 10003, 3),
    ("IAU_MARS_BARYCENTER", 10004, 4),
    ("IAU_JUPITER_BARYCENTER", 10005, 5),
    ("IAU_SATURN_BARYCENTER", 10006, 6),
    ("IAU_URANUS_BARYCENTER", 10007, 7),
    ("IAU_NEPTUNE_BARYCENTER", 10008, 8),
    ("IAU_PLUTO_BARYCENTER", 10009, 9),
    ("IAU_SUN", 10010, 10),
    ("IAU_MERCURY", 10011, 199),
    ("IAU_VENUS", 10012, 299),
    ("IAU_EARTH", 10013, 399),
    ("IAU_MARS", 10014, 499),
    ("IAU_JUPITER", 10015, 599),
    ("IAU_SATURN", 10016, 699),
    ("IAU_URANUS", 10017, 799),
    ("IAU_NEPTUNE", 10018, 899),
    ("IAU_PLUTO", 10019, 999),
    ("IAU_MOON", 10020, 301),
    ("IAU_PHOBOS", 10021, 401),
    ("IAU_DEIMOS", 10022, 402),
    ("IAU_IO", 10023, 501),
    ("IAU_EUROPA", 10024, 502),
    ("IAU_GANYMEDE", 10025, 503),
    ("IAU_CALLISTO", 10026, 504),
    ("IAU_AMALTHEA", 10027, 505),
    ("IAU_HIMALIA", 10028, 506),
    ("IAU_ELARA", 10029, 507),
    ("IAU_PASIPHAE", 10030, 508),
    ("IAU_SINOPE", 10031, 509),
    ("IAU_LYSITHEA", 10032, 510),
    ("IAU_CARME", 10033, 511),
    ("IAU_ANANKE", 10034, 512),
    ("IAU_LEDA", 10035, 513),
    ("IAU_THEBE", 10036, 514),
    ("IAU_ADRASTEA", 10037, 515),
    ("IAU_METIS", 10038, 516),
    ("IAU_MIMAS", 10039, 601),
    ("IAU_ENCELADUS", 10040, 602),
    ("IAU_TETHYS", 10041, 603),
    ("IAU_DIONE", 10042, 604),
    ("IAU_RHEA", 10043, 605),
    ("IAU_TITAN", 10044, 606),
    ("IAU_HYPERION", 10045, 607),
    ("IAU_IAPETUS", 10046, 608),
    ("IAU_PHOEBE", 10047, 609),
    ("IAU_JANUS", 10048, 610),
    ("IAU_EPIMETHEUS", 10049, 611),
    ("IAU_HELENE", 10050, 612),
    ("IAU_TELESTO", 10051, 613),
    ("IAU_CALYPSO", 10052, 614),
    ("IAU_ATLAS", 10053, 615),
    ("IAU_PROMETHEUS", 10054, 616),
    ("IAU_PANDORA", 10055, 617),
    ("IAU_ARIEL", 10056, 701),
    ("IAU_UMBRIEL", 10057, 702),
    ("IAU_TITANIA", 10058, 703),
    ("IAU_OBERON", 10059, 704),
    ("IAU_MIRANDA", 10060, 705),
    ("IAU_CORDELIA", 10061, 706),
    ("IAU_OPHELIA", 10062, 707),
    ("IAU_BIANCA", 10063, 708),
    ("IAU_CRESSIDA", 10064, 709),
    ("IAU_DESDEMONA", 10065, 710),
    ("IAU_JULIET", 10066, 711),
    ("IAU_PORTIA", 10067, 712),
    ("IAU_ROSALIND", 10068, 713),
    ("IAU_BELINDA", 10069, 714),
    ("IAU_PUCK", 10070, 715),
    ("IAU_TRITON", 10071, 801),
    ("IAU_NEREID", 10072, 802),
    ("IAU_NAIAD", 10073, 803),
    ("IAU_THALASSA", 10074, 804),
    ("IAU_DESPINA", 10075, 805),
    ("IAU_GALATEA", 10076, 806),
    ("IAU_LARISSA", 10077, 807),
    ("IAU_PROTEUS", 10078, 808),
    ("IAU_CHARON", 10079, 901),
    ("IAU_PAN", 10082, 618),
    ("IAU_GASPRA", 10083, 9511010),
    ("IAU_IDA", 10084, 2431010),
    ("IAU_EROS", 10085, 2000433),
    ("IAU_CALLIRRHOE", 10086, 517),
    ("IAU_THEMISTO", 10087, 518),
    ("IAU_MEGACLITE", 10088, 519),
    ("IAU_TAYGETE", 10089, 520),
    ("IAU_CHALDENE", 10090, 521),
    ("IAU_HARPALYKE", 10091, 522),
    ("IAU_KALYKE", 10092, 523),
    ("IAU_IOCASTE", 10093, 524),
    ("IAU_ERINOME", 10094, 525),
    ("IAU_ISONOE", 10095, 526),
    ("IAU_PRAXIDIKE", 10096, 527),
    ("IAU_BORRELLY", 10097, 1000005),
    ("IAU_TEMPEL_1", 10098, 1000093),
    ("IAU_VESTA", 10099, 2000004),
    ("IAU_ITOKAWA", 10100, 2025143),
    ("IAU_CERES", 10101, 2000001),
    ("IAU_PALLAS", 10102, 2000002),
    ("IAU_LUTETIA", 10103, 2000021),
    ("IAU_DAVIDA", 10104, 2000511),
    ("IAU_STEINS", 10105, 2002867),
    ("IAU_BENNU", 10106, 2101955),
    ("IAU_52_EUROPA", 10107, 2000052),
    ("IAU_NIX", 10108, 902),
    ("IAU_HYDRA", 10109, 903),
    ("IAU_RYUGU", 10110, 2162173),
    ("IAU_ARROKOTH", 10111, 2486958),
    ("IAU_DIDYMOS_BARYCENTER", 10112, 20065803),
    ("IAU_DIDYMOS", 10113, 920065803),
    ("IAU_DIMORPHOS", 10114, 120065803),
    ("IAU_DONALDJOHANSON", 10115, 20052246),
    ("IAU_EURYBATES", 10116, 920003548),
    ("IAU_EURYBATES_BARYCENTER", 10117, 20003548),
    ("IAU_QUETA", 10118, 120003548),
    ("IAU_POLYMELE", 10119, 20015094),
    ("IAU_LEUCUS", 10120, 20011351),
    ("IAU_ORUS", 10121, 20021900),
    ("IAU_PATROCLUS_BARYCENTER", 10122, 20000617),
    ("IAU_PATROCLUS", 10123, 920000617),
    ("IAU_MENOETIUS", 10124, 120000617),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ck::tests::{
        SPIN_JD0, TEST_CLOCK, rate_by_difference, spinning_ck_file, test_clock,
    };
    use crate::text::fk::tests::{TK_CHAIN, TK_CHAIN_EXPECTED};
    use kete_core::frames::{Ecliptic, Equatorial, StaticFrame};
    use nalgebra::Matrix3;

    /// The frame `frame_id` at `time` from the given kernels.
    fn resolve(
        text: &TextKernels,
        pck: &PckCollection,
        ck: &CkCollection,
        frame_id: DynFrameId,
        time: Time<TDB>,
    ) -> KeteResult<InstantFrame<Equatorial>> {
        Kernels { text, pck, ck }.try_frame(frame_id, time)
    }

    fn loaded(text: &str) -> TextKernels {
        let mut frames = TextKernels::default();
        frames.load_text(text).unwrap();
        frames
    }

    /// Names resolve in upper case, built-ins first; IDs give back their names.
    #[test]
    fn names_and_ids() {
        let frames = loaded(TK_CHAIN);
        assert_eq!(frames.frame_id("tk_ang").unwrap(), DynFrameId(1_400_202));
        assert_eq!(frames.frame_id(" J2000 ").unwrap(), DynFrameId::J2000);
        assert_eq!(frames.frame_id("itrf93").unwrap(), DynFrameId::ITRF93);
        assert!(frames.frame_id("NOPE").is_err());
        assert_eq!(
            frames.frame_name(DynFrameId(1_400_203)).as_deref(),
            Some("TK_Q")
        );
        assert_eq!(
            frames.frame_name(DynFrameId(17)).as_deref(),
            Some("ECLIPJ2000")
        );

        // A built-in frame cannot be redefined by a frames kernel.
        let frames = loaded(
            "\\begindata\nFRAME_ITRF93 = 1400010\nFRAME_13000_CLASS = 4\n\
             FRAME_13000_CLASS_ID = 13000\n",
        );
        assert_eq!(frames.frame_id("ITRF93").unwrap(), DynFrameId::ITRF93);
        assert_eq!(
            frames.frame_definition(DynFrameId::ITRF93).unwrap(),
            Some(FrameDef::Pck { class_id: 3000 })
        );
    }

    /// The ID lookups of the built-in tables agree with their contents.
    #[test]
    fn built_in_tables() {
        assert!(BUILT_IN_IAU.windows(2).all(|w| w[0].1 < w[1].1));
        for (name, id, _) in BUILT_IN_IAU {
            assert_eq!(built_in_name(DynFrameId(id)), Some(name));
        }
        // The static inertial frames have names but no definition.
        let vars = TextKernelVars::default();
        for (name, id, def) in BUILT_IN {
            let is_static = id.static_rotation_to_equatorial().is_some();
            assert!(!(is_static && def.is_some()), "{name}");
            assert_eq!(built_in(&vars, id).is_none(), is_static, "{name}");
        }
    }

    /// The built-in IAU frames are text PCK frames of their body, and
    /// `EARTH_FIXED` is a TK frame that a frames kernel defines.
    #[test]
    fn built_in_body_frames() {
        let frames = loaded(
            "\\begindata\nTKFRAME_EARTH_FIXED_RELATIVE = 'ITRF93'\n\
             TKFRAME_EARTH_FIXED_SPEC = 'MATRIX'\n\
             TKFRAME_EARTH_FIXED_MATRIX = ( 1 0 0 0 1 0 0 0 1 )\n",
        );
        let id = frames.frame_id("iau_earth").unwrap();
        assert_eq!(id, DynFrameId(10013));
        assert_eq!(frames.frame_name(id).as_deref(), Some("IAU_EARTH"));
        assert_eq!(
            frames.frame_definition(id).unwrap(),
            Some(FrameDef::Pck { class_id: 399 })
        );
        let id = frames.frame_id("EARTH_FIXED").unwrap();
        let Some(FrameDef::Tk(tk)) = frames.frame_definition(id).unwrap() else {
            panic!("not a TK frame");
        };
        assert_eq!(tk.reference, DynFrameId::ITRF93);
        assert!(loaded("").frame_definition(id).is_err());
    }

    /// Text kernels holding the test clock and `text`.
    fn frames_from(text: &str) -> TextKernels {
        let mut kernels = TextKernels::default();
        kernels.load_text(TEST_CLOCK).unwrap();
        kernels.load_text(text).unwrap();
        kernels
    }

    /// A chain of TK frames of each specification, ending in ECLIPJ2000 or
    /// J2000, matches reference rotations.
    #[test]
    fn tk_chain_matches_reference() {
        let frames = frames_from(TK_CHAIN);
        let (pck, ck) = (PckCollection::default(), CkCollection::default());
        let time = Time::<TDB>::new(2_451_545.0);
        for (name, expected) in TK_CHAIN_EXPECTED {
            let id = frames.frame_id(name).unwrap();
            let frame = resolve(&frames, &pck, &ck, id, time).unwrap();
            let (rot, rate) = (frame.rotation, frame.rotation_rate.unwrap());
            let expected = Matrix3::from_fn(|i, j| expected[i][j]);
            let err = (rot.matrix() - expected).abs().max();
            assert!(err < 1e-15, "{name}: {err:e}");
            assert_eq!(rate, Matrix3::zeros(), "{name}");
        }
    }

    /// A CK camera relative to a TK mount on a spinning CK spacecraft: the
    /// chain composes the three rotations, and its rate is the derivative of
    /// the result. The camera is in no frames kernel, so it resolves as a CK
    /// frame by its ID.
    #[test]
    fn ck_tk_ck_chain() {
        let mut ck = CkCollection::default();
        ck.load_file(&spinning_ck_file("resolve_base", -999_020, 1))
            .unwrap();
        ck.load_file(&spinning_ck_file("resolve_camera", -999_021, 1_400_501))
            .unwrap();
        let frames = frames_from(
            "\\begindata\n\
             FRAME_BASE = -999020\nFRAME_-999020_NAME = 'BASE'\n\
             FRAME_-999020_CLASS = 3\nFRAME_-999020_CLASS_ID = -999020\n\
             FRAME_MOUNT = 1400501\nFRAME_1400501_NAME = 'MOUNT'\n\
             FRAME_1400501_CLASS = 4\nFRAME_1400501_CLASS_ID = 1400501\n\
             TKFRAME_MOUNT_RELATIVE = 'BASE'\nTKFRAME_MOUNT_SPEC = 'ANGLES'\n\
             TKFRAME_MOUNT_UNITS = 'DEGREES'\nTKFRAME_MOUNT_AXES = ( 1 2 3 )\n\
             TKFRAME_MOUNT_ANGLES = ( 20 -35 50 )\n",
        );
        let pck = PckCollection::default();
        let resolve = |id: i32, jd: f64| {
            let frame = resolve(&frames, &pck, &ck, DynFrameId(id), Time::new(jd)).unwrap();
            (frame.rotation, frame.rotation_rate.unwrap())
        };
        let jd = SPIN_JD0 + 0.25;
        let Some(FrameDef::Tk(OffsetFrame {
            rotation: mount, ..
        })) = frames.frame_definition(DynFrameId(1_400_501)).unwrap()
        else {
            panic!("not a TK frame");
        };
        let camera = ck
            .try_get_pointing(Time::new(jd), -999_021, test_clock())
            .unwrap();
        let (base, _) = resolve(-999_020, jd);
        let (rot, rate) = resolve(-999_021, jd);
        let err = (rot.matrix() - (base * mount * camera.rotation).matrix())
            .abs()
            .max();
        assert!(err < 1e-15, "rotation error {err:e}");
        let numeric = rate_by_difference(|jd| *resolve(-999_021, jd).0.matrix(), jd, 1e-3);
        let err = (rate - numeric).abs().max();
        assert!(err < 1e-4, "rotation rate error {err:e}");
    }

    /// `CK_<id>_SCLK` names the clock of a CK frame, rather than the ID divided
    /// by 1000.
    #[test]
    fn ck_clock_from_frames_kernel() {
        let mut ck = CkCollection::default();
        ck.load_file(&spinning_ck_file("resolve_clock", -1_999_012, 1))
            .unwrap();
        let pck = PckCollection::default();
        let time = Time::new(SPIN_JD0 + 0.25);
        let id = DynFrameId(-1_999_012);
        let without = frames_from("");
        assert!(resolve(&without, &pck, &ck, id, time).is_err());
        let with = frames_from("\\begindata\nCK_-1999012_SCLK = -999\n");
        assert!(resolve(&with, &pck, &ck, id, time).is_ok());
    }

    /// Write a type 3 CK file, without angular velocity, for `ck_id` relative
    /// to `reference`: a rotation about z from 0 at `SPIN_JD0` to 0.2 rad a day
    /// later, on the test clock. Return its path.
    fn type3_file(name: &str, ck_id: i32, reference: i32) -> String {
        use crate::ck::type3::CkSegmentType3;
        use crate::daf::DafFile;
        let mut records = Vec::new();
        let mut times = Vec::new();
        for (dt, angle) in [(0.0, 0.0), (1.0, 0.2)] {
            let q = nalgebra::UnitQuaternion::from_axis_angle(&nalgebra::Vector3::z_axis(), angle);
            records.extend_from_slice(&[q.w, q.i, q.j, q.k]);
            times.push(test_clock().time_to_tick(Time::new(SPIN_JD0 + dt)).unwrap());
        }
        let array =
            CkSegmentType3::new_array(ck_id, reference, &records, &times, &times[..1], false, name)
                .unwrap();
        let mut daf = DafFile::new_ck("type 3 test", "");
        daf.arrays.push(array.daf);
        let path = std::env::temp_dir().join(format!("kete_resolve_{name}.bc"));
        daf.write_to(&mut std::fs::File::create(&path).unwrap())
            .unwrap();
        path.to_str().unwrap().to_string()
    }

    /// A type 3 segment reads its record times on the clock `CK_<id>_SCLK`
    /// names too.
    #[test]
    fn type3_record_times_use_the_named_clock() {
        let mut ck = CkCollection::default();
        ck.load_file(&type3_file("clock", -1_999_013, 1)).unwrap();
        let frames = frames_from("\\begindata\nCK_-1999013_SCLK = -999\n");
        let frame = resolve(
            &frames,
            &PckCollection::default(),
            &ck,
            DynFrameId(-1_999_013),
            Time::new(SPIN_JD0 + 0.5),
        )
        .unwrap();
        assert!((frame.rotation.angle() - 0.1).abs() < 1e-12);
    }

    /// A CK frame without angular velocity on a spinning CK frame: the chain
    /// gives the rotation, and no rate, rather than a rate that leaves out the
    /// missing one.
    #[test]
    fn a_missing_rate_in_the_chain_leaves_no_rate() {
        let mut ck = CkCollection::default();
        ck.load_file(&spinning_ck_file("norate_base", -999_030, 1))
            .unwrap();
        ck.load_file(&type3_file("norate_top", -999_031, -999_030))
            .unwrap();
        let frames = frames_from("");
        let pck = PckCollection::default();
        let time = Time::new(SPIN_JD0 + 0.5);
        let top = resolve(&frames, &pck, &ck, DynFrameId(-999_031), time).unwrap();
        let base = resolve(&frames, &pck, &ck, DynFrameId(-999_030), time).unwrap();
        assert!(base.rotation_rate.is_some());
        assert!(top.rotation_rate.is_none());
        assert!(top.to_equatorial([1.0, 0.0, 0.0], [0.0; 3]).is_err());
        let own = ck.try_get_pointing(time, -999_031, test_clock()).unwrap();
        let expected = base.rotation_to_equatorial() * own.rotation;
        let err = (top.rotation_to_equatorial().matrix() - expected.matrix())
            .abs()
            .max();
        assert!(err < 1e-15, "{err:e}");
    }

    /// A loop of reference frames, an undefined frame, and an unsupported class
    /// are errors.
    #[test]
    fn bad_chains_are_errors() {
        let frames = frames_from(
            "\\begindata\n\
             FRAME_LOOP_A = 1400601\nFRAME_1400601_CLASS = 4\nFRAME_1400601_CLASS_ID = 1400601\n\
             TKFRAME_1400601_RELATIVE = 'LOOP_B'\nTKFRAME_1400601_SPEC = 'MATRIX'\n\
             TKFRAME_1400601_MATRIX = ( 1 0 0 0 1 0 0 0 1 )\n\
             FRAME_LOOP_B = 1400602\nFRAME_1400602_CLASS = 4\nFRAME_1400602_CLASS_ID = 1400602\n\
             TKFRAME_1400602_RELATIVE = 'LOOP_A'\nTKFRAME_1400602_SPEC = 'MATRIX'\n\
             TKFRAME_1400602_MATRIX = ( 1 0 0 0 1 0 0 0 1 )\n\
             FRAME_DYN = 1400603\nFRAME_1400603_CLASS = 5\nFRAME_1400603_CLASS_ID = 1400603\n\
             FRAME_NOPCK = 1400604\nFRAME_1400604_CLASS = 2\nFRAME_1400604_CLASS_ID = 1400604\n",
        );
        let (pck, ck) = (PckCollection::default(), CkCollection::default());
        let time = Time::<TDB>::new(2_451_545.0);
        let err = |id| {
            resolve(&frames, &pck, &ck, DynFrameId(id), time)
                .unwrap_err()
                .to_string()
        };
        assert!(err(1_400_601).contains("cycle"));
        assert!(err(1_400_603).contains("dynamic"));
        assert!(err(1_400_604).contains("text PCK"));
        assert!(err(-987_654_000).contains("not defined"));
        assert!(err(4).contains("not supported"));
    }

    /// The text PCK constants of the Moon from pck00011.tpc, and CSPICE
    /// `sxform('IAU_MOON', 'J2000', 14610 days)` for them: rotation, and rate
    /// per day.
    const MOON_TPC: &str = r"KPL/PCK
\begindata
BODY3_NUT_PREC_ANGLES = ( 125.045 -1935.5364525000 250.089 -3871.0729050000
    260.008 475263.3328725000 176.625 487269.6299850000
    357.529 35999.0509575000 311.589 964468.4993100000
    134.963 477198.8693250000 276.617 12006.3007650000
    34.226 63863.5132425000 15.134 -5806.6093575000
    119.743 131.8406400000 239.961 6003.1503825000
    25.053 473327.7964200000 )
BODY301_POLE_RA = ( 269.9949 0.0031 0. )
BODY301_POLE_DEC = ( 66.5392 0.0130 0. )
BODY301_PM = ( 38.3213 13.17635815 -1.4D-12 )
BODY301_NUT_PREC_RA = ( -3.8787 -0.1204 0.0700 -0.0172
    0.0 0.0072 0.0 0.0
    0.0 -0.0052 0.0 0.0
    0.0043 )
BODY301_NUT_PREC_DEC = ( 1.5419 0.0239 -0.0278 0.0068
    0.0 -0.0029 0.0009 0.0
    0.0 0.0008 0.0 0.0
    -0.0009 )
BODY301_NUT_PREC_PM = ( 3.5610 0.1208 -0.0642 0.0158
    0.0252 -0.0066 -0.0047 -0.0046
    0.0028 0.0052 0.0040 0.0019
    -0.0044 )
\begintext
";
    const MOON_ROT: [[f64; 3]; 3] = [
        [
            5.720_547_105_837_333e-1,
            8.198_082_524_397_307e-1,
            -2.584_254_884_248_154_6e-2,
        ],
        [
            -7.610_737_698_623_49e-1,
            5.187_960_000_245_592e-1,
            -3.893_808_253_959_481_5e-1,
        ],
        [
            -3.058_106_030_314_734_5e-1,
            2.424_152_214_491_336e-1,
            9.207_142_528_946_177e-1,
        ],
    ];
    const MOON_RATE: [[f64; 3]; 3] = [
        [
            1.885_516_499_674_723_4e-1,
            -1.315_698_300_011_999_6e-1,
            -6.688_769_342_283_353e-6,
        ],
        [
            1.193_384_625_668_042_4e-1,
            1.750_255_216_644_558_7e-1,
            -5.863_938_867_503_373e-5,
        ],
        [
            5.570_927_164_739_238e-2,
            7.037_302_263_184_936e-2,
            -2.498_702_321_496_014_6e-5,
        ],
    ];

    /// `IAU_MOON` from text PCK constants, with nutation and precession terms,
    /// matches CSPICE in rotation and rate, to the round-off of its 1.9e5
    /// degree meridian angle.
    #[test]
    fn iau_moon_matches_spice() {
        let frames = frames_from(MOON_TPC);
        let id = frames.frame_id("IAU_MOON").unwrap();
        let time = Time::<TDB>::new(2_451_545.0) + 14_610.0;
        let frame = resolve(
            &frames,
            &PckCollection::default(),
            &CkCollection::default(),
            id,
            time,
        )
        .unwrap();
        let (rot, rate) = (frame.rotation, frame.rotation_rate.unwrap());
        let expected = Matrix3::from_fn(|i, j| MOON_ROT[i][j]);
        let err = (rot.matrix() - expected).abs().max();
        assert!(err < 1e-12, "rotation error {err:e}");
        let expected = Matrix3::from_fn(|i, j| MOON_RATE[i][j]);
        let err = (rate - expected).abs().max();
        assert!(err < 1e-12, "rate error {err:e}");
    }

    /// An inertial frame resolves to itself.
    #[test]
    fn inertial_frame_is_identity() {
        let (frames, pck, ck) = (
            frames_from(""),
            PckCollection::default(),
            CkCollection::default(),
        );
        let frame = resolve(
            &frames,
            &pck,
            &ck,
            DynFrameId::ECLIPJ2000,
            Time::new(2_451_545.0),
        )
        .unwrap();
        assert_eq!(frame.frame, DynFrameId::ECLIPJ2000);
        assert_eq!(&frame.rotation, Ecliptic::rotation_to_equatorial());
    }

    /// ITRF93 is the Earth frame the PCK files hold, class ID 3000. Runs where an
    /// Earth PCK is loaded (the user cache); kernels are not committed.
    #[test]
    fn itrf93_is_the_earth_pck_frame() {
        crate::test_data::ensure_test_spk();
        let time = Time::<TDB>::new(2_460_000.5);
        let Ok(direct) = LOADED_PCK
            .try_read()
            .unwrap()
            .try_get_orientation(3000, time)
        else {
            return;
        };
        let frame = SpiceFrames::loaded()
            .unwrap()
            .try_frame(DynFrameId::ITRF93, time)
            .unwrap();
        let v = nalgebra::Vector3::new(0.3, -0.2, 0.9);
        let (a, da) = frame.to_equatorial(v, nalgebra::Vector3::zeros()).unwrap();
        let (b, db) = direct.to_equatorial(v, nalgebra::Vector3::zeros()).unwrap();
        assert_eq!((a, da), (b, db));
    }

    /// A frames kernel cannot redefine a static inertial frame.
    #[test]
    fn static_frames_are_not_redefined() {
        let frames = loaded(
            "\\begindata\nFRAME_1_CLASS = 4\nFRAME_1_CLASS_ID = 1\n\
             TKFRAME_1_RELATIVE = 'ECLIPJ2000'\nTKFRAME_1_SPEC = 'MATRIX'\n\
             TKFRAME_1_MATRIX = ( 1 0 0 0 1 0 0 0 1 )\n",
        );
        assert_eq!(frames.frame_definition(DynFrameId::J2000).unwrap(), None);
    }
}
