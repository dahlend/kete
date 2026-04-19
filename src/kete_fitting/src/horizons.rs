//! JPL Horizons data representation
use std::fmt::Debug;
#[cfg(feature = "fetch")]
use std::fs;

use crate::UncertainState;
#[cfg(feature = "fetch")]
use kete_core::cache::cache_dir;
use kete_core::constants::GMS_SQRT;
use kete_core::desigs::Desig;
use kete_core::elements::CometElements;
use kete_core::errors::{Error, KeteResult};
use kete_core::forces::NonGravModel;
use kete_core::frames::{Ecliptic, Equatorial};
use kete_core::state::State;
use kete_core::time::Time;
use nalgebra::DMatrix;
#[cfg(feature = "fetch")]
use serde::Deserialize;

/// Physical, orbital, and observational properties of a solar system object
/// as recorded in JPL Horizons.
///
/// All orbital angles (`inclination`, `lon_of_ascending`, `peri_arg`) are in
/// degrees. Distances are in AU. Times are JD TDB.
#[derive(Clone, Debug)]
pub struct HorizonsProperties {
    /// The MPC designation of the object.
    pub desig: String,

    /// An optional group name to associate the object with a group.
    pub group: Option<String>,

    /// The epoch during which the orbital elements listed are accurate, in JD, TDB.
    pub epoch: Option<f64>,

    /// The eccentricity of the orbit.
    pub eccentricity: Option<f64>,

    /// The inclination of the orbit in degrees.
    pub inclination: Option<f64>,

    /// The longitudinal node of the orbit in degrees.
    pub lon_of_ascending: Option<f64>,

    /// The argument of perihelion in degrees.
    pub peri_arg: Option<f64>,

    /// The perihelion distance in AU.
    pub peri_dist: Option<f64>,

    /// The time of perihelion in JD, TDB scaled time.
    pub peri_time: Option<f64>,

    /// The H magnitude of the object.
    pub h_mag: Option<f64>,

    /// The visible albedo of the object, between 0 and 1.
    pub vis_albedo: Option<f64>,

    /// The diameter of the object in km.
    pub diameter: Option<f64>,

    /// The minimum orbital intersection distance between the object and Earth in AU.
    pub moid: Option<f64>,

    /// The g parameter of the object.
    pub g_phase: Option<f64>,

    /// If the object was previously known, this lists the length of time of the
    /// observations of the object in days.
    pub arc_len: Option<f64>,

    /// Uncertain state built from the Horizons covariance (if provided).
    pub uncertain_state: Option<UncertainState>,

    /// Non-gravitational model from Horizons model parameters.
    pub non_grav: Option<NonGravModel>,

    /// Alternate designations for this object.
    pub alternate_desigs: Vec<String>,

    /// Raw JSON response from SBDB, if fetched via API.
    pub raw_json: Option<String>,
}

impl HorizonsProperties {
    /// Construct a new [`HorizonsProperties`].
    ///
    /// Orbital angles (`inclination`, `lon_of_ascending`, `peri_arg`) are in
    /// degrees. `covariance_params` and `covariance_matrix` must both be
    /// provided to build a covariance; if only one is given it is ignored.
    /// `covariance_params` is a list of `(name, value)` pairs matching the
    /// matrix rows/columns. `covariance_epoch` defaults to `epoch` if omitted.
    ///
    /// # Errors
    /// Returns an error if the covariance dimensions are inconsistent.
    #[allow(
        clippy::too_many_arguments,
        reason = "Many optional parameters for Horizons properties"
    )]
    pub fn new(
        desig: String,
        group: Option<String>,
        epoch: Option<f64>,
        eccentricity: Option<f64>,
        inclination: Option<f64>,
        lon_of_ascending: Option<f64>,
        peri_arg: Option<f64>,
        peri_dist: Option<f64>,
        peri_time: Option<f64>,
        h_mag: Option<f64>,
        vis_albedo: Option<f64>,
        diameter: Option<f64>,
        moid: Option<f64>,
        g_phase: Option<f64>,
        arc_len: Option<f64>,
        covariance_params: Option<Vec<(String, f64)>>,
        covariance_matrix: Option<Vec<Vec<f64>>>,
        covariance_epoch: Option<f64>,
    ) -> KeteResult<Self> {
        let uncertain_state = match (covariance_params, covariance_matrix) {
            (Some(params), Some(cov_matrix)) => {
                let cov_epoch = covariance_epoch.or(epoch).unwrap_or(0.0);
                Some(build_uncertain_state(
                    &desig,
                    cov_epoch,
                    &params,
                    &cov_matrix,
                )?)
            }
            _ => None,
        };
        Ok(Self {
            desig,
            group,
            vis_albedo,
            diameter,
            moid,
            peri_dist,
            eccentricity,
            inclination,
            lon_of_ascending,
            peri_arg,
            peri_time,
            h_mag,
            g_phase,
            epoch,
            arc_len,
            uncertain_state,
            non_grav: None,
            alternate_desigs: vec![],
            raw_json: None,
        })
    }

    /// Draw `n_samples` states from the covariance distribution.
    ///
    /// - `n_samples` - Number of samples to draw.
    /// - `seed` - Optional RNG seed for reproducibility.
    ///
    /// # Errors
    /// Returns an error if no covariance is available.
    pub fn sample(
        &self,
        n_samples: usize,
        seed: Option<u64>,
    ) -> KeteResult<Vec<(State<Equatorial>, Option<NonGravModel>)>> {
        match &self.uncertain_state {
            None => Err(Error::ValueError(
                "This object does not have a covariance matrix, cannot sample from it.".into(),
            )),
            Some(us) => us.sample(n_samples, seed),
        }
    }

    /// Cometary orbital elements (q, e, i, node, peri, tp) from this object.
    ///
    /// # Errors
    /// Returns an error if any required element field is absent.
    pub fn elements(&self) -> KeteResult<CometElements> {
        Ok(CometElements {
            desig: Desig::Name(self.desig.clone()),
            epoch: self
                .epoch
                .ok_or(Error::ValueError("No Epoch defined".into()))?
                .into(),
            eccentricity: self
                .eccentricity
                .ok_or(Error::ValueError("No Eccentricity defined".into()))?,
            inclination: self
                .inclination
                .ok_or(Error::ValueError("No Inclination defined".into()))?
                .to_radians(),
            peri_arg: self
                .peri_arg
                .ok_or(Error::ValueError("No peri_arg defined".into()))?
                .to_radians(),
            peri_dist: self
                .peri_dist
                .ok_or(Error::ValueError("No peri_dist defined".into()))?,
            peri_time: self
                .peri_time
                .ok_or(Error::ValueError("No peri_time defined".into()))?
                .into(),
            lon_of_ascending: self
                .lon_of_ascending
                .ok_or(Error::ValueError(
                    "No longitude of ascending node defined".into(),
                ))?
                .to_radians(),
            center_id: 10,
            gm_sqrt: GMS_SQRT,
        })
    }

    /// Ecliptic state vector derived from the orbital elements.
    ///
    /// # Errors
    /// Returns an error if any required element field is absent or conversion fails.
    pub fn state(&self) -> KeteResult<State<Ecliptic>> {
        self.elements()?.try_to_state()
    }

    /// Fetch object properties from the JPL SBDB API.
    ///
    /// Results are cached to disk, and cache is checked before network requests unless
    /// `update_cache` is true.
    ///
    ///
    /// # Arguments
    ///
    /// - `name` - Object name or designation to query.
    /// - `update_name` - If true, use the designation returned by Horizons.
    /// - `update_cache` - If true, bypass cached data and re-query.
    /// - `exact_name` - If true, query by exact designation (`des=`) instead
    ///   of substring search (`sstr=`). Needed for fragment designations
    ///   like `"73P-B"`.
    ///
    /// # Errors
    /// Returns an error if the HTTP request fails, the response cannot be parsed,
    /// or no orbit information is present for the object.
    #[cfg(feature = "fetch")]
    pub fn fetch(
        name: &str,
        update_name: bool,
        update_cache: bool,
        exact_name: bool,
    ) -> KeteResult<Self> {
        let (resp, raw_json_str) = fetch_sbdb_json(name, exact_name, update_cache)?;

        // Resolve name from response if requested.
        let desig = if update_name {
            resp.object
                .as_ref()
                .and_then(|o| o.des.clone())
                .or_else(|| {
                    resp.object.as_ref().and_then(|o| {
                        o.fullname
                            .as_ref()
                            .map(|f| f.rsplit('(').next().unwrap_or(f).replace(')', ""))
                    })
                })
                .unwrap_or_else(|| name.to_string())
        } else {
            name.to_string()
        };

        let orbit = resp.orbit.as_ref().ok_or_else(|| {
            Error::ValueError(format!(
                "Horizons did not return orbit information for '{name}'"
            ))
        })?;

        // Orbital elements
        let find_element = |label: &str| -> Option<f64> {
            orbit
                .elements
                .as_ref()?
                .iter()
                .find(|el| el.label == label)?
                .value
                .as_ref()?
                .parse()
                .ok()
        };
        let eccentricity = find_element("e");
        let inclination = find_element("i");
        let lon_of_ascending = find_element("node");
        let peri_arg = find_element("peri");
        let peri_dist = find_element("q");
        let peri_time = find_element("tp");

        let epoch = orbit.epoch.as_ref().and_then(|s| s.parse::<f64>().ok());
        let moid = orbit.moid.as_ref().and_then(|s| s.parse::<f64>().ok());
        let arc_len = orbit.data_arc.as_ref().and_then(|s| s.parse::<f64>().ok());

        // Covariance (params + matrix + epoch), if present.
        let mut covariance_params: Option<Vec<(String, f64)>> = None;
        let mut covariance_matrix: Option<Vec<Vec<f64>>> = None;
        let mut covariance_epoch: Option<f64> = None;

        if let Some(cov) = &orbit.covariance
            && let Some(data) = &cov.data
            && let Some(labels) = &cov.labels
        {
            covariance_epoch = cov.epoch.as_ref().and_then(|s| s.parse().ok());

            let json_f64 = |v: &serde_json::Value| -> f64 {
                let f = match v {
                    serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.0),
                    serde_json::Value::String(s) => s.parse().unwrap_or(0.0),
                    serde_json::Value::Null
                    | serde_json::Value::Bool(_)
                    | serde_json::Value::Array(_)
                    | serde_json::Value::Object(_) => 0.0,
                };
                if f.is_nan() { 0.0 } else { f }
            };
            covariance_matrix = Some(
                data.iter()
                    .map(|row| row.iter().map(&json_f64).collect())
                    .collect(),
            );

            // Element values: prefer covariance-specific, fall back to orbital.
            let mut values: std::collections::HashMap<String, f64> =
                if let Some(cov_els) = &cov.elements {
                    cov_els
                        .iter()
                        .filter_map(|lv| {
                            lv.value
                                .as_ref()?
                                .parse()
                                .ok()
                                .map(|v| (lv.label.clone(), v))
                        })
                        .collect()
                } else {
                    [
                        ("e", eccentricity),
                        ("i", inclination),
                        ("node", lon_of_ascending),
                        ("peri", peri_arg),
                        ("q", peri_dist),
                        ("tp", peri_time),
                    ]
                    .into_iter()
                    .filter_map(|(k, v)| v.map(|val| (k.to_string(), val)))
                    .collect()
                };

            if let Some(model_pars) = &orbit.model_pars {
                for mp in model_pars {
                    if let Some(val_str) = &mp.value
                        && let Ok(v) = val_str.parse::<f64>()
                    {
                        let _ = values.insert(mp.name.clone(), v);
                    }
                }
            }

            covariance_params = Some(
                labels
                    .iter()
                    .map(|lab| {
                        let kete_name = match lab.as_str() {
                            "e" => "eccentricity",
                            "q" => "peri_dist",
                            "tp" => "peri_time",
                            "node" => "lon_of_ascending",
                            "peri" => "peri_arg",
                            "i" => "inclination",
                            other => other,
                        };
                        let val = values.get(lab.as_str()).copied().unwrap_or(f64::NAN);
                        (kete_name.to_string(), val)
                    })
                    .collect(),
            );
        }

        // Physical parameters
        let find_phys = |pname: &str| -> Option<f64> {
            resp.phys_par
                .as_ref()?
                .iter()
                .find(|p| p.name == pname)?
                .value
                .as_ref()?
                .parse()
                .ok()
        };

        let mut props = Self::new(
            desig.clone(),
            resp.object
                .as_ref()
                .and_then(|o| o.orbit_class.as_ref())
                .and_then(|oc| oc.name.clone()),
            epoch,
            eccentricity,
            inclination,
            lon_of_ascending,
            peri_arg,
            peri_dist,
            peri_time,
            find_phys("H"),
            find_phys("albedo"),
            find_phys("diameter"),
            moid,
            find_phys("G"),
            arc_len,
            covariance_params,
            covariance_matrix,
            covariance_epoch,
        )?;

        props.non_grav = orbit
            .model_pars
            .as_ref()
            .and_then(|pars| build_nongrav_from_model_pars(pars));

        props.alternate_desigs = {
            let mut desig_list = vec![desig.clone()];
            if let Some(obj) = &resp.object
                && let Some(alts) = &obj.des_alt
            {
                let prefix = format!("{desig}/");
                for alt in alts {
                    if let serde_json::Value::Object(map) = alt {
                        let lower: std::collections::HashMap<String, &serde_json::Value> =
                            map.iter().map(|(k, v)| (k.to_lowercase(), v)).collect();
                        if let Some(serde_json::Value::String(des)) = lower.get("des") {
                            desig_list.push(des.replace(&prefix, ""));
                        }
                    }
                }
            }
            desig_list
        };

        props.raw_json = Some(raw_json_str);
        Ok(props)
    }
}

/// Build a [`UncertainState`] from raw Horizons covariance data.
///
/// Handles both cometary-element and Cartesian parameterizations,
/// including automatic detection and construction of non-gravitational
/// models when A1/A2/A3 or beta parameters are present.
fn build_uncertain_state(
    desig: &str,
    epoch: f64,
    params: &[(String, f64)],
    cov_matrix: &[Vec<f64>],
) -> KeteResult<UncertainState> {
    let n_params = params.len();
    if cov_matrix.len() != n_params {
        return Err(Error::ValueError(format!(
            "Covariance matrix has {} rows but {} parameters",
            cov_matrix.len(),
            n_params
        )));
    }
    for (i, row) in cov_matrix.iter().enumerate() {
        if row.len() != n_params {
            return Err(Error::ValueError(format!(
                "Covariance matrix row {i} has length {}, expected {n_params}",
                row.len()
            )));
        }
    }
    let lower_names: Vec<String> = params.iter().map(|(k, _)| k.to_lowercase()).collect();
    let get = |key: &str| -> KeteResult<f64> {
        params
            .iter()
            .find(|(k, _)| k.to_lowercase() == key)
            .map(|(_, v)| *v)
            .ok_or_else(|| Error::ValueError(format!("Horizons covariance missing '{key}'")))
    };

    let elem_keys: &[&str] = &[
        "eccentricity",
        "peri_dist",
        "peri_time",
        "lon_of_ascending",
        "peri_arg",
        "inclination",
    ];
    let cart_keys: &[&str] = &["x", "y", "z", "vx", "vy", "vz"];
    let is_cometary = lower_names.iter().any(|k| elem_keys.contains(&k.as_str()));
    let core_keys = if is_cometary { elem_keys } else { cart_keys };

    let core_indices: Vec<usize> = core_keys
        .iter()
        .filter_map(|&key| lower_names.iter().position(|k| k == key))
        .collect();
    let nongrav_indices: Vec<usize> = (0..lower_names.len())
        .filter(|i| !core_indices.contains(i))
        .collect();

    let non_grav = if nongrav_indices.is_empty() {
        None
    } else {
        let ng_hash: std::collections::HashMap<&str, f64> = nongrav_indices
            .iter()
            .map(|&i| (lower_names[i].as_str(), params[i].1))
            .collect();
        build_nongrav_from_hash(&ng_hash)
    };

    let np = non_grav.as_ref().map_or(0, NonGravModel::n_free_params);
    let n = 6 + np;

    let ng_param_names: Vec<&str> = match &non_grav {
        Some(ng) => ng.param_names().to_vec(),
        None => Vec::new(),
    };
    let reorder: Vec<Option<usize>> = (0..n)
        .map(|i| {
            if i < 6 {
                Some(core_indices[i])
            } else {
                let model_name = ng_param_names.get(i - 6)?;
                nongrav_indices
                    .iter()
                    .find(|&&ni| lower_names[ni] == *model_name)
                    .copied()
            }
        })
        .collect();

    if is_cometary {
        let elements = CometElements {
            desig: Desig::Name(desig.to_string()),
            epoch: epoch.into(),
            eccentricity: get("eccentricity")?,
            inclination: get("inclination")?.to_radians(),
            peri_arg: get("peri_arg")?.to_radians(),
            peri_dist: get("peri_dist")?,
            peri_time: get("peri_time")?.into(),
            lon_of_ascending: get("lon_of_ascending")?.to_radians(),
            center_id: 10,
            gm_sqrt: GMS_SQRT,
        };

        let deg2rad = std::f64::consts::PI / 180.0;
        let scale: Vec<f64> = (0..n)
            .map(|i| if (3..6).contains(&i) { deg2rad } else { 1.0 })
            .collect();

        let mat = DMatrix::from_fn(n, n, |r, c| match (reorder[r], reorder[c]) {
            (Some(sr), Some(sc)) => cov_matrix[sr][sc] * scale[r] * scale[c],
            _ => 0.0,
        });

        UncertainState::from_cometary(&elements, &mat, non_grav)
    } else {
        let x = get("x")?;
        let y = get("y")?;
        let z = get("z")?;
        let vx = get("vx")?;
        let vy = get("vy")?;
        let vz = get("vz")?;

        let desig_val = match desig {
            "" => Desig::Empty,
            _ => Desig::Name(desig.to_string()),
        };
        let state: State<Equatorial> = State::new(
            desig_val,
            Time::new(epoch),
            [x, y, z].into(),
            [vx, vy, vz].into(),
            10,
        );

        let mat = DMatrix::from_fn(n, n, |r, c| match (reorder[r], reorder[c]) {
            (Some(sr), Some(sc)) => cov_matrix[sr][sc],
            _ => 0.0,
        });

        UncertainState::new(state, mat, non_grav)
    }
}

/// Build a [`NonGravModel`] from leftover (non-orbital) sampled parameters.
///
/// Returns `Some(model)` only when the parameter names match a supported
/// non-gravitational model:
///  - **Comet**: at least one of `a1`, `a2`, `a3` is present.
///  - **Dust**: `beta` is present.
///
/// Unrecognized parameter sets (e.g. `rho`, `amrat`) yield `None`;
/// the caller should then fall back to a pure orbital covariance.
fn build_nongrav_from_hash(hash: &std::collections::HashMap<&str, f64>) -> Option<NonGravModel> {
    let get = |key: &str, default: f64| -> f64 { hash.get(key).copied().unwrap_or(default) };

    let has_jpl = hash.contains_key("a1") || hash.contains_key("a2") || hash.contains_key("a3");
    let has_dust = hash.contains_key("beta");

    if has_jpl {
        Some(NonGravModel::new_jpl(
            get("a1", 0.0),
            get("a2", 0.0),
            get("a3", 0.0),
            get("alpha", 0.111_262_042_6),
            get("r_0", 2.808),
            get("m", 2.15),
            get("n", 5.093),
            get("k", 4.6142),
            get("dt", 0.0),
        ))
    } else if has_dust {
        Some(NonGravModel::new_dust(get("beta", 0.0)))
    } else {
        let unknown: Vec<&str> = hash.keys().copied().collect();
        eprintln!(
            "Warning: Horizons covariance contains unrecognized non-gravitational \
             parameters {unknown:?}; ignoring and using orbital covariance only."
        );
        None
    }
}

/// Build a [`NonGravModel`] from the `model_pars` section of a Horizons response.
#[cfg(feature = "fetch")]
fn build_nongrav_from_model_pars(pars: &[NameValue]) -> Option<NonGravModel> {
    let mut a1 = 0.0;
    let mut a2 = 0.0;
    let mut a3 = 0.0;
    let mut alpha = 0.111_262_042_6;
    let mut r_0 = 2.808;
    let mut m = 2.15;
    let mut n = 5.093;
    let mut k = 4.6142;
    let mut dt = 0.0;
    let mut found_any = false;

    for par in pars {
        let name_lower = par.name.to_lowercase();
        let Some(val_str) = &par.value else { continue };
        let Ok(val) = val_str.parse::<f64>() else {
            continue;
        };
        found_any = true;
        match name_lower.as_str() {
            "a1" => a1 = val,
            "a2" => a2 = val,
            "a3" => a3 = val,
            "aln" => alpha = val,
            "nm" => m = val,
            "r0" => r_0 = val,
            "nk" => k = val,
            "nn" => n = val,
            "dt" => dt = val,
            _ => {
                eprintln!("Warning: Unknown non-grav parameter: {}", par.name);
                found_any = false;
            }
        }
    }

    if found_any {
        Some(NonGravModel::new_jpl(a1, a2, a3, alpha, r_0, m, n, k, dt))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// SBDB API serde types (private, fetch feature only)
// ---------------------------------------------------------------------------

#[cfg(feature = "fetch")]
#[derive(Deserialize)]
struct SbdbResponse {
    object: Option<SbdbObject>,
    orbit: Option<SbdbOrbit>,
    phys_par: Option<Vec<NameValue>>,
    list: Option<Vec<SbdbListEntry>>,
}

#[cfg(feature = "fetch")]
#[derive(Deserialize)]
struct SbdbObject {
    des: Option<String>,
    fullname: Option<String>,
    orbit_class: Option<SbdbOrbitClass>,
    des_alt: Option<Vec<serde_json::Value>>,
}

#[cfg(feature = "fetch")]
#[derive(Deserialize)]
struct SbdbOrbitClass {
    name: Option<String>,
}

#[cfg(feature = "fetch")]
#[derive(Deserialize)]
struct SbdbOrbit {
    epoch: Option<String>,
    elements: Option<Vec<LabelValue>>,
    covariance: Option<SbdbCovariance>,
    model_pars: Option<Vec<NameValue>>,
    moid: Option<String>,
    data_arc: Option<String>,
}

#[cfg(feature = "fetch")]
#[derive(Deserialize)]
struct SbdbCovariance {
    epoch: Option<String>,
    data: Option<Vec<Vec<serde_json::Value>>>,
    labels: Option<Vec<String>>,
    elements: Option<Vec<LabelValue>>,
}

/// Used for orbit element entries where the JSON key is "label".
#[cfg(feature = "fetch")]
#[derive(Deserialize)]
struct LabelValue {
    #[serde(alias = "label")]
    label: String,
    value: Option<String>,
}

#[cfg(feature = "fetch")]
#[derive(Deserialize)]
struct NameValue {
    name: String,
    value: Option<String>,
}

#[cfg(feature = "fetch")]
#[derive(Deserialize)]
struct SbdbListEntry {
    pdes: Option<String>,
}

/// Perform a single HTTP request to the JPL SBDB API and return the deserialized response.
///
/// Handles HTTP 300 (multiple matches) by filtering out fragment designations
/// and retrying with an exact match when exactly one candidate remains.
/// Does not interact with the local cache.
#[cfg(feature = "fetch")]
fn query_sbdb_json(name: &str, exact: bool) -> KeteResult<(SbdbResponse, String)> {
    let resolved = Desig::parse_mpc_packed_designation(name.trim())
        .map_or_else(|_| name.to_string(), |d| d.to_string());

    let query_key = if exact { "des" } else { "sstr" };
    let url = format!(
        "https://ssd-api.jpl.nasa.gov/sbdb.api?{query_key}={resolved}\
         &phys-par=true&full-prec=true&cov=mat\
         &alt-des=true&alt-spk=true&alt-orbits=true&discovery=true"
    );

    let response = ureq::get(&url)
        .call()
        .map_err(|e| Error::IOError(format!("Horizons SBDB request failed: {e}")))?;

    let status = response.status();
    let body = response
        .into_body()
        .read_to_string()
        .map_err(|e| Error::IOError(format!("Failed to read response body: {e}")))?;

    if status == 300 {
        // Multiple matches -- try to narrow down.
        if let Ok(multi) = serde_json::from_str::<SbdbResponse>(&body)
            && let Some(list) = &multi.list
        {
            let candidates: Vec<&str> = list
                .iter()
                .filter_map(|e| e.pdes.as_deref())
                .filter(|p| !p.contains('-'))
                .collect();
            if candidates.len() == 1 {
                return query_sbdb_json(candidates[0], true);
            }
        }
        return Err(Error::IOError(format!(
            "Horizons returned multiple matches (HTTP 300) for '{name}'"
        )));
    }

    if status != 200 {
        return Err(Error::IOError(format!(
            "Horizons returned HTTP {status} for '{name}': {body}"
        )));
    }

    let resp: SbdbResponse = serde_json::from_str(&body)
        .map_err(|e| Error::IOError(format!("Failed to parse Horizons JSON: {e}")))?;
    Ok((resp, body))
}

/// Query the JPL SBDB API, using the local disk cache when available.
///
/// On a cache miss the query is performed via [`query_sbdb_json`] and the
/// result is written to the cache before returning.
#[cfg(feature = "fetch")]
fn fetch_sbdb_json(
    name: &str,
    exact: bool,
    update_cache: bool,
) -> KeteResult<(SbdbResponse, String)> {
    let mut dir = cache_dir()?;
    dir.push("horizons_props");
    if !dir.exists() {
        fs::create_dir_all(&dir)?;
    }
    let stem = Desig::parse_mpc_designation(name.trim())
        .and_then(|d| d.try_pack())
        .unwrap_or_else(|_| name.replace('/', "_"));
    let filename = dir.join(format!("{stem}.json"));

    if !update_cache
        && let Ok(json_str) = fs::read_to_string(&filename)
        && let Ok(resp) = serde_json::from_str::<SbdbResponse>(&json_str)
    {
        return Ok((resp, json_str));
    }

    let (resp, body) = query_sbdb_json(name, exact)?;

    let _ = fs::write(&filename, &body);

    Ok((resp, body))
}
