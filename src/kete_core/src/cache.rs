//! Managment of the cached files
//!
//! kete saves files in a cache directory, which by default exists in the
//! home directory of the user. This can be changed by setting the KETE_CACHE_DIR
//! environment variable. The cache directory is used to store files that are
//! downloaded from the internet, for example the SPICE kernels.

/// BSD 3-Clause License
///
/// Copyright (c) 2025, Dar Dahlen
///
use directories::UserDirs;
use std::{
    env::{self},
    path::PathBuf,
};

use crate::errors::{Error, KeteResult};

/// Get the cache directory for kete.
///
/// This first checks if the KETE_CACHE_DIR environment variable is set.
/// If it is set, it checks if the directory exists. If it does not exist,
/// it panics. If it does exist, it returns the path.
///
/// If the KETE_CACHE_DIR environment variable is not set, it
/// creates a directory in the home directory of the user.
pub fn get_cache_dir() -> KeteResult<PathBuf> {
    env::var("KETE_CACHE_DIR")
        .map(|env_path| {
            let path = PathBuf::from(env_path);
            if !path.exists() {
                return Err(Error::IOError(format!(
                    "KETE_CACHE_DIR does not exist: {:?}",
                    path
                )));
            }
            Ok(path)
        })
        .unwrap_or_else(|_| {
            let user_dirs =
                UserDirs::new().ok_or(Error::IOError("Failed to find home directory.".into()))?;
            let path = user_dirs.home_dir();
            let path = path.join(".kete");
            if !path.exists() {
                std::fs::create_dir_all(&path)?;
            }
            Ok(path)
        })
}
