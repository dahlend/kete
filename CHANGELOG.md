# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]


### Added

- Added support for PTF fields of view for all public data. PTF operated from 2009 to
  2018, however data is only publicly available through early 2015.
- Generalized TAP queries to support queries to the Canadian Astronomy Data Centre
  (CADC).
- Added initial support for SPICE SCLK text kernels, allowing the conversion of
  spacecraft formatted time strings to `kete.Time` objects. Conversion matches cSPICE
  to near numerical precision limits.

### Changed

- Saving files to any binary format will no longer be guaranteed to be compatible in
  future versions. If a file is saved by a particular version of kete, it may only be
  compatible for that version.
- Restructured frames of reference throughout the core rust code. This is a significant
  breaking change, which introduces new objects into the kete_core code. The biggest
  changes are that frames of reference are now individual classes which implement a new
  traits `InertialFrame` and `NonInertialFrame`, and removes the previous Enum for
  frames. As a result of this, there is a new class `Vector<T>` which has been added as
  well, which must be of type `InertialFrame`. Numerous places throughout the code now
  use these `Vector` classes, which guarantees the reference frames are preserved
  without manually performing a check for matching frames. `State` now use reference
  frames as well, and `SimultaneousStates` and `FOV` objects now all use
  `State<Equatorial>` as the base representation of the states of the objects.
- Significant rewrite of reading SPICE files, there are now wrapper view types to the
  for SPK and PCK.
- Field of Views as downloaded from IRSA are now saved as parquet files. This enables
  them to be backward compatible if there are future changes to FOV, States, or Vector
  definitions. This is a breaking change for previously downloaded FOV files, and these
  may be deleted.
- Moved plotting tools out of `kete.irsa` into `kete.plot`.
- Generalized `kete.irsa` tools to `kete.tap`, and added cached query support. Queries
  to any TAP service are now cached by default, and re-running the query will restore
  the existing results by default.
- Moved `kete.mpc.table_to_states` into `kete.conversion.table_to_states`.
- Renamed `kete.wise.fetch_wise_fovs` to `kete.wise.fetch_fovs`.
- Renamed `kete.ztf.fetch_ztf_fovs` to `kete.ztf.fetch_fovs`.
- Moved RA/DEC string parsing to rust backend.
- Many functions in python now support passing one or a list of objects, for example,
  `kete.propagate_n_body` can now be passed single or multiple states, and will return
  the correct type back.

### Fixed

- Fixed support for `MPCObservation` parsing of MPC files, this regression was 
  due to upstream changes at the MPC in their formats.

## [v1.1.0]

Announcement: Author of Kete (Dar Dahlen) has left IPAC Caltech to begin a PhD at
TU Braunschweig in Germany. Kete has been forked from the public copy maintained by
Caltech, and future development of this fork will occur as a personal project.

### Changed

- SPICE kernels were removed for the repository, and now automatically download on
  first use.

## [v1.0.8]

### Added

- Improved SPICE kernel lookup performance when using large numbers of kernels.
  Testing with 1500 kernels has increased lookup speed by ~26x.
- Added `HorizonsProperties.desigs` which lists all of the designations in horizons.
  This will use a cached json file, for existing installations you will have to update
  the cached horizons properties: `HorizonsProperties.fetch(desig, update_cache=True)`
- Updated the NAIF ID list for all comets, allowing kete to resolve recent comet
  spice IDs, this includes redesignations of known comets.

### Fixed

- Previous update to WISE spice kernel accidentally deleted the original mission spice
  kernel. This section has been re-added to the WISE kernel, which should now contain
  all phases of operation with the best orbit knowledge available.


## [v1.0.7]

### Added

- Added functions for estimating the position of Earth coordinates outside of the PCK
  limits. This allows for estimating Palomar's position in 1950.

### Changed

- Switched to using the MPC Observatory code file for the source of observatory
  locations.

### Fixed

- WGS84 conversion between ECEF and Lat/Lon/Alt was missing a square-root causing a
  small offset (typically 10s to 100s of meters). This has been fixed.


## [v1.0.6]

### Added

- Added support for SPICE kernels of type 3, this allows reading satellites of Mars.
- Added support for choosing to not load the DE440 file when reloading SPICE kernels.
- Added optional suppression of impact warnings during propagation.
- Added new definitions for orbital families of objects.

### Fixed

- Fixed the recently added Earth precession calculation which was providing a transposed
  matrix.

### Changed

- Updated PCK Files for Earth's orientation to the latest produced by NAIF.
- Updated leap second file (no new leap seconds).
- Updated the WISE spice kernels, which improves accuracy significantly for the final 2
  years of WISE/NEOWISE.


## [v1.0.5]

### Added

- Added support for SPICE kernels of type 9, this allows reading SOHO spice files.
- Added support for SPICE kernels of type 18, this allows reading Rosetta spice files.
- Added calculation of Earth's precession, allowing the computation of time dependent
  equatorial vectors.

### Changed

- Comet Magnitude estimates now accepts two phase correction values instead of 1.
- Restructured SPICE kernel memory management to make entire class of bugs impossible.

### Fixed

- Fixed a text case-sensitivity but on Horizons parameter parsing.
- Thermal model example had function arguments out of order.


## [v1.0.4]

### Added

- Added support for saving and loading `SimultaneousStates` as Parquet files.

### Changed

- Horizons orbit table query now includes, M1/2, K1/2, PC, and rotation period.

### Fixed

- NEOS Chip size calculation was slightly incorrect with regard to the placement of the
  gaps between the chips.
- NEOS FOV rotation was being calculated in the ecliptic frame, whereas images will be
  in the equatorial frame. Rotation is now defaulting to the equatorial frame.


## [v1.0.3]

### Added

- Added support for loading orbit information from JPL Horizons
- Added more complete testing for light delay computations in the various FOV checks.
- Added quality of life function to `SimultaneousState` for computing the RA/Dec along
  with the projected on sky rates of motion.

### Changed

- Changed the return signature for `fov_static_check`, this now returns indices of
  the inputs, instead of the original inputs themselves.

### Fixed

- Sampling of JPL Horizons orbit fits was slightly off, and appears to be fixed by
  assuming that the epoch time of the covariance fits is in UTC. This assumption is
  now included in the covariance matrix sampling. This is different than their other
  data products.
- Field of View checks for SPK checks was not returning the correct light delayed time,
  it was returning the position/velocity at the observed position, but the time was
  the instantaneous time.

### Removed

- Removed several polar coordinate transformations in the rust backend which were all
  equivalent variations of latitude/longitude conversions. Nothing in the Python was
  impacted.


## [v1.0.2]

### Added

- Added Python 3.13 to the built packages.
- Added `sample` to the `HorizonsProperties` object, allowing sampling of the orbit's
  uncertainty.
- Added support for time delayed non-gravitational forces, as is found a number of
  comets in JPL Horizons.

## [v1.0.1]

### Added

- Added worked example of calculating if an object is an Earth Trojan.
- Add Earth obliquity calculation.
- Exposed WGS84 coordinate conversions in the python frontend.

### Fixed

- Epoch and Perihelion time conversion when loading JPL Horizons covariance matrices was
  not being done for UTC to TDB, leading to small residuals in covariance sampling.


## [v1.0.0]

### Added

- Added final SPICE kernels for the WISE mission, it now contains all positions from
  all phases of operation. There is a gap for the years it was not operating.
- Updated WISE mission phases to reflect the final data products about to be released.
- Updated ZTF for the current release 22.
- Added `kete.RectangleFOV.from_wcs`, allowing the construction of a FOV from a given
  Astropy WCS object.
- Added `kete.conversion.bin_data`, which allows for binning matrix data such as images.
- Added tutorial showing precovery of an asteroid from a 1950's glass plate observation
  done at the Palomar Observatory.
- Added sun-shield rotation calculation for NEO Surveyor.
- All FOV's now have the `.jd()` method which returns the JD of the observer state.

### Changed

- Building FOVs from corners, now will flip the corner ordering if the final FOV is
  pointing in the opposite direction of the input corners. This means Rectangle FOVs
  cannot be defined from corners with on sky angles greater than 180 degrees.

### Fixed

- Orbital elements calculations for True Anomaly and Eccentric Anomaly were numerically
  unstable with nearly parabolic orbits.
- Time scaling bugs when loading times from both Horizons and the MPC were fixed, these
  were causing offsets of about a minute in the loaded states. However it was shifting
  both the perihelion time and the epoch time by the same amount, leading to only minor
  effective errors.
- Fixed a time offset in the FOV's downloaded from IRSA WISE/NEOWISE. They were offset
  by 4.4 seconds.
- Fixed rotation approximation in WISE/NEOWISE field of views which was causing a small
  percentage of objects to not be found during FOV checks when they were close to the
  edge of the field.
- Constant for the sqrt of GMS was incorrect by a small amount, this value was fixed.
- IRSA username/password options are now being passed through correctly in WISE.

### Removed

- Removed `plot_frame` from ZTF, as a better version of this is available in kete.irsa.
- Removed `cache_WISE_frame` and `fetch_WISE_frame` deprecated functions in WISE.


## [0.3.0] - 2024 - 8 - 28

### Added

- Added an Omni-Directional Field of View.
- Added a "getting started" example.

### Changed

- Renamed the project to Kete.
- Optimized the `moid` computation, improving performance by over 30x.

### Fixed

- Field of View checks for states was optimized for multi-core processing on millions
  of objects, leading to huge speed gains for large queries.
- Fixed a bug where saving lists of SimultaneousStates had a bug where field of view
  information was not being saved correctly.
- Fixed incompatibility with older versions of the rust compiler, working back to
  at least v1.75.0


## [0.2.5] - 2024 - 8 - 12

### Added

- Added support for long term integrations, on the scale of less than a mega-year.
- Added clear deprecation tooling, which helps indicate how to update kete when a
  function's signature changes or is removed.
- Added default plotting tools for fits files, which make decent scaling guesses for
  ZTF and WISE frames. This includes annotation and zooming functions.

### Changed

- Cached files are now zipped if possible after download.
- Cached WISE frames are sorted by folders by the last 2 digits of the scan id, this
  helps when downloading thousands of frames.
- Combined `fetch_WISE_frame` and `cache_WISE_frame` functions into `fetch_frame`.
- Renamed `cached_gzip_json_download` to `download_json`.
- Renamed `cached_file_download` to `download_file`.
- Optimizations to SPICE kernel queries leading to a 20% speedup in orbit propagation,
  along with >43% speedup in state queries from the spice kernels. Speedup from orbit
  propagation comes directly from the spice kernel optimization.
- Moved python wrappers over propagation into the rust backend entirely.
- Simplified polymorphic support in the rust wrappers, removing some intermediate
  objects which were over-complicating the interface.
- Refactored the computation of gravitational forces, improving maintainability.

### Fixed

- Fixed documentation on Time which was not being displayed correctly in python.

### Removed

- `cached_zip_download` was deprecated, this was automatically unzipping folders.


## [0.2.4] - 2024 - 7 - 15

### Added

- Add `J2` non-spherical terms for the gravitational models of Earth and Jupiter.
- Add non-gravitational force model for dust particles which includes the
  Poynting-Roberterson effect. 
- Added `Time.year_float` which converts the `Time` object to the Year as a float.
- Added `SimultaneousState.obs_vecs` which creates vectors from the observer to the
  objects contained within the state.
- Added `NEATM` tutorial which gives an overview of NEATM along with a small example.

### Changed

- Optimized SPICE kernel loading for Type 2 records, which is what DE440 is saved as.
  This means effectively all n-body propagation is now 15-20% faster than before.
- Removed `SpiceKernel` as a class, lowering all its methods to the submodule level,
  see #68 for more discussion.
- Removed `Time` object which was a wrapper over astropy.Time, instead making a
  custom implementation of time which is ~3-400x faster than previous.
- Improved orbital element conversion, leading to a 2-3x speedup in two body orbit
  propagation.
- Significant rewrite of the SPICE kernel file management, this rewrite is required
  so that future work can enable writing SPICE kernel files.
- Moved the downloading of Horizons spice kernels from `SpiceKernel` to `horizons`.
- Renamed all field of view checking functions to similar names: `fov_static_check`,
  `fov_state_check`, and `fov_spice_check`. These functions are exposed at the base
  level of kete.
- Moved much of the python documentation into the rust wrappers, and removed the
  remaining empty python files. Part of the rewrite involved moving and renaming many
  functions.
- Updated `nalgebra` to minimum version `^0.33.0`, which uses a simplified allocator.
- Fixed minimum versions for all rust libraries.

### Fixed

- Orbital Elements now correctly computes `true_anomaly`, `mean_anomaly`,
  `eccentric_anomaly`, `semi_major_axis`, and `mean_motion` for parabolic and
  hyperbolic orbits.
- Two body propagation for parabolic orbits was incorrect.
- Typo in Observability example was leading to an incorrect Magnitude value.
- Astropy WCS warnings are now suppressed.
- `SpiceKernels.moon_illuminated_frac` now agrees with JPL Horizons, this was doing a
  geometric calculation which was correct, but with a different interpretation than
  desired.

### Removed

- Removed support for SPK Files of type 3, these should be a trivial change from type 2
  however I do not have access to a file of type 3 for testing. Because it cannot be
  validated at the moment, the code has been removed.


## [0.2.3] - 2024 - 6 - 12

### Added

- Added computation for Tisserand's Parameter.
- Added support for NEOS Visit FOVs, which are joint FOVs containing 4 rectangles.
- Added python interface to WISE Color Correction functions.
- Added support for querying static sky sources in FOVs.
- SpiceKernels can now load SPK files directly, without needing the files to be in the
  cache.
- SpiceKernels now supports displaying the contents of the headers of SPk/PCK files.
- SpiceKernels can now return the available loaded SPK segments of an object.

### Changed

- Renamed "HorizonsCovariance" to "Covariance" and generalized it to support both
  cometary and cartesian representations.
- Made api more consistent for conversion to State objects by renaming all instances of
  `.as_state` to `.state`.
- Improved performance of integration and propagation by about 10%.
- Restructured the Rust FOVs to be organized by observatory.
- Renamed `SpiceKernels.cache_kernel_reload` to `kernel_reload` and changed it's args.

### Fixed

- Astropy will no longer warning about deprecated WCS header values for NEOWISE images.


## [0.2.2] - 2024 - 5 - 20

### Added

- Added support for downloading ZTF full fields FOV information into the kete cache.
- Added simple lookup for observatory ecliptic state using MPC observatory codes.
- Added 'Tutorials' to the documentation, these are larger worked examples which do not
  build at the same time as the standard docs. They are designed to be examples which
  take significant compute, and may run for many minutes.
- Added Tutorials for 'KONA' and 'WISE Precovery'.

### Changed

- Moved population definitions to the base level and renamed it `population`.
- Changed references to diameters in the broken power law sampler.
- FOV propagation tests now return a flatten list of states.
- Renamed `data` to `cache` to more accurately represent its function.

### Fixed

- Coordinate frame conversion was resulting in incorrect coordinate positions when
  passing Equatorial based frames to some FOV related functions. Specifically it was
  failing to convert the Equatorial frame to Ecliptic frame before performing orbit
  calculations.

### Removed

- Removed construction of populations entirely.
- Removed folder for `population`, the remaining contents were moved up to the base
  level of the package.
- Removed PDS related tools.


## [0.2.1]  - 2024 - 5 - 13

### Added

- Added support for querying SPK kernels of a specific epoch of fit from JPL Horizons.
- Added support for looking up the MPC preferred unpacked designation for asteroids.

### Changed

- Change documentation to clarify that designations are unpacked.
- Updated MPC Obs codes to include NEO Surveyor with code C58.
- Updated NAIF ID list to include new designations for about 20-30 comets.
- Renamed underlying `_rust` binary to `_core`.
- Moved NAIF ID list to a dedicated CSV file which is read during compilation.
- Rename `population.diameter` to `population.power_law`.

### Removed

- Removed main belt construction tools, out of scope and not accurate enough.
- Removed redundant MPC name resolver function.


## [0.2.0]  - 2024-3-16

Initial Release

- Initial version release of kete!
- kete's primary goal is to enable simulations of NEOs, however it also supports any 
  asteroid or comet. Included are n-body orbit propagation tools, thermal and optical
  modeling, tools for computing what minor planets can be seen by an observer.
  Along with many helpful interfaces to web tools such as JPL Horizons or IPAC's IRSA.


[Unreleased]: https://github.com/dahlend/kete/tree/main
[1.1.0]: https://github.com/dahlend/kete/releases/tag/v1.1.0
[1.0.8]: https://github.com/dahlend/kete/releases/tag/v1.0.8
[1.0.7]: https://github.com/dahlend/kete/releases/tag/v1.0.7
[1.0.6]: https://github.com/dahlend/kete/releases/tag/v1.0.6
[1.0.5]: https://github.com/dahlend/kete/releases/tag/v1.0.5
[1.0.4]: https://github.com/dahlend/kete/releases/tag/v1.0.4
[1.0.3]: https://github.com/dahlend/kete/releases/tag/v1.0.3
[1.0.2]: https://github.com/dahlend/kete/releases/tag/v1.0.2
[1.0.1]: https://github.com/dahlend/kete/releases/tag/v1.0.1
[1.0.0]: https://github.com/dahlend/kete/releases/tag/v1.0.0
[0.3.0]: https://github.com/dahlend/kete/releases/tag/v0.3.0
[0.2.5]: https://github.com/dahlend/kete/releases/tag/v0.2.5
[0.2.4]: https://github.com/dahlend/kete/releases/tag/v0.2.4
[0.2.3]: https://github.com/dahlend/kete/releases/tag/v0.2.3
[0.2.2]: https://github.com/dahlend/kete/releases/tag/v0.2.2
[0.2.1]: https://github.com/dahlend/kete/releases/tag/v0.2.1
[0.2.0]: https://github.com/dahlend/kete/releases/tag/v0.2.0
