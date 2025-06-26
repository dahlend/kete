Collection of data files which are used by kete_core.

Files ending in BSP or BPC are are packed binary SPICE kernels.
These are loaded and packed into the final rust binary, and are included automatically.

`mpc_obs.tsv` - Downloaded from https://minorplanetcenter.net/iau/lists/ObsCodes.html
    Contains obs codes and locations. These are loaded and compiled into the final
    binary. Entries with gaps, such as spacecraft, will return with NAN values.

`naif_ids.csv` - CSV containing a table of many of the known NAIF identifiers to full
    names. This is not 100% guaranteed to match, as JPL may decide to renumber objects.
    This is unlikely, but possible.

`Leap_Second.dat` - Leap seconds as provided by IERS, downloaded from:
    https://hpiers.obspm.fr/iers/bul/bulc/Leap_Second.dat
    https://data.iana.org/time-zones/data/leap-seconds.list

`masses.tsv` - List of masses of asteroids from the DE441 header contents, with the very
    distant objects removed.