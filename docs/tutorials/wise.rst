WISE Precovery Calculations
===========================

This example loads the last release year of NEOWISE fields of view, and predicts
which known numbered NEOs are present in all of the frames of NEOWISE during that time.

This can be thought of as an extended version of the KONA tutorial. 


WISE Fields of View
-------------------

A field of view is a patch of sky as seen from an observer. kete supports downloading
these FOVs directly from IPACs IRSA data repository.


.. code-block:: python

    import kete

    # During 2023 there were 2.369 million individual frames taken of both W1 and W2
    # bands, totalling just shy of 5 million individual frames.
    # This may take some time to download from IRSA.
    fovs = kete.wise.fetch_WISE_fovs('Reactivation_2023')

Minor Planet Center Data
------------------------

Now we have to download the orbit data of NEOs from the Minor Planet Center (MPC).


.. code-block:: python

    # This is the orbit table from the MPC, this is over 100Mb, which also may take a
    # minute to download. Note that kete caches these files after the first download.
    # Future queries will use the cached copy unless `force_download=True` is set.
    orb_table = kete.mpc.fetch_known_orbit_data()

    # Now we can down select these to just NEOs:
    is_neo = kete.population.neo(orb_table['peri_dist'], orb_table['ecc'])
    subset = orb_table[is_neo]

    # select only the numbered asteroids
    subset = subset[[str(n).isdigit() for n in subset.desig]]

    # Convert that table of values into cartesian state vectors:
    states = kete.conversion.table_to_states(subset)


Propagation
-----------

Now that we have the states loaded, we have to do a little preparation.

The MPC orbit table will have a mixture of epochs for the objects, we need to
first bring all of the epochs to a common time. A convenient time is the first
frame in the mission phase we have selected.

.. code-block:: python

    # Time of the first exposure.
    jd = fovs[0].jd

    # now we propagate the NEOs to that time, including the effects of the 5 largest
    # main belt asteroids to include more precision. This may take a few minutes.
    states = kete.propagate_n_body(states, jd, include_asteroids=True)

Visibility Test
---------------

Now that we have the FOVs and the States, we can compute what should have been
observable to NEOWISE during this mission phase.

.. note::

    This takes a while to run, as it is computing the State of all NEOs for
    every ~2.4 million frames during 2023 of NEOWISE. On a high end desktop this
    takes around 3 minutes, on a relatively standard macbook pro expect it to
    take around 10 minutes. This time scales relatively linearly with number
    of objects and the number of fields of view.
    
.. code-block:: python

    # This accepts States and a list of field of views, and computes when objects
    # would have been seen. The final output will only include times where objects
    # were seen, and it drops empty FOVs.

    # Compute observable objects.
    visible = kete.fov_state_check(states, fovs)


.. note::

    The outputs of this may be saved using the following:
    
    ``kete.SimultaneousStates.save_list(visible, "visible_wise_2023.bin")``

    The states may later be loaded using:

    ``visible = kete.SimultaneousStates.load_list("visible_wise_2023.bin")``


Computing Positions
-------------------

We can now compute the on-sky positions of these objects as seen from NEOWISE.

Here is a codeblock which prints the first `n_show=100` objects.

.. code-block:: python
        
    n_show = 100
    print("Found: ", len(visible))
    print(f"Displaying the first {n_show}")
    print(f"{'Name':<16}{'mjd':<16}{'RA':<16}{'DEC':<16}{'scan-frame':<16}")
    print("-"*(16 * 5))
    for vis in visible[:n_show]:
        for state in vis:
            vec = (state.pos - vis.fov.observer.pos).as_equatorial
            mjd = kete.Time(vis.fov.jd).mjd
            print((f"{state.desig:<15s},{mjd:<15.6f},{vec.ra_hms:<15s},"
                   f"{vec.dec_dms:<15s},{vis.fov.scan_id}-{str(vis.fov.frame_num)}"))


::
    
    Found:  77100
    Displaying the first 100
    Name            mjd             RA              DEC             scan-frame
    --------------------------------------------------------------------------------
    489453         ,59945.005428   ,01 08 21.224   ,+30 49 29.19   ,46370r-175
    279816         ,59945.015360   ,20 22 46.804   ,+69 13 34.94   ,46370r-261
    279816         ,59945.015487   ,20 22 46.852   ,+69 13 34.91   ,46370r-262
    254417         ,59945.016888   ,18 54 08.965   ,+68 51 47.51   ,46370r-274
    162926         ,59945.026820   ,13 45 32.679   ,+31 30 08.89   ,46372r-54
    4544           ,59945.029240   ,13 17 39.854   ,+19 49 48.24   ,46372r-75
    513572         ,59945.030386   ,13 08 11.075   ,+14 19 56.97   ,46372r-85
    455594         ,59945.030768   ,13 05 59.732   ,+12 09 09.00   ,46372r-88
    550271         ,59945.032169   ,12 51 36.151   ,+05 15 40.13   ,46372r-100
    620064         ,59945.032805   ,12 46 41.606   ,+01 59 36.38   ,46372r-106
    749593         ,59945.034206   ,12 36 32.326   ,-05 05 32.52   ,46372r-118
    277810         ,59945.035352   ,12 25 45.458   ,-10 45 33.34   ,46372r-128
    455687         ,59945.064003   ,02 06 58.058   ,-02 02 57.70   ,46373r-93
    506491         ,59945.065913   ,01 54 09.024   ,+07 51 45.32   ,46373r-110
    163373         ,59945.066932   ,01 46 19.033   ,+13 17 37.97   ,46373r-119
    427621         ,59945.066932   ,01 46 29.615   ,+13 21 44.67   ,46373r-119
    416151         ,59945.067951   ,01 37 03.921   ,+17 55 31.32   ,46373r-127
    434633         ,59945.069352   ,01 25 37.632   ,+25 20 03.03   ,46373r-139
    138852         ,59945.069606   ,01 23 18.970   ,+26 28 03.67   ,46373r-142
    279816         ,59945.080557   ,20 23 00.053   ,+69 13 17.35   ,46373r-236
    162926         ,59945.092017   ,13 45 36.258   ,+31 30 46.73   ,46374r-78
    455594         ,59945.095838   ,13 06 07.094   ,+12 07 38.15   ,46374r-111
    455594         ,59945.095965   ,13 06 07.109   ,+12 07 38.07   ,46374r-113
    495833         ,59945.097875   ,12 50 40.533   ,+02 05 59.98   ,46374r-129
    1627           ,59945.098257   ,12 47 13.867   ,+00 13 16.64   ,46374r-132
    749593         ,59945.099276   ,12 36 40.709   ,-05 01 13.07   ,46374r-141
    277810         ,59945.100422   ,12 25 50.726   ,-10 46 03.61   ,46374r-151
    378842         ,59945.102077   ,12 12 38.730   ,-19 14 51.46   ,46374r-165
    162082         ,59945.103987   ,11 53 32.525   ,-28 42 28.03   ,46374r-182
    8566           ,59945.121560   ,03 21 18.422   ,-39 41 22.61   ,46375r-52
    481918         ,59945.130092   ,02 04 42.505   ,+03 23 43.24   ,46375r-125
    194268         ,59945.130219   ,02 01 54.746   ,+04 00 00.35   ,46375r-126
    162926         ,59945.157087   ,13 45 39.825   ,+31 31 24.50   ,46376r-24
    4544           ,59945.159507   ,13 17 53.093   ,+19 49 21.42   ,46376r-45
    513572         ,59945.160653   ,13 08 33.979   ,+14 20 08.22   ,46376r-55
    455594         ,59945.161035   ,13 06 14.475   ,+12 06 07.07   ,46376r-59
    550271         ,59945.162435   ,12 51 45.573   ,+05 13 37.57   ,46376r-71
    620064         ,59945.163072   ,12 46 44.931   ,+01 59 31.40   ,46376r-76
    749593         ,59945.164473   ,12 36 49.108   ,-04 56 52.62   ,46376r-88
    277810         ,59945.165619   ,12 25 56.000   ,-10 46 33.75   ,46376r-98
    481918         ,59945.195162   ,02 04 41.870   ,+03 23 44.15   ,46377r-35
    481918         ,59945.195289   ,02 04 41.869   ,+03 23 44.12   ,46377r-36
    162926         ,59945.222284   ,13 45 43.395   ,+31 32 02.42   ,46378r-77
    455594         ,59945.226104   ,13 06 21.845   ,+12 04 35.94   ,46378r-110
    455594         ,59945.226232   ,13 06 21.861   ,+12 04 35.86   ,46378r-112
    495833         ,59945.228142   ,12 50 44.119   ,+02 05 47.94   ,46378r-128
    1627           ,59945.228524   ,12 47 23.485   ,+00 12 49.94   ,46378r-131
    749593         ,59945.229543   ,12 36 57.481   ,-04 52 32.82   ,46378r-140
    277810         ,59945.230689   ,12 26 01.256   ,-10 47 03.92   ,46378r-150
    378842         ,59945.232344   ,12 12 54.185   ,-19 17 15.53   ,46378r-164
    162082         ,59945.234254   ,11 53 44.261   ,-28 44 43.31   ,46378r-181
    8566           ,59945.251827   ,03 21 17.899   ,-39 38 24.34   ,46379r-51
    482650         ,59945.254756   ,02 52 08.728   ,-24 45 03.47   ,46379r-76
    530531         ,59945.258576   ,02 22 16.630   ,-05 39 35.48   ,46379r-109
    497230         ,59945.260613   ,02 06 51.116   ,+04 54 12.57   ,46379r-127
    441641         ,59945.262141   ,01 54 24.910   ,+12 50 35.60   ,46379r-140
    475950         ,59945.262141   ,01 55 21.450   ,+12 40 29.63   ,46379r-140
    424392         ,59945.262778   ,01 48 55.200   ,+16 24 58.33   ,46379r-145
    424392         ,59945.262905   ,01 48 55.207   ,+16 24 58.18   ,46379r-147
    254417         ,59945.277422   ,18 55 02.489   ,+68 58 07.10   ,46379r-272
    162926         ,59945.287354   ,13 45 46.954   ,+31 32 40.27   ,46380r-52
    513572         ,59945.290920   ,13 08 56.820   ,+14 20 19.57   ,46380r-83
    455594         ,59945.291302   ,13 06 29.235   ,+12 03 04.58   ,46380r-86
    550271         ,59945.292702   ,12 51 54.977   ,+05 11 34.97   ,46380r-98
    620064         ,59945.293339   ,12 46 48.242   ,+01 59 26.54   ,46380r-104
    749593         ,59945.294612   ,12 37 05.848   ,-04 48 12.86   ,46380r-115
    277810         ,59945.295886   ,12 26 06.519   ,-10 47 33.96   ,46380r-126
    8566           ,59945.317024   ,03 21 17.654   ,-39 36 55.12   ,46381r-27
    481918         ,59945.325428   ,02 04 40.609   ,+03 23 46.00   ,46381r-100
    481918         ,59945.325556   ,02 04 40.608   ,+03 23 45.97   ,46381r-101
    162926         ,59945.352424   ,13 45 50.509   ,+31 33 18.17   ,46382r-76
    162926         ,59945.352551   ,13 45 50.516   ,+31 33 18.28   ,46382r-78
    455594         ,59945.356371   ,13 06 36.614   ,+12 01 33.15   ,46382r-111
    455594         ,59945.356499   ,13 06 36.629   ,+12 01 33.07   ,46382r-112
    495833         ,59945.358409   ,12 50 47.689   ,+02 05 36.01   ,46382r-128
    1627           ,59945.358791   ,12 47 33.093   ,+00 12 23.37   ,46382r-131
    749593         ,59945.359809   ,12 37 14.230   ,-04 43 51.88   ,46382r-140
    378842         ,59945.362611   ,12 13 09.658   ,-19 19 39.74   ,46382r-164
    162082         ,59945.364521   ,11 53 55.976   ,-28 46 58.43   ,46382r-181
    8566           ,59945.382094   ,03 21 17.419   ,-39 35 25.95   ,46383r-51
    482650         ,59945.385023   ,02 52 04.814   ,-24 42 54.37   ,46383r-76
    486607         ,59945.388333   ,02 26 09.379   ,-08 05 50.98   ,46383r-105
    530531         ,59945.388843   ,02 22 16.527   ,-05 33 50.64   ,46383r-109
    497230         ,59945.390880   ,02 06 50.586   ,+04 54 33.87   ,46383r-127
    441641         ,59945.392408   ,01 54 23.840   ,+12 50 51.28   ,46383r-140
    475950         ,59945.392408   ,01 55 21.477   ,+12 40 40.07   ,46383r-140
    424392         ,59945.393045   ,01 49 00.039   ,+16 25 20.87   ,46383r-145
    424392         ,59945.393172   ,01 49 00.045   ,+16 25 20.72   ,46383r-147
    138846         ,59945.394828   ,01 34 51.198   ,+25 38 04.67   ,46383r-161
    138846         ,59945.394955   ,01 34 51.200   ,+25 38 04.55   ,46383r-162
    199003         ,59945.399030   ,00 46 38.382   ,+46 06 48.22   ,46383r-197
    254417         ,59945.407562   ,18 55 29.363   ,+69 01 16.94   ,46383r-271
    254417         ,59945.407689   ,18 55 29.405   ,+69 01 17.17   ,46383r-272
    162926         ,59945.417621   ,13 45 54.066   ,+31 33 56.22   ,46384r-52
    513572         ,59945.421059   ,13 09 19.575   ,+14 20 30.74   ,46384r-82
    513572         ,59945.421186   ,13 09 19.598   ,+14 20 31.00   ,46384r-83
    455594         ,59945.421568   ,13 06 44.011   ,+12 00 01.50   ,46384r-86
    749593         ,59945.424879   ,12 37 22.586   ,-04 39 31.57   ,46384r-115
    277810         ,59945.426153   ,12 26 17.013   ,-10 48 33.96   ,46384r-126
    8566           ,59945.447291   ,03 21 17.195   ,-39 33 56.67   ,46385r-27
    162926         ,59945.482691   ,13 45 57.612   ,+31 34 34.21   ,46386r-77
    162926         ,59945.482818   ,13 45 57.619   ,+31 34 34.32   ,46386r-78
    455594         ,59945.486638   ,13 06 51.399   ,+11 58 29.79   ,46386r-111