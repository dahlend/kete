from __future__ import annotations
import io
from functools import lru_cache
import time
import logging
from xml.etree import ElementTree
import requests
import pandas as pd
from .deprecation import rename
from .plot import plot_fits_image, zoom_plot, annotate_plot


IRSA_URL = "https://irsa.ipac.caltech.edu"


logger = logging.getLogger(__name__)

# rename the function to match the new location
plot_fits_image = rename(
    plot_fits_image,
    "1.2.0",
    old_name="plot_fits_image",
    additional_msg="Use `kete.plot.plot_fits_image` instead.",
)
zoom_plot = rename(
    zoom_plot,
    "1.2.0",
    old_name="zoom_plot",
    additional_msg="Use `kete.plot.zoom_plot` instead.",
)
annotate_plot = rename(
    annotate_plot,
    "1.2.0",
    old_name="annotate_plot",
    additional_msg="Use `kete.plot.annotate_plot` instead.",
)


@lru_cache()
def query_column_data(table_name, base_url=IRSA_URL, auth=None):
    """
    Retrieve the column data for a specified IRSA table.

    This will return a dataframe containing the column properties of the target table.

    Parameters
    ----------
    table_name :
        The name of the table to query the columns of.
    base_url :
        The URL of the TAPS service to query, this defaults to IRSA.
    auth :
        An optional (username, password), this may be used to access restricted data.
    """
    return query_irsa_tap(
        f"""SELECT * FROM TAP_SCHEMA.columns WHERE table_name='{table_name}'""",
        base_url=base_url,
        auth=auth,
    )


def query_irsa_tap(
    query, upload_table=None, base_url=IRSA_URL, auth=None, timeout=None, verbose=False
):
    """
    Query IRSA's TAP service, optionally upload a table which will be included in the
    query data. The pandas dataframe table will be labeled as `my_table` and columns in
    the query can be used like so:

    .. testcode::
        :skipif: True

        import kete
        import pandas as pd

        # Column names cannot match the column names in the IRSA table you are querying
        # 0 has been added to the end of these column names to satisfy this constraint.

        data = pd.DataFrame([['foo', 56823.933738, 186.249070833, 22.8977],
                            ['bar', 55232.963786, 49.14175, 21.63811111]],
                            columns=['name', 'mjd0', 'ra0', 'dec0'])

        jd = kete.Time.from_mjd(56823.933738).jd

        # This time corresponds to this phase:
        phase = kete.wise.mission_phase_from_jd(jd)

        # Single source table on IRSA is then: phase.source_table

        # The columns of data available in this source table are
        column_information = kete.irsa.IRSA_column_data(phase.source_table)

        # Note that lots of information is available in column_information

        # Now select which columns we want IRSA to return.
        # Using TAP_UPLOAD.my_table.name we can get back the column of data we sent
        columns_to_fetch = "TAP_UPLOAD.my_table.name, mjd, ra, dec"

        query = (f"select {columns_to_fetch} from {phase.source_table} where " +
                "CONTAINS(POINT('J2000',ra,dec)," +
                "         CIRCLE('J2000'," +
                "                TAP_UPLOAD.my_table.ra0," +
                "                TAP_UPLOAD.my_table.dec0," +
                "                0.01)" +
                "         )=1 " +
                " and (((mjd - mjd0) < 0.0001) " +
                " and ((mjd0 - mjd) < 0.0001))")

        result = kete.irsa.query_irsa_tap(query, upload_table=data)

    This is a blocking operation using TAP Async queries. This submits the query,
    receives a response from IRSA containing a URL, then queries that URL for job
    status. This continues until the job either completes or errors.

    Parameters
    ----------
    query :
        An SQL text query.
    upload_table :
        An optional pandas dataframe.
    base_url :
        The URL of the TAPS service to query, this defaults to IRSA.
    auth :
        An optional (username, password), this may be used to access restricted data.
    timeout :
        Timeout for web queries. This will raise an exception if the servers to not
        respond within this time.
    verbose :
        Print status responses as they are fetched from IRSA.
    """
    data = dict(FORMAT="CSV", QUERY=query)
    files = None
    if upload_table is not None:
        data["UPLOAD"] = "my_table,param:table.tbl"

        csv_output = io.StringIO()
        pd.DataFrame(upload_table).to_csv(csv_output, index=False)
        csv_output.seek(0)
        files = {"table.tbl": csv_output.read().encode()}

    submit = requests.post(
        base_url + "/TAP/async", data=data, files=files, auth=auth, timeout=timeout
    )
    submit.raise_for_status()

    tree = ElementTree.fromstring(submit.content.decode())
    element = tree.find("{*}results")

    urls = [v for k, v in element[0].attrib.items() if "href" in k]
    if len(urls) != 1:
        raise ValueError("Unexpected results: ", submit.content.decode())
    url = urls[0]

    phase_url = url.replace("results/result", "phase")

    status = requests.get(phase_url, timeout=timeout, auth=auth)
    status.raise_for_status()

    # Status results can have one of 4 outcomes:
    # QUEUED, EXECUTING, ERROR, COMPLETED

    start = time.time()
    time.sleep(0.15)
    delay = 0.85
    while status.content.decode().upper() in ["QUEUED", "EXECUTING"]:
        elapsed = time.time() - start
        # job is not complete
        if verbose:
            logger.info(
                f"IRSA response ({elapsed:0.1f} sec elapsed): %s",
                status.content.decode(),
            )
        time.sleep(delay)
        status = requests.get(phase_url, timeout=timeout, auth=auth)
        status.raise_for_status()

        # Increase time between queries until there is 30 seconds between.
        # Then continue forever.
        if delay < 30:
            delay += 1

    if status.content.decode().upper() != "COMPLETED":
        raise ValueError("Job Failed: ", status.content.decode())

    if verbose:
        logger.info(
            f"IRSA response ({elapsed:0.1f} sec elapsed): Completed, downloading..."
        )
    result = requests.get(url, timeout=timeout, auth=auth)
    result.raise_for_status()
    return pd.read_csv(io.StringIO(result.text))
