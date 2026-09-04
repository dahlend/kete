"""
Query tools for TAP services such as IRSA
"""

from __future__ import annotations

import glob
import gzip
import hashlib
import io
import json
import logging
import os
import time
import urllib.parse
from functools import lru_cache
from xml.etree import ElementTree

import pandas as pd
import requests

from .cache import cache_path, download_file

__all__ = ["AsyncTapQuery", "tap_column_info", "query_tap", "TAP_SERVERS"]

IRSA_URL = "https://irsa.ipac.caltech.edu"
IRSA_TAP_URL = "https://irsa.ipac.caltech.edu/TAP/async"
CADC_TAP_URL = "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/argus/async"
GAIA_TAP_URL = "https://gaia.aip.de/tap/async"
MAST_TAP_URL = "https://mast.stsci.edu/vo-tap/api/v0.1/caom/async"


TAP_SERVERS = {
    "IRSA": IRSA_TAP_URL,
    "CADC": CADC_TAP_URL,
    "GAIA": GAIA_TAP_URL,
    "MAST": MAST_TAP_URL,
}
"""
Defined TAP servers for easy lookup.

If you have a TAP compliant server that you would like to add to this list, please
submit a github issue with the URL.
"""

logger = logging.getLogger(__name__)

STATUS_ATTEMPTS = 3
"""Number of responses which are not a phase before a job is given up on."""

TAP_PHASES = frozenset(
    {
        "PENDING",
        "QUEUED",
        "EXECUTING",
        "COMPLETED",
        "ERROR",
        "ABORTED",
        "UNKNOWN",
        "HELD",
        "SUSPENDED",
        "ARCHIVED",
    }
)
"""Job phases defined by the TAP standard."""


@lru_cache
def tap_column_info(table_name, service="IRSA", auth=None):
    """
    Retrieve the column data for a specified TAP table.

    This will return a dataframe containing the column properties of the target table.

    Parameters
    ----------
    table_name :
        The name of the table to query the columns of.
    service :
        The URL or known name of the TAPS service to query, this defaults to IRSA.
    auth :
        An optional (username, password), this may be used to access restricted data.
    """
    service = TAP_SERVERS.get(service.upper(), service)
    return query_tap(
        f"""SELECT * FROM TAP_SCHEMA.columns WHERE table_name='{table_name}'""",
        service=service,
        auth=auth,
    )


@lru_cache
def tap_table_info(service="IRSA", auth=None):
    """
    Retrieve the available tables provided by the specified TAP service.

    This will return a dataframe containing the tables available from the TAP service.

    Parameters
    ----------
    service :
        The URL or known name of the TAPS service to query, this defaults to IRSA.
    auth :
        An optional (username, password), this may be used to access restricted data.
    """
    service = TAP_SERVERS.get(service.upper(), service)
    return query_tap(
        """SELECT * FROM TAP_SCHEMA.tables""",
        service=service,
        auth=auth,
    )


def query_tap(
    query,
    upload_table=None,
    service="IRSA",
    auth=None,
    timeout=None,
    verbose=False,
    cache=True,
    update_cache=False,
):
    """
    Query TAP service, optionally upload a table which will be included in the
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
        column_information = kete.tap.tap_column_info(phase.source_table)

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

        result = kete.tap.query_tap(query, upload_table=data)

    This is a blocking operation using TAP Async queries. This submits the query,
    receives a response from the TAP service containing a URL, then queries that URL
    for job status. This continues until the job either completes or errors.

    By default, queries are cached, and any calls of this function with the same
    parameters will return the cached results. This may be disabled with the `cache`
    keywords, additionally the cache can be forcibly updated using `update_cache`.

    The id of the submitted job is recorded whether or not the results are
    cached. A query which stops before its results are downloaded is therefore
    continued by a later call with the same parameters. The query is not
    submitted a second time.

    Results are downloaded to disk as they arrive. A download which stops part
    way is continued from where it stopped, if the service supports range
    requests. When `cache` is false the record of the job is removed along with
    the results once they are returned.

    Parameters
    ----------
    query :
        An SQL text query.
    upload_table :
        An optional pandas dataframe.
    service :
        The URL or known name of the TAPS service to query, this defaults to IRSA.
    auth :
        An optional (username, password), this may be used to access restricted data.
    timeout :
        Timeout for web queries. This raises an exception if the servers do
        not respond within this time. Result downloads use the timeouts of
        :func:`kete.cache.download_file` if this is not specified. A download
        which stops receiving data is then retried.
    verbose :
        Print status responses as they are fetched from the TAP service.
    cache :
        Bool to indicate whether or not the query results should be kept in the
        cache. A query which is still running is recorded either way, so that it
        can be continued rather than submitted a second time.
    update_cache :
        This value can specify if the cache is forcibly updated. IE: previous
        query results are ignored and resubmitted to the TAP service.
    """
    service = TAP_SERVERS.get(service.upper(), service)
    query = AsyncTapQuery(
        query=query,
        upload_table=upload_table,
        service=service,
        auth=auth,
        timeout=timeout,
        verbose=verbose,
        cache=cache,
        update_cache=update_cache,
    )
    return query.query_blocking()


class AsyncTapQuery:
    """
    Async Tap Queries

    Parameters are the same as the `query_tap` function.

    This allows for jobs to be submitted without blocking.
    """

    def __init__(
        self,
        query,
        upload_table=None,
        service="IRSA",
        auth=None,
        timeout=None,
        verbose=False,
        cache=True,
        update_cache=False,
    ):
        base_url = TAP_SERVERS.get(service.upper(), service)
        self.query = " ".join(query.strip().split())
        self.upload_table = upload_table
        self.base_url = base_url
        self.auth = auth
        self.timeout = timeout
        self.verbose = verbose
        self.update_cache = update_cache
        self.cache = cache

        self.data = dict(FORMAT="csv", QUERY=query, LANG="ADQL", REQUEST="doQuery")
        files = None
        if upload_table is not None:
            self.data["UPLOAD"] = "my_table,param:table.tbl"

            csv_output = io.StringIO()
            pd.DataFrame(upload_table).to_csv(csv_output, index=False)
            csv_output.seek(0)
            files = {"table.tbl": csv_output.read().encode()}
        self._files = files
        if files is not None:
            _hash = int(
                hashlib.md5(
                    str((base_url, query, tuple(files.values()))).encode()
                ).hexdigest(),
                16,
            )
        else:
            _hash = int(hashlib.md5(str((base_url, query)).encode()).hexdigest(), 16)
        self._hash = str(abs(_hash))[:16]
        path = cache_path(sub_path="tap")
        path = os.path.join(path, f"{self._hash[:3]}")
        if not os.path.isdir(path):
            os.makedirs(path)
        job_path = os.path.join(path, f"{self._hash}.json.gz")
        resp_path = os.path.join(path, f"{self._hash}.parquet")

        self._cache_subfolder = os.path.join("tap", self._hash[:3])
        self._cache_dir = path
        self.job_path = job_path
        self.resp_path = resp_path

        if update_cache:
            self.clear_cache()
        if not os.path.exists(job_path):
            with gzip.open(job_path, "wb") as f:
                f.write(json.dumps({"status": "NOT_SUBMITTED"}).encode())

        # Recover the id of a job which was submitted by a previous call. A job
        # which still runs, or whose results the server still holds, is then
        # reused rather than submitted a second time. A record which cannot be
        # read is treated as there being no job. The record is written
        # repeatedly while a job runs, so a stopped call can leave a partial
        # one.
        self._job_id = None
        try:
            with gzip.open(job_path, "rb") as f:
                self._job_id = json.loads(f.read().decode()).get("job_id")
        except (OSError, EOFError, ValueError) as exc:
            logger.debug("Could not read the cached job record: %s", exc)

    def query_blocking(self):
        """
        Submit the query to the TAP service, and block until the results are returned.
        """
        start = time.time()
        status = self.query_status()
        if status == "NOT_SUBMITTED":
            self.submit()
            status = "QUEUED"
        elif status == "PENDING":
            # A previous call created the job but stopped before starting it.
            # It is started here rather than waited on, because a job in this
            # phase does not run.
            self._start()
            status = "QUEUED"

        delay = 0.05
        last_print = 0
        while status in ["QUEUED", "EXECUTING"]:
            cur_time = time.time()
            elapsed = cur_time - start
            time.sleep(delay)
            status = self.query_status()

            # Increase time between queries until there is 30 seconds between.
            # Then continue forever.
            if elapsed < 2:
                pass
            elif delay < 3:
                delay += 0.05
            elif delay < 30:
                delay += 1
            if self.verbose and abs(cur_time - last_print) > 3:
                logger.info(
                    f"TAP response ({elapsed:0.1f} sec elapsed): %s",
                    status,
                )
                last_print = cur_time
        if status == "ERROR":
            raise ValueError("Job Failed: ", self.query_error())
        if status == "NOT_SUBMITTED":
            raise ValueError(
                "The TAP service no longer holds this job. Run the query again "
                "to submit it a second time."
            )
        if status != "COMPLETED":
            raise ValueError(f"Job did not complete, it reported: {status}")

        return self.result()

    def submit(self):
        """
        Submit the job to the TAP service.

        If the job results already exist in the cache, this will skip submission.
        """
        if self.cache and os.path.exists(self.resp_path):
            if self.verbose:
                logger.info(
                    (
                        "TAP query has already been completed and saved ",
                        "to cache, not submitting.",
                    ),
                )
            return

        submit = requests.post(
            self.base_url,
            data=self.data,
            files=self._files,
            auth=self.auth,
            timeout=self.timeout,
        )
        submit.raise_for_status()

        tree = ElementTree.fromstring(submit.content.decode())
        element = tree.find("{*}jobId")
        if element is not None:
            self._job_id = element.text.strip()
        else:
            raise ValueError(submit.content.decode())

        # Results partially downloaded from a job which this one replaces can be
        # large, and are no longer of any use.
        for old in glob.glob(os.path.join(self._cache_dir, f"{self._hash}.*.csv*")):
            os.remove(old)

        with gzip.open(self.job_path, "rb") as f:
            status_file = json.loads(f.read().decode())
        status_file["status"] = "QUEUED"
        status_file["job_id"] = self._job_id
        with gzip.open(self.job_path, "wb") as f:
            f.write(json.dumps(status_file).encode())

        if self.query_status() == "PENDING":
            self._start()

    def query_status(self):
        """
        Query the status from the TAP service. If the job results already exist in
        the cache, then this will return as COMPLETED without querying.

        Status results can have one of outcomes:
        NOT_SUBMITTED, QUEUED, PENDING, EXECUTING, ERROR, COMPLETED
        """
        if self.cache and os.path.exists(self.resp_path):
            return "COMPLETED"
        if self._status_url is None:
            status = "NOT_SUBMITTED"
        else:
            # Services report a job they no longer hold in several ways. Some
            # send a 404, and some send a document which describes the error. A
            # service which is briefly unwell sends whatever its front end
            # produces. The request is repeated before the job is treated as
            # gone, so that one bad response does not abandon a running job.
            status = None
            for attempt in range(STATUS_ATTEMPTS):
                response = requests.get(
                    self._status_url, timeout=self.timeout, auth=self.auth
                )
                if response.status_code in (404, 410):
                    break
                response.raise_for_status()
                phase = response.content.decode().strip().upper()
                if phase in TAP_PHASES:
                    status = phase
                    break
                logger.debug("Job status response: %s", phase)
                if attempt + 1 < STATUS_ATTEMPTS:
                    time.sleep(2**attempt)

            if status is None:
                logger.info(
                    "Job %s is no longer available from the TAP service. It is "
                    "submitted again.",
                    self._job_id,
                )
                self._job_id = None
                status = "NOT_SUBMITTED"

        with gzip.open(self.job_path, "rb") as f:
            status_file = json.loads(f.read().decode())
        status_file["status"] = status
        status_file["job_id"] = self._job_id
        with gzip.open(self.job_path, "wb") as f:
            f.write(json.dumps(status_file).encode())
        return status

    def result(self):
        """
        Fetch the finished results from the TAP service, caching the results if
        requested.

        If the results already exist in the cache, then they are returned without
        submitting any queries to the TAP service.

        Returns a Pandas Dataframe of the query results.
        """
        if self.cache and os.path.exists(self.resp_path):
            return pd.read_parquet(self.resp_path)

        if self._job_id is None:
            raise ValueError("No job has been submitted.")

        if self.verbose:
            logger.info("Downloading results...")
        # The results are downloaded under a name of their own. The URL they
        # come from ends in the same name for every job.
        job_id = "".join(c for c in self._job_id if c.isalnum() or c in "-_")
        # The timeout of the query is for the requests which drive the job. A
        # download which was not given one uses the default of `download_file`.
        # A connection which stops delivering data is then retried.
        timeout = {"timeout": self.timeout} if self.timeout is not None else {}
        path = download_file(
            self._download_url(),
            subfolder=self._cache_subfolder,
            filename=f"{self._hash}.{job_id}.csv",
            auth=self.auth,
            **timeout,
        )

        with open(path, "rb") as f:
            head = f.read(2000)
        if head.startswith(b"<"):
            # Results are requested as CSV. A document in their place
            # describes a problem with the job and holds no data.
            os.remove(path)
            raise ValueError(
                f"TAP service returned no results: {head.decode(errors='replace')}"
            )

        try:
            result = pd.read_csv(path)
        except (pd.errors.ParserError, UnicodeDecodeError):
            # The download is the length the service reported, but it cannot
            # be read. It is discarded, because a later call resumes a complete
            # file and fails to read it again.
            os.remove(path)
            raise

        if self.cache:
            result.to_parquet(self.resp_path, index=False)
            if self.verbose:
                logger.info("Results saved to cache.")
        os.remove(path)

        if not self.cache:
            # The results are not kept, and neither is the record of the job
            # which produced them.
            self.clear_cache()

        if self.verbose:
            logger.info("Download complete.")
        return result

    def clear_cache(self):
        """
        Delete all cached files associated with this query.

        This removes the record of the submitted job along with any downloaded
        results. A later query is submitted to the TAP service again.
        """
        for file in glob.glob(os.path.join(self._cache_dir, f"{self._hash}.*")):
            os.remove(file)

    def query_error(self):
        if self.query_status() != "ERROR":
            raise ValueError("Job has not failed.")

        submit = requests.get(
            self._error_url,
            auth=self.auth,
            timeout=self.timeout,
        )
        submit.raise_for_status()
        return submit.content.decode()

    def _start(self):
        """
        Ask the TAP service to begin running a job which has been created.

        A service which begins a job as soon as it is submitted reports the job
        as queued rather than pending. Such a service does not need this.
        """
        requests.post(
            self.base_url + "/" + self._job_id + "/phase",
            data={"PHASE": "RUN"},
            auth=self.auth,
            timeout=self.timeout,
        ).raise_for_status()

    def _download_url(self):
        """
        Return the URL which the results are downloaded from.

        A range request is what allows a stopped download to be continued. The
        results URL of a TAP service does not necessarily support one.

        IRSA also publishes the results of a job as a static file, and that file
        does support range requests. It is used when it is available and reports
        the same size as the results URL. Other services use their results URL,
        and restart a download which stops part way.
        """
        url = self._result_url
        if not self.base_url.startswith(IRSA_URL):
            return url
        try:
            job = requests.get(
                self.base_url + "/" + self._job_id,
                timeout=self.timeout,
                auth=self.auth,
            )
            job.raise_for_status()
            tree = ElementTree.fromstring(job.content.decode())
            identifier = None
            for param in tree.findall(".//{*}parameter"):
                if param.get("id") == "query_identifier" and param.text is not None:
                    identifier = param.text.strip()
                    break
            if identifier is None:
                return url

            root = urllib.parse.urlparse(self.base_url)
            static = (
                f"{root.scheme}://{root.netloc}/pubspace/{identifier}/{self._job_id}"
            )
            head = requests.head(static, timeout=self.timeout, auth=self.auth)
            if head.status_code != 200 or head.headers.get("Accept-Ranges") != "bytes":
                return url

            # The static file is used only when it is the size the results URL
            # reports. A file of a different size is not the results of the job.
            length = requests.head(
                url,
                headers={"Accept-Encoding": "identity"},
                timeout=self.timeout,
                auth=self.auth,
            ).headers.get("Content-Length")
            if length is not None and length != head.headers.get("Content-Length"):
                return url
            if self.verbose:
                logger.info("Downloading from %s, which supports resuming.", static)
            return static
        except (requests.exceptions.RequestException, ElementTree.ParseError) as exc:
            # Locating the static file is optional. The results URL is used
            # when it cannot be located.
            logger.debug("Could not locate a resumable download URL: %s", exc)
            return url

    @property
    def _result_url(self):
        if self._job_id is None:
            return None
        return self.base_url + "/" + self._job_id + "/results/result"

    @property
    def _status_url(self):
        if self._job_id is None:
            return None
        return self.base_url + "/" + self._job_id + "/phase"

    @property
    def _error_url(self):
        if self._job_id is None:
            return None
        return self.base_url + "/" + self._job_id + "/error"
