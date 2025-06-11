from __future__ import annotations

import gzip
import json
import logging
import os
import shutil
import urllib

import requests
import requests.adapters
from urllib3.util.retry import Retry

from ._core import cache_path

logger = logging.getLogger(__name__)

__all__ = [
    "cache_path",
]


def _zip_existing(path):
    """
    Check if a file exists and is not zipped.
    Zip the file if possible, and delete the original.
    """
    if not os.path.isfile(path) or os.path.splitext(path)[1] in [".gz", ".fz"]:
        return
    logger.info(
        "Unzipped version of file found, zipping it before continuing. \n%s", path
    )
    with open(path, "rb") as f_in, gzip.open(path + ".gz", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(path)


def download_file(url, force_download=False, auto_zip=False, subfolder=""):
    """
    Download a file from the specified URL and return the path where it is saved.

    This operation is cached, so requesting the same URL will result in the previously
    fetched results being returned. Setting `force_download` to true will force the
    cached file to be re-downloaded.
    """
    parsed = urllib.parse.urlparse(url)
    filename = parsed.path.split("/")[-1]

    if auto_zip:
        _zip_existing(os.path.join(cache_path(subfolder), filename))

    if auto_zip and os.path.splitext(filename)[1] not in [".gz", ".fz"]:
        open_fn = gzip.open
        filename = filename + ".gz"
    else:
        open_fn = open
    path = os.path.join(cache_path(subfolder), filename)
    # check if already downloaded
    if not os.path.isfile(path) or force_download:
        with requests.Session() as session:
            retry = Retry(connect=3, backoff_factor=0.5)
            adapter = requests.adapters.HTTPAdapter(max_retries=retry)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            with session.get(url, stream=True, timeout=10) as res:
                res.raise_for_status()
                logger.info("Downloading file from (%s)", url)

                with open_fn(path, "wb") as f:
                    for chunk in res.iter_content(chunk_size=None):
                        if chunk:
                            f.write(chunk)
    else:
        logger.debug("Previously cached file (%s)", path)
    return path


def download_json(url, force_download=False, subfolder=""):
    """
    Download a gzipped json file from the specified URL.

    This operation is cached, so requesting the same URL will result in the previously
    fetched results being returned. Setting force_download to true will force the cached
    file to be re-downloaded.
    """
    filename = download_file(url, force_download, subfolder=subfolder, auto_zip=True)
    # unpack the gzip, then the json
    with gzip.open(filename, "rb") as f:
        raw_data = f.read().decode()
    return json.loads(raw_data)
