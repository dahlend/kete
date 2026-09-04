from __future__ import annotations

import gzip
import json
import logging
import os
import shutil
import time
import urllib

import requests

from ._core import cache_path

logger = logging.getLogger(__name__)

__all__ = [
    "cache_path",
]


def download_file(
    url,
    force_download=False,
    auto_zip=False,
    subfolder="",
    filename=None,
    auth=None,
    timeout=(10, 120),
    attempts=5,
):
    """
    Download a file from the URL and return the path where it is saved.

    This operation is cached. A call with the same URL returns the file which
    the first call saved. Set `force_download` to download the file again.

    The file enters the cache once it is complete. A download which stops part
    way is continued by the next call, if the server supports range requests.

    Parameters
    ----------
    url : str
        The URL to download.
    force_download : bool, optional
        Download the file again even if it is in the cache. Defaults to
        ``False``.
    auto_zip : bool, optional
        Compress the file after it is downloaded, unless it is already a
        compressed file. Defaults to ``False``.
    subfolder : str, optional
        Folder of the cache to save the file into. Defaults to the root of the
        cache.
    filename : str, optional
        Name to save the file under. The name in the URL is used if this is
        not given, so a name is needed when several URLs end in the same one.
    auth : tuple, optional
        A (username, password) pair for data which requires it.
    timeout : float or tuple, optional
        Connect and read timeouts for the download, in seconds. Defaults to
        ``(10, 120)``.
    attempts : int, optional
        Number of requests made without the download advancing before it is
        given up on. Defaults to ``5``.

    Returns
    -------
    str
        Path of the file in the cache.

    Raises
    ------
    ValueError
        If the download does not advance within `attempts` requests.
    requests.exceptions.HTTPError
        If the server refuses the request.
    """
    if filename is None:
        filename = urllib.parse.urlparse(url).path.split("/")[-1]
    folder = cache_path(subfolder)

    if auto_zip:
        _zip_existing(os.path.join(folder, filename))

    zip_after = auto_zip and os.path.splitext(filename)[1] not in [".gz", ".fz"]
    path = os.path.join(folder, filename + (".gz" if zip_after else ""))
    if os.path.isfile(path) and not force_download:
        logger.debug("Previously cached file (%s)", path)
        return path

    # A file compressed after the download is written under its own name first.
    # Compressing the bytes as they arrive would leave no way to compare them
    # against the length the server reported, and no way to continue from them.
    download_path = os.path.join(folder, filename) if zip_after else path
    part = download_path + ".part"

    logger.info("Downloading file from (%s)", url)
    total = None
    failures = 0
    resumable = True
    while True:
        pos = os.path.getsize(part) if os.path.exists(part) else 0
        # A server which reports no length gives nothing to check against, so
        # its response is accepted when the connection closes without an error.
        if total is not None and pos == total:
            break
        if total is not None and pos > total:
            # There is more data on disk than the server reports holding. What
            # is on disk cannot be a piece of this file.
            os.remove(part)
            pos = 0
        if failures >= attempts:
            raise ValueError(
                f"Failed to download {url}, {pos} of {total} bytes received."
            )
        headers = {"Accept-Encoding": "identity"}
        if pos > 0:
            headers["Range"] = f"bytes={pos}-"
        try:
            with requests.get(
                url, headers=headers, stream=True, timeout=timeout, auth=auth
            ) as res:
                if res.status_code == 416:
                    # The requested range is past the end of the file, so
                    # everything is already downloaded.
                    total = _range_total(res.headers.get("Content-Range"))
                    if total is None:
                        os.remove(part)
                    elif pos < total:
                        # The refused range is one the server holds. It does
                        # not serve the rest of the file.
                        failures += 1
                        time.sleep(min(30, 2**failures))
                    continue
                if res.status_code == 429 or res.status_code >= 500:
                    # The server is temporarily unable to send the file.
                    logger.info(
                        "Server responded with %s, trying again.", res.status_code
                    )
                    failures += 1
                    time.sleep(min(30, 2**failures))
                    continue
                res.raise_for_status()
                if res.status_code == 206:
                    total = _range_total(res.headers.get("Content-Range"))
                    mode = "ab"
                    logger.info("Continuing download from byte %s.", pos)
                else:
                    length = res.headers.get("Content-Length")
                    total = int(length) if length is not None else None
                    mode = "wb"
                    if pos > 0:
                        # The whole file came back from a range request, so
                        # what is on disk has to be discarded.
                        logger.info(
                            "Download cannot be continued from this server, "
                            "restarting it."
                        )
                        resumable = False
                with open(part, mode) as f:
                    for chunk in res.iter_content(chunk_size=2**20):
                        f.write(chunk)
                if total is None:
                    break
        except (
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as exc:
            logger.info("Download interrupted, continuing it. (%s)", exc)
        received = os.path.getsize(part) if os.path.exists(part) else 0
        failures = 0 if received > pos and resumable else failures + 1
        if failures > 0:
            time.sleep(min(30, 2**failures))

    os.replace(part, download_path)
    if zip_after:
        _zip_existing(download_path)
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


def _range_total(content_range):
    """
    Return the total length of the resource described by a Content-Range header.

    Returns None if the header is missing or cannot be parsed.
    """
    if content_range is None or "/" not in content_range:
        return None
    total = content_range.rsplit("/", maxsplit=1)[1].strip()
    return int(total) if total.isdigit() else None


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
    with open(path, "rb") as f_in, gzip.open(path + ".gz.part", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.replace(path + ".gz.part", path + ".gz")
    os.remove(path)
