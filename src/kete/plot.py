from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.visualization import (
    AsymmetricPercentileInterval,
    ImageNormalize,
    LinearStretch,
    PowerDistStretch,
    ZScaleInterval,
)
from astropy.wcs import WCS

from .vector import Vector

__all__ = ["plot_fits_image", "zoom_plot", "annotate_plot"]


def plot_fits_image(fit, percentiles=(0.1, 99.95), power_stretch=0.5, cmap="gray"):
    """
    Plot a FITS image, returning a WCS object which may be used to plot future points
    correctly onto the current image.

    This estimates the standard deviation, subtracts the median, and scales the
    displayed image by number of standard deviations from the median value.

    This returns the WCS which is constructed during the plotting process.

    This will use the existing matplotlib plotting axis if available.

    Parameters
    ----------
    fit:
        Fits file from Astropy.
    percentiles :
        Statistical percentile limit for which data to plot. By default this is set
        to 0.1% and 99.95%. If this is set to `None`, then this uses Astropy's
        `ZScaleInterval`.
    power_stretch :
        The scaling of the intensity of the plot is a power law, this defines the power
        of that power law. By default plots are sqrt scaled. If this is set to 1, then
        this becomes a linear scaling.
    cmap :
        Color map to use for the plot.
    """

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        wcs = WCS(fit.header)

    if not plt.get_fignums():
        plt.figure(dpi=200, figsize=(6, 6), facecolor="w")

    # This is a little fancy footwork to get this to play nice with subplots
    # This gets the current subplot geometry, pulls up the current axis and
    # nukes it. Then in its place it inserts a new axis with the correct
    # projection.
    fig = plt.gcf()
    rows, cols, start, _ = plt.gca().get_subplotspec().get_geometry()
    fig.axes[start].remove()
    ax = fig.add_subplot(rows, cols, start + 1, projection=wcs)
    ax.set_aspect("equal", adjustable="box")
    fig.axes[start] = ax

    stretch = LinearStretch() if power_stretch == 1 else PowerDistStretch(power_stretch)

    if percentiles is None:
        interval = ZScaleInterval()
    else:
        interval = AsymmetricPercentileInterval(*percentiles)

    norm = ImageNormalize(fit.data, interval=interval, stretch=stretch)
    data = np.nan_to_num(fit.data, nan=np.nanpercentile(fit.data, 5))

    ax.imshow(data, origin="lower", norm=norm, cmap=cmap)
    ax.set_xlabel("RA")
    ax.set_ylabel("DEC")
    ax.set_aspect("equal", adjustable="box")
    return wcs


def _ra_dec(ra, dec=None):
    """
    Given either a RA/Dec pair, or Vector, return an RA/Dec pair.
    """
    if isinstance(ra, Vector):
        vec = ra.as_equatorial
        ra = vec.ra
        dec = vec.dec
    return ra, dec


def zoom_plot(wcs, ra, dec=None, zoom=100):
    """
    Given a WCS, zoom the current plot to the specified RA/Dec.

    Parameters
    ----------
    wcs :
        An Astropy World Coordinate system from the image.
    ra :
        The RA in degrees, can be a `Vector`, if so then dec is ignored.
    dec :
        The DEC in degrees.
    zoom :
        Optional zoom region in pixels
    """
    ra, dec = _ra_dec(ra, dec)
    pix = wcs.world_to_pixel(SkyCoord(ra, dec, unit="deg"))
    plt.gca().set_xlim(pix[0] - zoom, pix[0] + zoom)
    plt.gca().set_ylim(pix[1] - zoom, pix[1] + zoom)


def annotate_plot(
    wcs,
    ra,
    dec=None,
    text=None,
    px_gap=70,
    length=50,
    lw=1,
    c="red",
    text_color="White",
    style="+",
    text_dx=0,
    text_dy=0,
    text_fs=None,
):
    """
    Add an annotation for a point in a FITS plot, this requires a world coordinate
    system (wcs) as returned by the plotting function above.

    Parameters
    ----------
    wcs :
        An Astropy World Coordinate system from the image.
    ra :
        The RA in degrees, can be a `Vector`, if so then dec is ignored.
    dec :
        The DEC in degrees.
    text :
        Optional text to display.
    px_gap :
        How many pixels should the annotation be offset from the specified RA/DEC.
    length :
        Length of the bars in pixels.
    lw :
        Line width of the marker.
    c :
        Color of the marker, uses matplotlib colors.
    text_color :
        If text is provided, this defines the text color.
    style :
        Style of marker, this can be either "o", "+", or "L".
    text_dx :
        Offset of the text x location in pixels.
    text_dy :
        Offset of the text y location in pixels.
    text_fs :
        Text font size.
    """
    ra, dec = _ra_dec(ra, dec)
    x, y = wcs.world_to_pixel(SkyCoord(ra, dec, unit="deg"))
    total = length + px_gap
    if style == "+":
        plt.plot([x - total, x - px_gap], [y, y], c=c, lw=lw)
        plt.plot([x + px_gap, x + total], [y, y], c=c, lw=lw)
        plt.plot([x, x], [y - px_gap, y - total], c=c, lw=lw)
        plt.plot([x, x], [y + px_gap, y + total], c=c, lw=lw)
    elif style == "L":
        plt.plot([x + px_gap, x + total], [y, y], c=c, lw=lw)
        plt.plot([x, x], [y + px_gap, y + total], c=c, lw=lw)
    elif style == "o":
        plt.scatter(x, y, fc="None", ec=c, s=5 * px_gap, lw=lw)
    else:
        raise ValueError("Style is not recognized, must be one of: o, +, L")

    if text:
        plt.text(x + text_dx, y + text_dy, text, c=text_color, fontsize=text_fs)
