"""
Previously this was IRSA specific tools, but it has been generalized and broken
up into `kete.tap` and `kete.plot`.

This will be removed in a future version.
"""

from __future__ import annotations

from .deprecation import rename
from .plot import annotate_plot, plot_fits_image, zoom_plot
from .tap import query_tap, tap_column_info

# rename the function to match the new location
plot_fits_image = rename(
    plot_fits_image,
    "2.0.0",
    old_name="plot_fits_image",
    additional_msg="Use `kete.plot.plot_fits_image` instead.",
)

zoom_plot = rename(
    zoom_plot,
    "2.0.0",
    old_name="zoom_plot",
    additional_msg="Use `kete.plot.zoom_plot` instead.",
)

annotate_plot = rename(
    annotate_plot,
    "2.0.0",
    old_name="annotate_plot",
    additional_msg="Use `kete.plot.annotate_plot` instead.",
)

query_column_data = rename(
    tap_column_info,
    "2.0.0",
    old_name="query_column_data",
    additional_msg="Use `kete.tap.tap_column_info` instead.",
)

query_irsa_tap = rename(
    query_tap,
    "2.0.0",
    old_name="query_irsa_tap",
    additional_msg="Use `kete.tap.query_tap` instead.",
)
