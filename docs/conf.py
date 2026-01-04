# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "kete"
copyright = "2026, Dar Dahlen"
author = "Dar Dahlen"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.doctest",
    "sphinx_gallery.gen_gallery",
    "matplotlib.sphinxext.plot_directive",
]

suppress_warnings = ["config.cache"]


autodoc_typehints = "description"
autodoc_inherit_docstrings = True
autodoc_warningiserror = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
}

autoclass_content = "both"
exclude_patterns = [
    "**/.ipynb_checkpoints",
]


# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/kete.svg"
html_favicon = "_static/favicon.png"


# -- Sphinx gallery settings --------------------------------------------------
sphinx_gallery_conf = {
    "run_stale_examples": True,
    "filename_pattern": "",
    "examples_dirs": ["../src/examples"],
}

keep_warnings = True

# -- doctest settings ----------------------------------------------------------

doctest_global_setup = """
import kete
import matplotlib.pyplot as plt
import numpy as np
from kete import *
"""

# -- Nitpick settings ----------------------------------------------------------
nitpicky = True
nitpick_ignore = [
    # Ignore links to external packages
    ("py:class", "numpy.dtype"),
    ("py:class", "numpy.floating"),
    ("py:class", "numpy.ndarray"),
    ("py:class", "ArrayLike"),
    ("py:class", "datetime.datetime"),
    ("py:class", "astropy.time.core.Time"),
    ("py:class", "numpy._typing._generic_alias.ScalarType"),
    ("py:class", "numpy.ma.core.MaskedArray"),
    ("py:class", "numpy.core.records.recarray"),
    ("py:class", "numpy._typing._array_like._ScalarType_co"),
]
