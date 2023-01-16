# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

# sys.path.insert(0, os.path.abspath("../../pedpy/methods"))
# sys.path.insert(0, os.path.abspath("../../pedpy/io"))
# sys.path.insert(0, os.path.abspath("../../pedpy/data"))
sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "PedPy"
copyright = "2023, Forschungszentrum Jülich GmbH, IAS-7"
author = "Tobias Schrödter"
release = "1.0.0b1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
    "sphinx_design",
    "sphinx_copybutton",
    'sphinx.ext.viewcode',
    'autoapi.extension'
]

autoapi_dirs = ['../../pedpy']
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autoapi_keep_files = True
autodoc_typehints = "both"

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]

nbsphinx_allow_errors = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_logo = "_static/logo.png"

html_css_files = [
    "css/custom.css",
]
html_context = {"default_mode": "light"}

html_favicon = "_static/logo.png"

html_theme_options = {
    "show_nav_level": 5,
    "github_url": "https://github.com/PedestrianDynamics/pedpy",
    "header_links_before_dropdown": 5,
    "show_toc_level": 5,
    "navbar_end": ["navbar-icon-links"],
}

# -- Options for EPUB output
epub_show_urls = "footnote"
