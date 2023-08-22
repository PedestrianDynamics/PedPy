# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import datetime
import os
import sys

basedir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "pedpy")
)
sys.path.insert(0, basedir)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

current_year = datetime.datetime.today().year

project = "PedPy"
copyright = f"{current_year}, Forschungszentrum Jülich GmbH, IAS-7"

import pedpy

release = pedpy.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "myst_nb",
    "sphinx_favicon",
    "notfound.extension",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autodoc",
    "autoapi.extension",
]

# -- Automatic execution of jupyter notebooks --------------------------------
nb_execution_excludepatterns = ["readthedocs.ipynb"]

# -- Automatic generation of API doc -----------------------------------------
autoapi_dirs = [
    "../../pedpy",
]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
]
autoapi_ignore = ["**/tests/**"]
autoapi_member_order = ["groupwise"]
autodoc_typehints = "description"
autoapi_generate_api_docs = False
autoapi_python_class_content = "both"
autoapi_member_order = "bysource"

add_module_names = False

# -- Linking ---------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "shapely": ("https://shapely.readthedocs.io/en/2.0.1/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}
intersphinx_disabled_domains = ["std"]

# -- HTML generation ---------------------------------------------------------
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "readthedocs.ipynb",
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

html_logo = "_static/logo_text.svg"
html_favicon = "_static/logo.svg"

html_css_files = ["css/custom.css", "css/breadcrumbs_custom"]
html_context = {"default_mode": "light"}

html_theme_options = {
    "show_nav_level": 5,
    "show_toc_level": 2,
    "use_fullscreen_button": False,
    "use_issues_button": False,
    "use_download_button": False,
    "article_header_end": ["breadcrumbs", "toggle-secondary-sidebar"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/PedestrianDynamics/PedPy",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/pedpy/",
            "icon": "https://img.shields.io/pypi/v/PedPy",
            "type": "url",
        },
    ],
}

html_sidebars = {
    "**": ["navbar-logo", "icon-links", "search-field", "sbt-sidebar-nav.html"]
}
