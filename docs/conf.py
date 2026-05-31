"""Sphinx configuration for SAbR."""

project = "SAbR"
copyright = "2026, Diego del Alamo"
author = "Diego del Alamo"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx_click",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
html_theme = "sphinx_rtd_theme"
autodoc_typehints = "description"
autodoc_member_order = "bysource"
sphinx_click_mock_imports = []
