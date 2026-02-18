import sphinx_rtd_theme
import sys
import os
import re

# sys.path.append('/home/limozin/Documents/GitHub/rtd-tutorial')
sys.path.insert(0, os.path.abspath("./../../"))
sys.path.insert(0, os.path.abspath("./../../examples/"))

VERSIONFILE = os.path.abspath("./../../celldetective/_version.py")
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__\s*=\s*(?:version\s*=\s*)?['\"]([^'\"]*)['\"]"

mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))


# Configuration file for the Sphinx documentation builder.

# -- Project information

project = "celldetective"
copyright = "2025, Laboratoire Adhésion et Inflammation (LAI) U1067 INSERM CNRS"
author = "Rémy Torro"

release = verstr
version = verstr  #'1.2.2.post2'

# -- General configuration

extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "hoverxref.extension",
    "nbsphinx_link",
    "sphinxcontrib.jquery",
]

autosummary_generate = True

hoverxref_autoref = True
hoverxref_roles = ["term"]
hoverxref_domains = ["std"]

hoverxref_role_types = {
    "term": "tooltip",
}

hoverxref_intersphinx = [
    "https://docs.python.org/3/",
]

autoapi_dirs = ["celldetective"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
# intersphinx_disabled_domains = ['std']

templates_path = ["_templates"]

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/logo.png"
html_css_files = [
    "https://cdn.jsdelivr.net/npm/@mdi/font@7.4.47/css/materialdesignicons.min.css",
    #'https://fonts.googleapis.com/icon?family=Material+Icons',
    #'custom.css',
]
html_favicon = "_static/favicon.png"
html_theme_options = {"style_nav_header_background": "#b9c3cb"}

# -- Options for EPUB output
epub_show_urls = "footnote"

from html import escape
from docutils import nodes
from docutils.parsers.rst import roles

# def material_icon(name, rawtext, text, lineno, inliner, options={}, content=[]):
#     # Parse the input: icon_name[,color[,size]]
#     parts = [p.strip() for p in text.split(",")]
#     icon_name = parts[0]
#     color = parts[1] if len(parts) > 1 else "inherit"
#     size = parts[2] if len(parts) > 2 else "1em"
#
#     # Escape HTML-sensitive content
#     icon_html = escape(icon_name)
#
#     html = (
#         f'<span class="material-icons" '
#         f'style="vertical-align: middle; color: {color}; font-size: {size};">'
#         f'{icon_html}</span>'
#     )
#     node = nodes.raw('', html, format='html')
#     return [node], []
#
# roles.register_local_role('mdi', material_icon)


def blue_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    html = f'<span style="color: #1565c0;">{text}</span>'
    return [nodes.raw("", html, format="html")], []


roles.register_local_role("blue", blue_role)


def mdi_icon(name, rawtext, text, lineno, inliner, options={}, content=[]):
    # Parse “icon_name[,color[,size]]”
    parts = [p.strip() for p in text.split(",")]
    icon_name = parts[0]
    color = parts[1] if len(parts) > 1 else "inherit"
    size = parts[2] if len(parts) > 2 else "1em"

    # Escape icon_name in case
    icon_html = escape(icon_name)

    html = (
        f'<span class="mdi mdi-{icon_html}" '
        f'style="vertical-align: middle; color: {color}; font-size: {size};">'
        f"</span>"
    )
    node = nodes.raw("", html, format="html")
    return [node], []


roles.register_local_role("icon", mdi_icon)
