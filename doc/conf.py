#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# pylint: skip-file

import os
import re
import sys

try:
    import neutronpy
except ImportError:
    raise RuntimeError('Cannot import neutronpy, it must be installed before building documentation. Please investigate.')

from distutils.version import LooseVersion
import sphinx
if LooseVersion(sphinx.__version__) < LooseVersion('1'):
    raise RuntimeError('Need sphinx >= 1 for numpydoc to work correctly')

needs_sphinx = '1.0'

# -----------------------------------------------------------------------------
# releases (changelog) configuration
# -----------------------------------------------------------------------------
releases_issue_uri = "https://github.com/neutronpy/neutronpy/issues/%s"
releases_release_uri = "https://github.com/neutronpy/neutronpy/tree/%s"
releases_github_path = "neutronpy/neutronpy"

# -----------------------------------------------------------------------------
# General configuration
# -----------------------------------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

sys.path.insert(1, os.path.abspath('sphinxext'))

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.coverage',
              'sphinx.ext.pngmath',
              'sphinx.ext.intersphinx',
              'matplotlib.sphinxext.plot_directive',
              'numpydoc',
              'releases',
#               'autodoc_cython',
              ]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The encoding of source files.
source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'neutronpy'
copyright = '2014, David M Fobes'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = neutronpy.__version__.split('-')[0]
# The full version, including alpha/beta/rc tags.
release = neutronpy.__version__

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = []
exclude_dirs = []

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = 'autolink'

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
# show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx_rtd_theme_custom.support.LightStyle'

# -----------------------------------------------------------------------------
# HTML output
# -----------------------------------------------------------------------------

import sphinx_rtd_theme_custom
html_theme = "sphinx_rtd_theme_custom"
html_theme_path = [sphinx_rtd_theme_custom.get_html_theme_path()]

html_static_path = [os.path.join('.', '_static')]

html_theme_options = {'logo': 'logo.png',
                      'logo_name': True,
                      'logo_text_align': 'center',
                      'description': "",
                      'github_user': 'neutronpy',
                      'github_repo': 'neutronpy',
                      'travis_button': True,
                      'github_banner': True,
                      'link': '#3782BE',
                      'link_hover': '#3782BE',
                      'sidebar_includehidden': True, 
                      #'pygments_style': pygments_style,
                      }

# Sister-site links to API docs
# html_theme_options['extra_nav_links'] = {
#     "NeutronPy Docs": 'http://neutronpy.github.io/reference',
# }

# Add any extra paths that contain custom files (such as robots.txt or
# .htaccess) here, relative to this directory. These files are copied
# directly to the root of the documentation.
# html_extra_path = []

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
# html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
html_sidebars = {'**': ['about.html',
                        'navigation.html',
                        'searchbox.html',
                        'donate.html']}

# Additional templates that should be rendered to pages, maps page names to
# template names.
# html_additional_pages = {'index': 'indexcontent.html',}

# If false, no module index is generated.
# html_domain_indices = True

# If false, no index is generated.
# html_use_index = True

# If true, the index is split into individual pages for each letter.
# html_split_index = True

# If true, links to the reST sources are added to the pages.
# html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
# html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
# html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
# html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
# html_file_suffix = None

# Output file base name for HTML help builder.
htmlhelp_basename = 'neutronpydoc'

# -----------------------------------------------------------------------------
# LaTeX output
# -----------------------------------------------------------------------------

# The paper size ('letter' or 'a4').
# latex_paper_size = 'letter'

# The font size ('10pt', '11pt' or '12pt').
# latex_font_size = '10pt'

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, document class [howto/manual]).
_stdauthor = 'David M Fobes'
latex_documents = [
  ('reference/index', 'neutronpy-ref.tex', 'NeutronPy Reference',
   _stdauthor, 'manual'),
]

# Additional stuff for the LaTeX preamble.
latex_preamble = r'''
\usepackage{amsmath}
\DeclareUnicodeCharacter{00A0}{\nobreakspace}

% In the parameters section, place a newline after the Parameters
% header
\usepackage{expdlist}
\let\latexdescription=\description
\def\description{\latexdescription{}{} \breaklabel}

% Make Examples/etc section headers smaller and more compact
\makeatletter
\titleformat{\paragraph}{\normalsize\py@HeaderFamily}%
            {\py@TitleColor}{0em}{\py@TitleColor}{\py@NormalColor}
\titlespacing*{\paragraph}{0pt}{1ex}{0pt}
\makeatother

% Fix footer/header
\renewcommand{\chaptermark}[1]{\markboth{\MakeUppercase{\thechapter.\ #1}}{}}
\renewcommand{\sectionmark}[1]{\markright{\MakeUppercase{\thesection.\ #1}}}
'''

# Documents to append as an appendix to all manuals.
# latex_appendices = []

# If false, no module index is generated.
latex_use_modindex = False


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('index', 'neutronpy', 'neutronpy Documentation',
     ['David M Fobes'], 1)
]

# If true, show URL addresses after external links.
# man_show_urls = False


# -----------------------------------------------------------------------------
# Texinfo output
# -----------------------------------------------------------------------------

texinfo_documents = [
  ("contents", 'numpy', 'Numpy Documentation', _stdauthor, 'Numpy',
   "NumPy: array processing for numbers, strings, records, and objects.",
   'Programming',
   1),
]

# -----------------------------------------------------------------------------
# Intersphinx configuration
# -----------------------------------------------------------------------------

intersphinx_mapping = {'http://docs.python.org/dev': None}


# -----------------------------------------------------------------------------
# Autosummary
# -----------------------------------------------------------------------------
# from sphinx.ext import autodoc
# class DocsonlyMethodDocumenter(autodoc.MethodDocumenter):
#     def format_args(self):
#         return None
#
# autodoc.add_documenter(DocsonlyMethodDocumenter)

import glob
numpydoc_show_class_members = False
# autodoc_default_flags = ['members']
autodoc_docstring_signature = True
autosummary_generate = glob.glob("reference/*.rst")

# -----------------------------------------------------------------------------
# Coverage checker
# -----------------------------------------------------------------------------
coverage_ignore_modules = r"""
    """.split()
coverage_ignore_functions = r"""
    test($|_) (some|all)true bitwise_not cumproduct pkgload
    generic\.
    """.split()
coverage_ignore_classes = r"""
    """.split()

coverage_c_path = []
coverage_c_regexes = {}
coverage_ignore_c_items = {}

# -----------------------------------------------------------------------------
# Source code links
# -----------------------------------------------------------------------------

import inspect
from os.path import relpath, dirname

for name in ['sphinx.ext.linkcode', 'numpydoc.linkcode']:
    try:
        __import__(name)
        extensions.append(name)
        break
    except ImportError:
        pass
else:
    print("NOTE: linkcode extension not found -- no links to source generated")

def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object
    """
    if domain != 'py':
        return None

    modname = info['module']
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except:
            return None

    try:
        fn = inspect.getsourcefile(obj)
    except:
        fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.findsource(obj)
    except:
        lineno = None

    if lineno:
        linespec = "#L%d" % (lineno + 1)
    else:
        linespec = ""

    fn = relpath(fn, start=dirname(neutronpy.__file__))

    if 'dev' in neutronpy.__version__:
        return "http://github.com/neutronpy/neutronpy/blob/master/neutronpy/%s%s" % (
           fn, linespec)
    else:
        return "http://github.com/neutronpy/neutronpy/blob/v%s/neutronpy/%s%s" % (
           neutronpy.__version__, fn, linespec)
