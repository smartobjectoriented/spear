# -*- coding: utf-8 -*-
#
# SPEAR documentation build configuration.
#
# Modelled on the SO3 documentation (soo/so3/doc) so both projects share the
# same look, the same drawio-based diagram pipeline and the same build targets.

import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath('.'))

# -- General configuration -------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.extlinks',
    'sphinx.ext.imgmath',
]

# Diagrams are authored in source/img/spear.drawio (multi-page, editable) and
# exported to PNG. The .drawio file is itself generated from
# source/img/gen_spear_diagrams.py; export with source/img/export_png.sh.

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'

project = u'SPEAR'
copyright = u'2026, HEIG-VD/REDS'

# The served model is the one thing that dates this documentation fastest, so
# the version string is derived from the active model profile rather than being
# maintained by hand.


def _spear_version():
    conf = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'spear', 'active-model.conf')
    try:
        with open(conf) as fh:
            model = os.path.basename(fh.read().strip())
    except OSError:
        return 'unknown'
    return model.replace('.gguf', '') or 'unknown'


release = _spear_version()
version = release

language = u'en'
numfig = True

exclude_patterns = []
pygments_style = 'sphinx'

# -- Options for HTML output -----------------------------------------------

html_theme = u'default'

try:
    import sphinx_rtd_theme  # noqa: F401
    html_theme = 'sphinx_rtd_theme'

    def setup(app):
        app.add_css_file('style.css')
except ImportError:
    sys.stderr.write(
        "Warning: the 'sphinx_rtd_theme' HTML theme was not found. "
        "Falling back to the default theme.\n")

html_theme_options = {'body_max_width': '100%'}

html_static_path = ['_static']
html_css_files = ['theme_overrides.css']

html_last_updated_fmt = '%d %b %Y, %H:%M'
html_show_sourcelink = False
html_show_sphinx = False
html_show_copyright = True
html_file_suffix = None
htmlhelp_basename = 'speardoc'

# -- Options for LaTeX output ----------------------------------------------

latex_paper_size = u'a4'
latex_font_size = u'10pt'
latex_documents = [
    ('index', 'SPEAR.tex', u'SPEAR Documentation', u'HEIG-VD/REDS', 'manual'),
]
latex_elements = {'babel': '\\usepackage[english]{babel}'}
latex_use_parts = False

# -- Options for manual page output ----------------------------------------

man_pages = [('index', 'spear', u'SPEAR Documentation',
              [u'Daniel Rossier'], 1)]

# -- Additional options ----------------------------------------------------

todo_include_todos = True
