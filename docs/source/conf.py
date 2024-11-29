# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Uni-Mol'
copyright = '2023, cuiyaning'
author = 'cuiyaning'
release = '0.1.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'myst_parser',
    ]

templates_path = ['_templates']
exclude_patterns = []

highlight_language = 'python'


# List of modules to be mocked up. This is useful when some external
# dependencies are not met at build time and break the building process.
autodoc_mock_imports = [
    'rdkit',
    'unicore',
    'torch',
    'sklearn',
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Autodoc configuration ---------------------------------------------------

autoclass_content = 'class'

# 显式地设置成员的顺序，确保构造函数的参数首先显示
autodoc_member_order = 'bysource'

# 设置构造函数的默认选项，包括显示参数

autodoc_default_options = {
    'members': True,
    'special-members': '__init__',
    #'undoc-members': False,
    'private-members': True,
    #'show-inheritance': False,
}
