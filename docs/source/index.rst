SAbR Documentation
==================

**Structure-based Antibody Renumbering**

.. image:: https://img.shields.io/pypi/v/sabr-kit.svg
   :target: https://pypi.org/project/sabr-kit/
   :alt: PyPI version

.. image:: https://img.shields.io/badge/python-3.11+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.11+

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

SAbR (**S**\ tructure-based **A**\ nti\ **b**\ ody **R**\ enumbering) renumbers
antibody PDB files using the 3D coordinates of backbone atoms. It uses neural
network embeddings from `SoftAlign <https://github.com/delalamo/SoftAlign>`_
and antibody numbering schemes from `ANARCI <https://github.com/delalamo/ANARCI>`_
to align structures to consensus embeddings and apply standard numbering schemes.

Key Features
------------

- **Structure-based alignment**: Uses 3D backbone coordinates instead of sequence
- **Multiple numbering schemes**: IMGT, Chothia, Kabat, Martin, Aho, Wolfguy
- **Automatic chain detection**: Distinguishes heavy, kappa, and lambda chains
- **PDB and mmCIF support**: Input and output in both formats
- **Extended insertions**: Support for very long CDR loops (mmCIF only)

.. note::

   This project is currently in development. If you encounter any bugs, please
   report them at https://github.com/delalamo/SAbR/issues

Quick Start
-----------

Installation::

   pip install sabr-kit

Basic usage::

   sabr -i input.pdb -c A -o output.pdb -n imgt

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   usage
   numbering_schemes

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/cli
   api/softaligner
   api/mpnn_embeddings
   api/edit_pdb
   api/aln2hmm
   api/util
   api/constants

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
