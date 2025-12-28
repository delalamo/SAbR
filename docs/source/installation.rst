Installation
============

Requirements
------------

- Python 3.11 or higher
- JAX (CPU or GPU)

Installing from PyPI
--------------------

The simplest way to install SAbR is via pip::

   pip install sabr-kit

This will install the latest release from PyPI along with all dependencies.

Installing from Source
----------------------

For the latest development version, clone the repository::

   git clone --recursive https://github.com/delalamo/SAbR.git
   cd SAbR/
   pip install -e .

The ``--recursive`` flag is important as it clones the required submodules
(SoftAlign and ANARCI).

Docker
------

SAbR is also available as a Docker container::

   docker run --rm ghcr.io/delalamo/sabr:latest -i input.pdb -o output.pdb -c A

To use local files, mount your directory::

   docker run --rm -v $(pwd):/data ghcr.io/delalamo/sabr:latest \
       -i /data/input.pdb -o /data/output.pdb -c A

Apple Silicon (M1/M2/M3)
------------------------

When running on Apple Silicon Macs, set the JAX platform to CPU::

   export JAX_PLATFORMS=cpu
   sabr -i input.pdb -c A -o output.pdb

This is required because JAX's Metal backend may have compatibility issues.

Dependencies
------------

SAbR depends on the following packages:

- **biopython** (>=1.85): PDB/mmCIF file parsing
- **click** (>=8.1): Command-line interface
- **jax** (>=0.4.20): Neural network computations
- **jaxlib** (>=0.4.20): JAX backend
- **numpy** (>=2.4): Numerical operations
- **pandas** (>=2.2): Data manipulation
- **scipy** (>=1.12): Scientific computing
- **dm-haiku** (>=0.0.12): Neural network library

Development Dependencies
------------------------

For development and testing, install the optional test dependencies::

   pip install sabr-kit[test]

Or if installing from source::

   pip install -e ".[test]"

This includes:

- **pytest** (>=7.0): Testing framework
- **pytest-cov** (>=4.1): Coverage reporting

Verifying Installation
----------------------

After installation, verify that SAbR is correctly installed::

   sabr --help

You should see the command-line help message with all available options.
