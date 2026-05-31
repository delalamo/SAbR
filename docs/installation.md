# Installation

SAbR requires Python 3.11 or newer.

Install the published package:

```bash
pip install sabr-kit
```

Install from source when developing:

```bash
git clone --recursive https://github.com/delalamo/SAbR.git
cd SAbR
pip install -e .[test,docs]
```

On Apple Silicon systems running an x86 Python, JAX may fail to import because
the installed `jaxlib` wheel targets unsupported CPU instructions. Use a native
ARM Python environment or set `JAX_PLATFORMS=cpu` when running SAbR.

Docker users can run the maintained image:

```bash
docker run --rm ghcr.io/delalamo/sabr:latest \
  -i input.pdb -c H -o output.pdb
```
