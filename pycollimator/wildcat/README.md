# Lynx

Proof-of-concept simulation engine - basically a port of [Drake](https://drake.mit.edu/) to [JAX](https://jax.readthedocs.io/en/latest/), with minor modifications and a minimal interface to the Collimator app.

This repo is mainly intended as a "code-based" explanation of the [Notion proposal](https://www.notion.so/Simulation-engine-design-doc-f311d3b3894246768dd6257f90076567?d=1ced569a3b014b91a023d5639a726108#bf068f457d864d0582bced9f9bb7cad0) for a standalone simulation engine.  The source code and proposal are both works in progress, so comments/criticisms are welcome.

See the notebooks in `examples/` for some examples basic usage, including parsing JSON app output in `examples/app/`.  At the moment, these don't extend beyond current capabilities of CMLC.

## Installation

### Conda installation

> These are Jared's original instructions

Conda may not be the best way to do this, but it works for now.  Install dependencies with

```bash
conda env create -f environment.yml
conda activate lynx
pip install -e .
```

You may also need to install diffrax:
```bash
pip install diffrax
```

Test the install with
```bash
pytest test/
```

### Alternatively: venv + pip installation

@jp-andre tested with a simple `venv` and pip install:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
pip install pytest
pip install pytest-xdist
pytest -nauto test/
```
