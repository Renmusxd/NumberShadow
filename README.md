# Number Conserving Shadows

This repo provides a proof-of-concept implementation of the protocol described in https://arxiv.org/abs/2311.09291

A simple use case is described in [this notebook](./jupyter/SimpleExample.ipynb).

## Installation
1. Install rust on your system: https://www.rust-lang.org/learn/get-started
2. Make a python/conda virtualenv if desired.
3. Prepare your python environment by installing `maturin`, `numpy`, `wheel`, and upgrading `pip`:
    1. `> pip install maturin numpy wheel`
    2. `> pip install --upgrade pip`
4. Clone the repository:
    1. `> git clone git@github.com:Renmusxd/NumberShadow.git`
5. Run `make` in the parent directory
    1. `> make`
6. Install the resulting wheel with pip
    1. `> pip install target/wheels/*`

Some sparse matrix features require scipy, however this is not necessary for MVP.