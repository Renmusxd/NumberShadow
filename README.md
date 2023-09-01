# Number Conserving Shadows

Uses all-pairs reconstruction [see notebook](./jupyter/SimulationConsInverse.ipynb).

Some (not all) notebooks use `qutip` (from `pip install qutip`) to generate clifford gates, as well as the usual `jupyter`, `scipy`, `numpy`, `matplotlib`.

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
