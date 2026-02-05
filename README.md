# bcc-nqs
Code for finding ground states of Heisenberg models on the body-centered cubic lattice using neural quantum states.
This repository is built on top of [NetKet](https://github.com/netket/netket).

# Installation
The repository uses [uv](https://docs.astral.sh/uv/) to manage dependencies.
Download the repository and use the command `uv run` to run any `.py` scripts in the correct environment. `uv` will take care of installing the required dependencies.

# Contents
A good place to start is the `examples` folder, which contains:
- `nets`: examples of how to use the ViT with netket to perform ground state optimizations on the body-centered cubic lattice.
- `vmc`: examples of how to use the optimization protocols and expectation value computation scripts defined in the `vmc` package.

Packages in the repository are contained in `src`:
- `nets`: The ViT NQS (for use on 1, 2 and 3-dimensional lattices) and some utility functions.
- `vmc`: Variational Monte Carlo protocols for performing optimization and calculating expectation values from the NQS, with the use of different symmetrization stages.
- `nk_extensions`: Extra functionality for dealing with space groups, definitions of the BCC lattice and a few other utilities which extend on NetKet.
- `data`: Defines the `SpinCorr` class for working with spin-spin correlation data, including computing static structure factors.

A suite of tests are contained in `test`.
