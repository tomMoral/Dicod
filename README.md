This package is still under development. If you have any trouble running this code, please contact <thomas.moreau.2010@gmail.com>


## DICOD  [![Build Status](https://travis-ci.org/tomMoral/Dicod.svg?branch=master)](https://travis-ci.org/tomMoral/Dicod)

Package to run the experiments for the ICML paper [DICOD: Distributed Convolutional Coordinate Descent for Convolutional Sparse Coding](http://proceedings.mlr.press/v80/moreau18a.html), ICML 2018, T. Moreau, L. Oudre, N. Vayatis.

#### Requirements

All the tests were done with python3.4.
This package depends on the python library `numpy`, `matplotlib`, `scipy`, `mpi4py`, `joblib`
and the libraries `openMPI` and `fftw3`.
They can be installed with

```bash
sudo apt install libopenmpi-dev fftw-dev
pip install numpy matplotlib scipy mpi4py joblib
```

To install the package, first build it with the utility script `./build` and then run `pip install -e .`

#### Usage

Figure 2 can be generated using
```bash
$ python main_dicod.py --met -K 25 -T 600 --timeout 7200 -d 10 --njobs 60 --hostfile hostfile --exp results
```

where hostfile is the configuration for the spawning of MPI processes.

```
host1 slots=32
host2 slots=8
...
```

Then the figures can be plotted using
```bash
$ python plot_dicod.py --met --dir save_exp/results
```

