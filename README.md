This package is still under development. If you have any trouble running this code, please contact <thomas.moreau.2010@gmail.com>


## DICOD

Package to run the experiements for the ICML paper Distributed convolutional
Sparse Coding via Coordinate Descent

#### Requirements

All the tests were done with python3.4.
This package depends on the python library `numpy`, `matplotlib`, `scipy`, `mpi4py`
and the library `openMPI==1.6.5` and `fftw3`.
They can be installed with

```bash
sudo apt install libopenmpi-dev fftw-dev
pip install numpy matplotlib scipy mpi4py
```

To build the package, use the utility script `./build`.

#### Usage

Figure 2 can be generated using
```bash
python main_dcp.py --met -K 25 -T 600 --tmax 7200 -d 100 --njobs 60 --hostfile hostfile --save figure2
```

where hostfile is the configuration for the spawning of MPI processes

