## DICOD

Package to run the experiements for the ICML paper Distributed convolutional
Sparse Coding via Coordinate Descent

#### Requirements

All the tests were done with python3.4.
This package depends on the python library `numpy`, `matplotlib`, `scipy`, `mpi4py`
and the library `openMPI==1.6.5`.
They can be installed with

```bash
pip install nnumpy matplotlib scipy mpi4py
sudo apt-get install openmpi
```

To build the package, use the utility script `./build`.

#### Usage

Figure 2 can be generated using
```bash
python main_dcp.py --met -T 250 -K 10 --njobs 30 --tmax 500
python main_dcp.py --jobs -T 250 --njobs 75 --tmax 500 --hostfile hostfile
```

where hostfile is the configuration for the spawning of MPI processes

