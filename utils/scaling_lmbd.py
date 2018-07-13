import os
import errno
import datetime
import itertools
import threading
import numpy as np
from time import sleep
from joblib import Parallel, delayed, Memory

from dicod.dicod import DICOD, ALGO_GS
from utils.rand_problem import fun_rand_problem


mem = Memory(cachedir=".", verbose=0)


class DummyCtx():
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class FileLock():
    def __init__(self, fname):
        self.file_lock_name = fname + '.lock'
        self.fd = None
        self.is_lock = False

    def __getstate__(self):
        return (self.file_lock_name,)

    def __setstate__(self, state):
        self.file_lock_name, = state
        self.fd = None
        self.is_lock = False

    def __enter__(self):
        while not self.is_lock:
            try:
                self.fd = os.open(self.file_lock_name, os.O_CREAT | os.O_EXCL)
                self.is_lock = True
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
                sleep(.001)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.is_lock:
            raise RuntimeError("Releasing an unlocked file??")

        os.close(self.fd)
        self.is_lock = False
        self.fd = None
        os.unlink(self.file_lock_name)


def run_one(args_pb, lmbd, optimizer, optimizer_kwargs, fname, file_lock):

    n_pb, args_pb, seed_pb = args_pb
    pb = fun_rand_problem(*args_pb, seed=seed_pb)

    if isinstance(optimizer, str):
        method = optimizer
        if optimizer == "lgcd":
            from dicod.dicod import DICOD
            optimizer = DICOD(None, **optimizer_kwargs)
        elif optimizer == "fista":
            from dicod.fista import FISTA
            optimizer = FISTA(None, **optimizer_kwargs)
    elif getattr(optimizer, "fit", None) is None:
        raise ValueError("`optimizer` parameter should be a string or an "
                         "optimizer object.")
    else:
        method = "dicod"

    # Do not count the initialization cost of the MPI pool of workers
    pb.lmbd = lmbd_max = pb.get_lmbd_max()
    optimizer.fit(pb)

    pb.lmbd = lmbd_max * lmbd
    pb.reset()

    optimizer.fit(pb)
    import time
    time.sleep(1)
    sparsity = len(pb.pt.nonzero()[0]) / pb.pt.size
    out_str = 'Pb{},{},{},{},{},{},{}\n'.format(n_pb, lmbd, optimizer.runtime,
                                                optimizer.t, lmbd_max, method,
                                                sparsity)
    with file_lock:
        with open(fname, 'a') as f:
            f.write(out_str)

    print('=' * 79)
    print('[{}] PB{}: End process with lmbd={}({}) in {:.2f}s'
          ''.format(datetime.datetime.now().strftime("%Ih%M"),
                    n_pb, lmbd, lmbd * lmbd_max, optimizer.runtime))
    print('\n' + '=' * 79)
    sleep(.5)
    return out_str


def scaling_lmbd(T=300, n_jobs=75, n_rep=10, save_dir=None, i_max=1e8,
                 t_max=7200, hostfile=None, lgg=False, optimizer="dicod",
                 debug=0, seed=None):
    '''Run DICOD algorithm for a certain problem with different value
    for lmbd and store the runtime in csv files if given a save_dir.

    Parameters
    ----------
    T: int, optional (default: 300)
        Size of the generated problems
    n_rep: int, optional (default: 10)
        Number of different problem solved for all the different
        number of cores.
    save_dir: str, optional (default: None)
        If not None, all the runtimes will be saved in csv files
        contained in the given directory. The directory must exist.
        This will create a file for each problem size T and save
        the Pb number, the number of core and the runtime computed
        in two different ways.
    i_max: int, optional (default: 5e6)
        maximal number of iteration run by DICOD
    t_max: int, optional (default: 7200)
        maximal running time for DICOD. The default timeout
        is 2 hours
    hostfile: str, optional (default: None)
        hostfile for the openMPI API to connect to the other
        running server to spawn the processes over different
        nodes
    lgg: bool, optional (default: False)
        If set to true, enable the logging of the iteration cost
        during the run. It might slow down a bit the execution
        time and the collection of the results
    optimizer: str, optional (default: 'dicod')
        Algorithm used to compute the CSC solution. Should be in {'dicod',
        'fista'}.
    debug: int, optional (default:0)
        The greater it is, the more verbose the algorithm
    seed: int, optional (default:None)
        seed the rng of numpy to obtain fixed set of problems
    '''

    # Make sure the output directory exists
    if save_dir is not None and not os.path.exists(save_dir):
        os.mkdir(save_dir)

    fname = 'runtimes_lmbd_{}_{}.csv'.format(T, optimizer)
    fname = os.path.join(save_dir, fname)
    print(fname)

    # Set the problem arguments
    S = 150
    K = 10
    d = 7
    noise_level = 1
    optimizer_kwargs = dict(logging=lgg, log_rate='log1.6', i_max=i_max,
                            t_max=t_max, debug=debug, tol=1e-2)

    # Set the solver arguments
    backend = None
    outer_jobs = n_jobs
    if optimizer == "lgcd":
        optimizer_kwargs['use_seg'] = T // 2
        optimizer_kwargs['algorithm'] = ALGO_GS
        optimizer_kwargs['hostfile'] = hostfile

        backend = "threading"
        file_lock = threading.Lock()

    elif optimizer == "dicod":
        optimizer_kwargs['hostfile'] = hostfile
        outer_jobs = 1
        file_lock = DummyCtx()
        optimizer = DICOD(None, n_jobs=n_jobs, use_seg=1,
                          **optimizer_kwargs)

    elif optimizer == "fista":
        optimizer_kwargs['fixe'] = True
        file_lock = FileLock(fname)
    else:
        raise RuntimeError("Unknown optimizer {}".format(optimizer))

    rng = np.random.RandomState(seed)

    lmbds = np.logspace(-6, np.log10(.8), 15)
    lmbds = lmbds[::-1]

    list_args_pb = []
    for j in range(n_rep):
        seed_pb = rng.randint(4294967295)
        list_args_pb += [(j, (T, S, K, d, 1000, noise_level), seed_pb)]

    grid_args = itertools.product(list_args_pb, lmbds)

    cached_run_one = mem.cache(run_one, ignore=['file_lock'])
    runtimes = Parallel(n_jobs=outer_jobs, backend=backend)(
        delayed(cached_run_one)(args_pb, lmbd, optimizer, optimizer_kwargs,
                                fname, file_lock)
        for args_pb, lmbd in grid_args)

    print(runtimes)
