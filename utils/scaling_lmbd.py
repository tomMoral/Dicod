import datetime
import threading
import numpy as np
import os.path as osp
from time import sleep
from joblib import Parallel, delayed, Memory

from cdl.dicod import DICOD, ALGO_GS
from utils.rand_problem import fun_rand_problem


mem = Memory(cachedir=".", verbose=0)


class DummyCtx():
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def run_one(args_pb, lmbds, common_args, n_jobs, n_seg, fname, file_lock,
            dcp=None):

    n_pb, args_pb, seed_pb = args_pb
    pb = fun_rand_problem(*args_pb, seed=seed_pb)
    pb.lmbd = lmbd_max = pb.get_lmbd_max()

    if dcp is None:
        from cdl.dicod import DICOD
        dcp = DICOD(None, n_jobs=n_jobs, use_seg=n_seg,
                    **common_args)
    # Do not count the initialization cost of the MPI pool of workers
    dcp.fit(pb)

    runtimes = []
    for lmbd in lmbds:
        pb.lmbd = lmbd_max * lmbd
        pb.reset()

        dcp.fit(pb)
        import time
        time.sleep(1)
        out_str = 'Pb{},{},{},{},{}\n'.format(n_pb, lmbd, dcp.runtime, dcp.t,
                                              lmbd_max)
        with file_lock:
            with open(fname, 'a') as f:
                f.write(out_str)

        runtimes += [out_str]
        print('=' * 79)
        print('[{}] PB{}: End process with lmbd={}({}) in {:.2f}s'
              ''.format(datetime.datetime.now().strftime("%Ih%M"),
                        n_pb, lmbd, lmbd * lmbd_max, dcp.runtime))
        print('\n' + '=' * 79)
        sleep(.5)
    return runtimes


def scaling_lmbd(T=300, n_jobs=75, n_rep=10, save_dir=None, i_max=1e8,
                 t_max=7200, hostfile=None, run='all', lgg=False,
                 use_seg=False, graphical_cost=None, algorithm=ALGO_GS,
                 debug=0, seed=None):
    '''Run DICOD algorithm for a certain problem with different value
    for lmbd and store the runtime in csv files if given a save_dir.

    Parameters
    ----------
    T: int, optional (default: 300)
        Size of the generated problems
    max_jobs: int, optional (default: 75)
        The algorithm will be run on problems with a number
        of cores varying from 5 to max_jobs in a log2 fashion
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
    run: list or str, optional (default: 'all')
        if all, run all the possible runs. Else, it should be a list composed
        of int for njobs to run or of str {n_jobs: n_rep} for specific cases.
    lgg: bool, optional (default: False)
        If set to true, enable the logging of the iteration cost
        during the run. It might slow down a bit the execution
        time and the collection of the results
    graphical_cost: dict, optional (default: None)
        Setup option to enable a graphical logging of the cost
        function.
    algorithm: enum, optional (default: ALGO_GS)
        Algorithm used to select the update for the coordinate descent. It
        should be either ALGO_GS (greedy selection) or ALGO_RANDOM (random
        selection).
    debug: int, optional (default:0)
        The greater it is, the more verbose the algorithm
    seed: int, optional (default:None)
        seed the rng of numpy to obtain fixed set of problems

    '''

    # Do not use seg with ALGO_RANDOM
    # assert not use_seg or algorithm == ALGO_GS

    # Make sure the output directory exists
    if save_dir is not None and not osp.exists(save_dir):
        import os
        os.mkdir(save_dir)

    # Set the problem arguments
    S = 150
    K = 10
    d = 7
    lmbd = 0.1
    noise_level = 1
    common_args = dict(logging=lgg, log_rate='log1.6', i_max=i_max,
                       t_max=t_max, graphical_cost=graphical_cost,
                       debug=debug, tol=1e-2, hostfile=hostfile,
                       algorithm=algorithm, patience=1000)

    # Set the solver arguments
    if use_seg:
        n_seg = T // 2
        suffix = "_seg"
        n_jobs = 1
        outer_jobs = 10

        file_lock = threading.Lock()
        dcp = None
    else:
        n_seg = 1
        suffix = ""
        outer_jobs = 1
        file_lock = DummyCtx()
        dcp = DICOD(None, n_jobs=n_jobs, use_seg=n_seg,
                    **common_args)

    rng = np.random.RandomState(seed)
    fname = 'runtimes_lmbd_{}{}.csv'.format(T, suffix)
    fname = osp.join(save_dir, fname)
    print(fname)

    lmbds = np.logspace(-6, np.log10(.8), 15)
    lmbds = lmbds[::-1]

    list_args_pb = []
    for j in range(n_rep):
        seed_pb = rng.randint(4294967295)
        list_args_pb += [(j, (T, S, K, d, lmbd, noise_level), seed_pb)]

    cached_run_one = mem.cache(run_one, ignore=['file_lock', 'dcp'])
    runtimes = Parallel(n_jobs=outer_jobs, backend="threading")(
        delayed(cached_run_one)(args_pb, lmbds, common_args, n_jobs, n_seg,
                                fname, file_lock, dcp=dcp)
        for args_pb in list_args_pb)

    print(runtimes)
