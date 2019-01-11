import datetime
import numpy as np
import os.path as osp
from time import sleep
import matplotlib.pyplot as plt
from collections import defaultdict

from dicod.dicod import DICOD, ALGO_GS
from dicod.utils import TimingLogs
from utils.rand_problem import fun_rand_problem


def scaling_n_jobs(T=300, max_jobs=75, n_rep=10, save_dir=None, max_iter=5e6,
                   timeout=7200, hostfile=None, run='all', lgg=False,
                   use_seg=False, algorithm=ALGO_GS, debug=0, seed=None):
    '''Run DICOD algorithm for a certain problem with different value
    for n_jobs and store the runtime in csv files if given a save_dir.

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
    max_iter: int, optional (default: 5e6)
        maximal number of iteration run by DICOD
    timeout: int, optional (default: 7200)
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
    algorithm: enum, optional (default: ALGO_GS)
        Algorithm used to select the update for the coordinate descent. It
        should be either ALGO_GS (greedy selection) or ALGO_RANDOM (random
        selection).
    debug: int, optional (default:0)
        The greater it is, the more verbose the algorithm
    seed: int, optional (default:None)
        seed the rng of numpy to obtain fixed set of problems

    '''
    common_args = dict(logging=lgg, log_rate='log1.6', max_iter=max_iter,
                       timeout=timeout, debug=debug, tol=5e-2, patience=1000,
                       hostfile=hostfile, algorithm=algorithm)

    # Do not use seg with ALGO_RANDOM
    assert not use_seg or algorithm == ALGO_GS

    S = 150
    K = 10
    d = 7
    lmbd = 0.1
    noise_level = 1

    if save_dir is not None and not osp.exists(save_dir):
        import os
        os.mkdir(save_dir)
    elif save_dir is None:
        save_dir = "."

    rng = np.random.RandomState(seed)
    suffix = "_random" if algorithm else "_seg" if use_seg else ""
    file_name = 'runtimes_n_jobs_{}{}.csv'.format(T, suffix)
    file_name = osp.join(save_dir, file_name)

    for j in range(n_rep):
        seed_pb = rng.randint(4294967295)
        pb = fun_rand_problem(T, S, K, d, lmbd, noise_level, seed=seed_pb)

        dicod = DICOD(n_jobs=2, **common_args)

        runtimes = []
        n_jobs = np.logspace(0, np.log2(75), 10, base=2)
        n_jobs = [int(round(nj)) for nj in n_jobs if nj <= max_jobs]
        n_jobs = np.unique(n_jobs)
        n_jobs = n_jobs[::-1]
        for nj in n_jobs:
            code_run = "{}:{}".format(nj, j)
            if (run != 'all' and str(nj) not in run and code_run not in run):
                continue
            dicod.reset()
            pb.reset()
            dicod.n_jobs = nj
            dicod.use_seg = T // nj if use_seg else 1

            dicod.fit(pb)
            timings = TimingLogs(time=dicod.time, runtime=dicod.runtime,
                                 t_init=dicod.t_int)
            runtimes += [[timings]]
            import time
            time.sleep(1)
            if save_dir is not None:
                with open(file_name, 'a') as f:
                    f.write('Pb{},{},{},{}\n'.format(
                        j, nj, timings[0], timings[1]))
            print('=' * 79)
            print('[{}] PB{}: End process with {} jobs  in {:.2f}s'
                  ''.format(datetime.datetime.now().strftime("%I:%M"),
                            j, nj, timings[0]))
            print('\n' + '=' * 79)
            sleep(.5)

    min_njobs = 0
    fig, axs = plt.subplots(1, 1, sharex=True, num="scaling")
    with open(file_name) as f:
        lines = f.readlines()
    arr = defaultdict(lambda: [])
    for l in lines:
        r = list(map(float, l.split(',')[1:]))
        arr[r[0]] += [r]
    axk = axs
    l, L = 1e6, 0
    for k, v in arr.items():
        if k > min_njobs:
            V = np.mean(v, axis=0)[1]
            axk.scatter(k, V, color="b")
            l, L = min(l, V), max(L, V)
    axk.set_xscale('log')
    axk.set_yscale('log')
    n_jobs = np.array([k for k in arr.keys() if k > min_njobs]).astype(int)
    n_jobs.sort()
    m, M = n_jobs.min(), n_jobs.max()
    t = np.logspace(np.log2(m), np.log2(2 * M), 200, base=2)
    R0 = np.mean(arr[m], axis=0)[1]

    axk.plot(t, R0 * m / t, 'k--')
    axk.plot(t, R0 * (m / t)**2, 'r--')
    scaling = R0 / (t * t * np.maximum(1 - 2 * (t / T)**2 * (
        1 + 2 * (t / T)**2)**(t / 2 - 1), 1e-5))
    break_p = np.where((scaling[2:] > scaling[1:-1]) &
                       (scaling[:-2] > scaling[1:-1]))[0] + 1

    axk.plot(t, scaling, "g-.", label="theoretical speedup")
    axk.vlines(t[break_p], .1, 100000, "g", linestyle="-", linewidth=2)
    axk.set_xlim((m * .7, 1.7 * M))
    axk.set_ylim((.5 * l, 1.7 * L))
    axk.set_title("$T={}$".format(T), fontsize="x-large")
    # if i == 0:
    axk.legend(fontsize="large")
    axk.set_ylim((.2 * l, 1.7 * L))
    tt = 8
    axk.text(tt, .4 * R0 * (m / tt)**2, "quadratic", rotation=-22)
    axk.text(tt, R0 * m / tt, "linear", rotation=-14, bbox=dict(
        facecolor="white", edgecolor="white"))

    axk.text(.9 * t[break_p], .7 * R0 * m / tt, "$M^*$", rotation=0,
             bbox=dict(facecolor="w", edgecolor="w"))
    axk.minorticks_off()

    axk.set_xticks(n_jobs)
    axk.set_xticklabels(n_jobs)
    axk.set_xlabel("# cores $M$", fontsize="x-large")
    axk.set_ylabel("Runtime (s)", fontsize="x-large")
    axk.set_xticks([])
    axk.set_xticklabels([], [])
    axk.set_yticks([])
    axk.set_yticklabels([], [])
    plt.subplots_adjust(left=.1, right=.99, top=.95, bottom=.1)
    plt.show()
    input()
