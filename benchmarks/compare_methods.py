import os
import pickle
import logging

from dicod.dicod import DICOD
from dicod.fista import FISTA
# from dicod.feature_sign_search import FSS
from dicod.fcsc import FCSC
from utils.rand_problem import fun_rand_problem


log = logging.getLogger('dicod')


def compare_met(T=80, K=10, save_dir=None, max_iter=5e6, timeout=7200,
                n_jobs=3, hostfile=None, display=True, debug=0):
    '''Run DICOD algorithm for a certain problem with different value
    for n_jobs and store the runtime in csv files if given a save_dir.

    Parameters
    ----------
    T: int, optional (default: 80)
        Size of the generated problems
    save_dir: str, optional (default: None)
        If not None, all the runtimes will be saved in csv files
        contained in the given directory. The directory must exist.
        This will create a file for each problem size T and save
        the Pb number, the number of core and the runtime computed
        in two different ways.
    max_iter: int, optional (default: 5e6)
        maximal number of iteration run by the algorithms
    timeout: int, optional (default: 7200)
        maximal running time for each algorithm. The default timeout
        is 2 hours
    n_jobs: int , optional (default: 3)
        Maximal number of jobs used for distributed algorithms
    hostfile: str, optional (default: None)
        hostfile used to launch MPI jobs
    debug: int, optional (default:0)
        The greater it is, the more verbose the algorithm
    '''
    common_args = dict(logging=True, log_rate='log1.6', max_iter=max_iter,
                       timeout=timeout, debug=debug, tol=1e-4)
    S = 200
    d = 7
    lmbd = 1
    noise_level = .1
    pb = fun_rand_problem(T, S, K, d, lmbd, noise_level, seed=42)

    if save_dir is not None:
        save_dir = os.path.join("save_exp", save_dir)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    file_name = 'cost_curves_T{}_K{}_njobs{}.pkl'.format(T, K, n_jobs)
    file_name = os.path.join(save_dir, file_name)

    from collections import OrderedDict
    algos = OrderedDict()
    algos['DICOD$_{{{}}}$'.format(n_jobs // 2)] = (DICOD(
        n_jobs=n_jobs // 2, hostfile=hostfile, **common_args),
        'ms-'
    )
    algos['DICOD$_{{{}}}$'.format(n_jobs)] = (DICOD(
        n_jobs=n_jobs, hostfile=hostfile, **common_args),
        'bH-'
    )
    algos['CD'] = (DICOD(
        n_jobs=1, hostfile=hostfile, **common_args), 'rd-')
    algos['RCD'] = (DICOD(
        algorithm=1, n_jobs=1, hostfile=hostfile, patience=5e5,
        **common_args), 'cd-')
    algos['Fista'] = (FISTA(fixe=True, **common_args), 'y*-')
    # algos['FSS'] = (FSS(n_zero_coef=40, **common_args), 'go-')
    algos['FCSC'] = (FCSC(tau=1.01, **common_args), 'k.-')
    algos['LGCD$_{{{}}}$'.format(n_jobs)] = (DICOD(
        n_jobs=1, use_seg=n_jobs, hostfile=hostfile, **common_args),
        'c^-'
    )
    algos['LGCD$_{{{}}}$'.format(n_jobs * 10)] = (DICOD(
        n_jobs=1, use_seg=n_jobs * 10, hostfile=hostfile, **common_args),
        'c^-'
    )

    curves = {}
    if save_dir is not None:
        try:
            with open(file_name, 'rb') as f:
                curves = pickle.load(f)
        except FileNotFoundError:
            pass

    for name, (algo, _) in algos.items():
        pb.reset()
        log.info('='*10 + ' {} '.format(name) + '='*10)
        algo.fit(pb)

        curves[name] = algo.cost_curve

        if save_dir is not None:
            # Try loading previous values
            try:
                with open(file_name, 'rb') as f:
                    o_curves = pickle.load(f)
            except FileNotFoundError:
                o_curves = {}
            o_curves[name] = curves[name]
            curves = o_curves
            with open(file_name, 'wb') as f:
                pickle.dump(o_curves, f)

    import IPython
    IPython.embed()
