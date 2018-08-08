import logging
import os.path as osp
from time import sleep

from dicod.dicod import DICOD
from dicod.fista import FISTA
from dicod.feature_sign_search import FSS
from dicod.fcsc import FCSC
from utils.rand_problem import fun_rand_problem


log = logging.getLogger('dicod')



def compare_met(T=80, K=10, save_dir=None, max_iter=5e6, timeout=7200,
                n_jobs=3, hostfile=None, graphical_cost=None,
                display=True, debug=0):
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
    graphical_cost: dict, optional (default: None)
        Setup option to enable a graphical logging of the cost
        function.
    debug: int, optional (default:0)
        The greater it is, the more verbose the algorithm
    '''
    common_args = dict(logging=True, log_rate='log1.6', max_iter=max_iter,
                       timeout=timeout, graphical_cost=graphical_cost,
                       debug=debug, tol=1e-4)
    S = 200
    d = 7
    lmbd = 1
    noise_level = .1
    pb = fun_rand_problem(T, S, K, d, lmbd, noise_level, seed=42)

    if save_dir is not None:
        save_dir = osp.join("save_exp", save_dir)
        assert osp.exists(save_dir)

    from collections import OrderedDict
    algos = OrderedDict()
    # algos['DICOD$_{{{}}}$'.format(n_jobs // 2)] = (DICOD(
    #     pb, n_jobs=n_jobs // 2, hostfile=hostfile, **common_args),
    #     'ms-'
    # )
    # algos['DICOD$_{{{}}}$'.format(n_jobs)] = (DICOD(
    #     pb, n_jobs=n_jobs, hostfile=hostfile, **common_args),
    #     'bH-'
    # )
    # algos['CD'] = (DICOD(
    #     pb, n_jobs=1, hostfile=hostfile, **common_args), 'rd-')
    # algos['RCD'] = (DICOD(
    #     pb, algorithm=1, n_jobs=1, hostfile=hostfile, patience=5e5,
    #     **common_args), 'cd-')
    algos['Fista'] = (FISTA(pb, fixe=True, **common_args), 'y*-')
    # # algos['FSS'] = (FSS(pb, n_zero_coef=40, **common_args), 'go-')
    # algos['FCSC'] = (FCSC(pb, tau=1.01, **common_args), 'k.-')
    # algos['SeqDICOD$_{{{}}}$'.format(n_jobs)] = (DICOD(
    #     pb, n_jobs=1, use_seg=n_jobs, hostfile=hostfile, **common_args),
    #     'c^-'
    # )
    # algos['SeqDICOD$_{{{}}}$'.format(n_jobs * 10)] = (DICOD(
    #     pb, n_jobs=1, use_seg=n_jobs * 10, hostfile=hostfile, **common_args),
    #     'c^-'
    # )

    curves = {}

    if save_dir is not None:
        import pickle
        try:
            fname = osp.join(save_dir, 'cost_curves_T{}_K{}.pkl'.format(T, K))
            with open(fname, 'rb') as f:
                curves = pickle.load(f)
        except FileNotFoundError:
            pass
    for name, (algo, _) in algos.items():
        pb.reset()
        print('\n\n')
        log.info(name)
        algo.fit(pb)
        log.process_queue()
        sleep(1)
        i = int(str(algo).split('_')[-1])
        cost = log.output.log_objects['cost{}'.format(i)]
        t = log.output.log_objects['cost{}_t'.format(i)]
        t = [4e-2] + t
        cost = [cost[0]] + cost

        it = log.output.log_objects['cost{}_i'.format(i)]

        curves[name] = (it, t, cost)

        if save_dir is not None:
            import pickle
            # Try loading previous values
            try:
                fname = osp.join(save_dir, 'cost_curves_T{}_K{}.pkl'.format(T, K))
                with open(fname, 'rb') as f:
                    o_curves = pickle.load(f)
            except FileNotFoundError:
                o_curves = {}
            o_curves[name] = curves[name]
            curves = o_curves
            with open(fname, 'wb') as f:
                pickle.dump(o_curves, f)

    import IPython
    IPython.embed()
