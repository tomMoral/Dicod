from utils.rand_problem import fun_rand_problem
import os.path as osp
from time import sleep

from cdl.dicod import DICOD
from cdl.fista import FISTA
from cdl.feature_sign_search import FSS

from toolbox.logger import Logger
log = Logger('Root')


def compare_met(T=80, save_dir=None, i_max=5e6, t_max=7200,
                n_jobs=3, hostfile=None, graphical_cost=None,
                display=True, debug=0):
    '''Run DICOD algorithm for a certain problem with different value
    for n_jobs and store the runtime in csv files if given a save_dir.

    Parameters
    ----------
    T: int, optional (default: 300)
        Size of the generated problems
    save_dir: str, optional (default: None)
        If not None, all the runtimes will be saved in csv files
        contained in the given directory. The directory must exist.
        This will create a file for each problem size T and save
        the Pb number, the number of core and the runtime computed
        in two different ways.
    i_max: int, optional (default: 5e6)
        maximal number of iteration run by the algorithms
    t_max: int, optional (default: 7200)
        maximal running time for each algorithm. The default timeout
        is 2 hours
    n_jobs: int , optional (default: 3)
        Maximal number of jobs used for distributed algorithms
    hostfile: str, optional (default: None)
        hostfile used to launch MPI jobs
    graphical_cost: dict, optional (default: None)
        Setup option to enable a grpahical logging of the cost
        function.
    debug: int, optional (default:0)
        The greater it is, the more verbose the algorithm
    '''
    common_args = dict(logging=True, log_rate='log1.6', i_max=i_max,
                       t_max=t_max, graphical_cost=graphical_cost,
                       debug=debug, tol=5e-2)
    S = 200
    K = 10
    d = 7
    lmbd = 0.1
    noise_level = 1
    pb = fun_rand_problem(T, S, K, d, lmbd, noise_level)

    from collections import OrderedDict
    algos = OrderedDict()
    algos['CD'] = (DICOD(
        pb, n_jobs=1, hostfile=hostfile, **common_args), 'rd-')
    algos['Fista'] = (FISTA(pb, fixe=True, **common_args), 'y*-')
    algos['FSS'] = (FSS(pb, n_zero_coef=40, **common_args), 'go-')
    algos['SeqDICOD$_{{{}}}$'.format(n_jobs)] = (DICOD(
        pb, n_jobs=1, use_seg=n_jobs, hostfile=hostfile, **common_args),
        'c^-'
    )
    algos['DICOD$_{{{}}}$'.format(n_jobs//2)] = (DICOD(
        pb, n_jobs=n_jobs//2, hostfile=hostfile, **common_args),
        'ms-'
    )
    algos['DICOD$_{{{}}}$'.format(n_jobs)] = (DICOD(
        pb, n_jobs=n_jobs, hostfile=hostfile, **common_args),
        'bH-'
    )

    for name, (algo, _) in algos.items():
        pb.reset()
        print('\n\n')
        log.info(name)
        algo.fit(pb)
        log.process_queue()
        sleep(2)

    curves = {}

    if not display:
        import matplotlib as mpl
        mpl.use('Agg')
    import matplotlib.pyplot as plt
    tlim = [1e-1, 1e-1]
    base_cost = pb.cost(pb.x0)
    clim = [base_cost, base_cost*1.1]
    ilim = [0, 0]
    for (name, (algo, styl)) in algos.items():
        i = int(str(algo).split('_')[-1])
        cost = log.output.log_objects['cost{}'.format(i)]
        t = log.output.log_objects['cost{}_t'.format(i)]
        t = [4e-2] + t
        cost = [cost[0]] + cost

        it = log.output.log_objects['cost{}_i'.format(i)]

        curves[name] = (it, t, cost)

        clim[0] = min(clim[0], cost[-1]*.9)
        tlim[0] = min(tlim[0], t[1]*.9)
        tlim[1] = max(tlim[1], t[-1]*1.1)
        ilim[1] = max(ilim[1], it[-1]*1.1)

        plt.figure('Time')
        plt.loglog(t, cost, styl, label=name, linewidth=2,
                   markersize=9)
        plt.figure('Iteration')
        plt.loglog(it, cost[1:], styl, label=name, linewidth=2,
                   markersize=9)

    import pickle
    if save_dir is not None:
        with open(osp.join(save_dir, 'cost_curves_T{}.pkl'.format(T)),
                  'wb') as f:
            pickle.dump(curves, f)

    # Format the figures
    plt.figure('Iteration')
    plt.hlines([clim[0]/.9], ilim[0], ilim[1],
               linestyles='--', colors='k')
    plt.legend(fontsize=16)
    plt.xlabel('# iteration', fontsize=18)
    plt.ylabel('Cost', fontsize=18)
    plt.xlim(ilim)
    plt.ylim(clim)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.subplots_adjust(top=.97, left=.1, bottom=.1, right=.98)

    plt.figure('Time')
    plt.hlines([clim[0]/.9], tlim[0], tlim[1],
               linestyles='--', colors='k')
    plt.legend(fontsize=16)
    plt.xlabel('Time (s)', fontsize=18)
    plt.ylabel('Cost', fontsize=18)
    plt.xlim(tlim)
    plt.ylim(clim)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.subplots_adjust(top=.97, left=.1, bottom=.1, right=.98)

    plt.show()
    if display:
        input()

    if save_dir is not None:
        plt.figure('Time')
        plt.savefig(osp.join(save_dir, 'cost_time_T{}.pdf'.format(T)),
                    dpi=150)
        plt.figure('Iteration')
        plt.savefig(osp.join(save_dir, 'cost_iter_T{}.pdf'.format(T)),
                    dpi=150)
