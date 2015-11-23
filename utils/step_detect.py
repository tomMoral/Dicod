import numpy as np
import matplotlib.pyplot as plt
import IPython

from utils.rand_problem import fun_step_problem, _whiten_sig
from cdl.dicod import DICOD

from toolbox.logger import Logger
log = Logger('Root')


def step_detect(save_dir=None, i_max=5e6, t_max=7200, n_jobs=2,
                hostfile=None, n_iter=100,
                graphical_cost=None, debug=0):
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
        maximal number of iteration run by DICOD
    t_max: int, optional (default: 7200)
        maximal running time for DICOD. The default timeout
        is 2 hours
    graphical_cost: dict, optional (default: None)
        Setup option to enable a grpahical logging of the cost
        function.
    debug: int, optional (default:0)
        The greater it is, the more verbose the algorithm

    '''
    try:
        common_args = dict(logging=True, log_rate='log1.6', i_max=i_max,
                           t_max=t_max, debug=debug, tol=5e-2)

        print('construct problem')
        pbs, D, D_labels = fun_step_problem(K=30, N=300, lmbd=1)
        N = len(pbs)
        print('End\n')
        lmbd = .2

        dcp = DICOD(pbs[0][0], n_jobs=n_jobs, hostfile=hostfile,
                    positive=True, use_seg=1, **common_args)

        grad_D = [np.zeros(D.shape) for _ in range(N)]
        grad_nz = set()
        cost = np.zeros(N)

        for j in range(n_iter):
            # Stochastic choice of a ponit
            i0 = np.random.randint(N)
            pb, ex, foot = pbs[i0]
            pb.D = D

            # Sparse coding
            pb.reset()
            dcp.fit(pb)

            # Update cost and D gradient
            new = cost[i0] == 0
            cost[i0] = pb.cost()
            grad_D[i0] = pb.grad_D(pb.pt)
            grad_nz.add(i0)

            # Logging
            N_see = len(grad_nz)
            cost_i = np.sum(cost, axis=0)/N_see
            print('-'*79 + '\n End epoch {} with cost {:e}'.format(j, cost_i))
            print('New : ', new)
            print('-'*79)
            if graphical_cost is not None:
                log.graphical_cost(cost=np.sum(cost, axis=0)/N_see)

            # reg = np.zeros(D.shape)
            # reg[:, :, :-2] += D[:, :, 2:]
            # reg[:, :, :] -= D[:, :, :]
            # reg[:, :, :] /= np.sqrt(1e-2+D[:, :, :]*D[:, :, :])

            # Update dictionary
            grad = np.sum(grad_D, axis=0)/N_see
            D -= lmbd*grad

            if j % 10 == 0 and j > 0:
                #IPython.embed()
                lmbd *= .5
    finally:
        IPython.embed()
