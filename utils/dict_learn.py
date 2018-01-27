import numpy as np
from time import time

import IPython

from utils.rand_problem import fun_rand_problems
from cdl.dicod import DICOD

from toolboxTom.logger import Logger
log = Logger('Root')


def dict_learn(i_max=5e6, t_max=7200, n_jobs=2, hostfile=None,
               n_epoch=10, graphical_cost=None, save_dir=None,
               debug=0):
    '''Run DICOD algorithm for a certain problem with different value
    for n_jobs and store the runtime in csv files if given a save_dir.

    Parameters
    ----------
    i_max: int, optional (default: 5e6)
        maximal number of iteration run by DICOD
    t_max: int, optional (default: 7200)
        maximal running time for DICOD. The default timeout
        is 2 hours
    n_jobs: int, optional (default: 2)
        Maximal number of jobs used to compute the convolutional
        sparse coding
    hostfile: str, optional (default: None)
        MPI cluster confifg file, permit to specify multiple machine
        to run the convolutional sparse coding
    n_epoch: int, optional (default: 10)
        number of epoch run by the algorithm
    graphical_cost: dict, optional (default: None)
        Setup option to enable a grpahical logging of the cost
        function
    save_dir: str, optional (default: None)
        If not None, all the runtimes will be saved in csv files
        contained in the given directory. The directory must exist.
        This will create a file for each problem size T and save
        the Pb number, the number of core and the runtime computed
        in two different ways.
    debug: int, optional (default:0)
        The greater it is, the more verbose the algorithm

    '''
    try:
        common_args = dict(logging=False, log_rate='log1.6', i_max=i_max,
                           t_max=t_max, debug=debug, tol=5e-2)

        print('construct problem')
        pbs, D = fun_rand_problems(N=50, S=100, K=10, d=6,
                                   noise_level=0)
        D0 = np.copy(D)
        N = len(pbs)
        print('End\n')
        lmbd = .3

        pbs[0].compute_DD()
        dcp = DICOD(pbs[0], n_jobs=n_jobs, hostfile=hostfile,
                    positive=True, use_seg=5, **common_args)

        grad_D = [np.zeros(D.shape) for _ in range(N)]
        grad_nz = set()
        cost = np.zeros(N)
        cost_i = 1e6

        order = np.arange(N)
        np.random.shuffle(order)
        mini_batch_size = 5
        n_batch = N // mini_batch_size
        current_batch = 0
        current_epoch = 0
        time_epoch = time()

        while current_epoch < n_epoch:
            # Stochastic choice of a ponit

            pb_batch = [
                [pbs[i], i] for i in order[
                    current_batch * mini_batch_size:
                    (current_batch + 1) * mini_batch_size]]
            current_batch += 1
            current_batch %= n_batch
            DD = None
            new = False
            for pb, i0 in pb_batch:
                pb.D = D

                # Sparse coding
                pb.reset()
                DD = dcp.fit(pb, DD=DD)

                # Update cost and D gradient
                new |= cost[i0] == 0
                cost[i0] = pb.cost()
                grad_D[i0] = pb.grad_D(pb.pt)
                grad_nz.add(i0)

            # Logging
            N_see = len(grad_nz)
            cost_i1 = cost_i
            cost_i = np.sum(cost, axis=0) / N_see
            print('End mini_batch {:3} with cost {:e}'
                  ''.format(int(current_batch), cost_i))
            if graphical_cost is not None:
                log.graphical_cost(cost=np.sum(cost, axis=0) / N_see)
            if current_batch == 0:
                print('=' * 79)
                print('End Epoch {} in {:.2}s'
                      ''.format(current_epoch, time() - time_epoch))
                time_epoch = time()
                current_epoch += 1
                np.random.shuffle(order)
                print('=' * 79)

            # reg = np.zeros(D.shape)
            # reg[:, :, :-2] += D[:, :, 2:]
            # reg[:, :, :] -= D[:, :, :]
            # reg[:, :, :] /= np.sqrt(1e-2+D[:, :, :]*D[:, :, :])

            # Update dictionary with unit norm regularization
            grad = np.sum(grad_D, axis=0) / N_see
            gw = np.sum(grad * D, axis=-1)
            gg = np.sum(grad * grad, axis=-1)
            lmbd = min(np.min(.5 / np.sqrt(gg - gw * gw)), lmbd)
            assert not np.isnan(lmbd)
            alpha = lmbd * gw - np.sqrt(1 - lmbd * lmbd * (gg - gw * gw))
            D *= alpha[:, :, None]
            D -= lmbd * grad

            if cost_i >= cost_i1 and not new:
                # IPython.embed()
                lmbd *= .7
        from sys import stdout as out
        print('=' * 79)
        print('Fit the pb to the latest dictionary')
        print('=' * 79)
        for i, pb in enumerate(pbs):
            pb.D = D
            pb.reset()
            dcp.fit(pb)
            out.write('\rCompute rpz: {:7.2%}'.format(i / N))
            out.flush()
        print('\rCompute rpz: {:7}'.format('Done'))
    except KeyboardInterrupt:
        from sys import stdout as out
        print('=' * 79)
        print('Fit the pb to the latest dictionary')
        print('=' * 79)
        for i, pb in enumerate(pbs):
            pb.D = D
            pb.reset()
            dcp.fit(pb)
            out.write('\rCompute rpz: {:7.2%}'.format(i / N))
            out.flush()
        print('\rCompute rpz: {:7}'.format('Done'))

    finally:
        print("Frob norm D", np.sum((D - D0) ** 2) / np.sum(D0 * D0))
        IPython.embed()
        log.end()
