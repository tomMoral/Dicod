import numpy as np
from time import time
from numpy.random import rand, randint

from cdl.multivariate_convolutional_coding_problem \
    import MultivariateConvolutionalCodingProblem
from utils.rand_problem import fun_rand_problem

if __name__ == '__main__':
    from toolbox.logger import Logger
    log = Logger(name='root', levl=20)
    import argparse
    parser = argparse.ArgumentParser('Test For CCP')
    parser.add_argument('-d', action='store_true',
                        help='Debug mode')
    parser.add_argument('-g', action='store_true',
                        help='show codes')
    parser.add_argument('--hostfile', type=str, default=None,
                        help='Use the hostfile to setup the system')
    parser.add_argument('--dcp', action='store_true', help='run for DCD')
    parser.add_argument('--dcp10', action='store_true', help='run for DCD10')
    parser.add_argument('--cp', action='store_true', help='run for CCD')
    parser.add_argument('--fss', action='store_true', help='run for FSS')
    parser.add_argument('--fista', action='store_true', help='run for Fista')
    parser.add_argument('--sdcp', action='store_true', help='run for SeqDICOD')
    parser.add_argument('--do', type=int, nargs='+', default=[])
    parser.add_argument('--fix', action='store_true', help='with fix sparsity')
    args = parser.parse_args()

    from cdl.feature_sign_search import FSS
    from cdl.fista import FISTA
    from cdl.dicod import DICOD

    # Construct a problem
    S = 200
    Ts = [80, 160, 250, 500, 750, 1000, 2000]
    K = 10
    d = 7
    lmbd = 0.1
    noise_level = 1

    i_max = 5e6
    debug = 0
    t_max = 7200
    graphical_cost = 'cost'
    met = None
    rep = 10
    nj = 1
    SAVE_DIR = "save_exp/runtimes/"

    if args.d:
        log.set_level(10)
        debug = 5

    common_args = dict(logging=True, log_rate='log1.6', i_max=i_max,
                       t_max=t_max, graphical_cost=graphical_cost,
                       debug=debug, tol=5e-2)

    np.random.seed(23543)
    pb = fun_rand_problem(4, 2, 2, 2, lmbd, noise_level)
    methods = []
    if len(args.do) == 0:
        args.do = range(5)
    if args.dcp:
        methods += [('DCP30', DICOD(pb, n_jobs=30,
                                    hostfile=args.hostfile,
                                    **common_args))]
    if args.dcp10:
        methods += [('DCP15', DICOD(pb, n_jobs=15,
                                    hostfile=args.hostfile,
                                    **common_args))]
    if args.sdcp:
        methods += [('sDCP30', DICOD(pb, n_jobs=1, use_seg=30,
                                     **common_args))]
    if args.cp:
        methods += [('CD', DICOD(pb, n_jobs=1,
                                 **common_args))]
    if args.fss:
        methods += [('FSS', FSS(pb, n_zero_coef=40,
                                **common_args))]
    if args.fista:
        methods += [('Fista', FISTA(pb, fixe=True,
                                    **common_args))]
    do = args.do

    n_l2 = lambda x: np.sqrt((x*x).sum())
    n_linf = lambda x: np.max(abs(x))
    np.random.seed(23543)
    for T in Ts:
        print('='*79+'\nT: {}\n'.format(T)+'='*79)
        for j in range(rep):
            pb = fun_rand_problem(T, S, K, d, lmbd, noise_level)
            if j in do:
                cost_init = pb.cost()
                for i, (name, met) in enumerate(methods):
                    log.info(name)
                    pb.reset()
                    met.fit(pb)
                    cost_end = pb.cost()
                    if not args.d:
                        with open(SAVE_DIR+'runtime_{}_T{}.csv'
                                  ''.format(name, T),
                                  'a') as f:
                            f.write(';'.join([str(T), str(met.runtime),
                                              str(cost_init),
                                              str(cost_end),
                                              str(met.iteration)]))
                            f.write('\n')
                    print('Cost_gap:', cost_init-cost_end)
                    print('Time:', met.runtime)
                    print('-'*79+'\n')
                    if met.runtime > t_max:
                        del methods[i]
        log.process_queue()

    import IPython
    IPython.embed()
