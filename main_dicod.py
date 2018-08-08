from dicod.dicod import ALGO_GS, ALGO_RANDOM



if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser('Test for the DICOD algorithm')
    parser.add_argument('--njobs', type=int, default=10,
                        help='# max of process launched')
    parser.add_argument('--nepoch', type=int, default=4,
                        help='# max of process launched')
    parser.add_argument('-T', type=int, default=150,
                        help='Size of the problem')
    parser.add_argument('-K', type=int, default=100,
                        help='Number of dictionary elements')
    parser.add_argument('--debug', '-d', dest='d', type=int, default=0,
                        help='Debug level for the algorithm')
    parser.add_argument('--hostfile', type=str, default=None,
                        help='Hostfile to pass to MPI')
    parser.add_argument('--nrep', type=int, default=10,
                        help='# of repetition for each value of M')
    parser.add_argument('--tmax', type=int, default=60,
                        help='Max time for each algorithm in sec')
    parser.add_argument('--exp', type=str, default=None,
                        metavar='DIRECTORY', help='If present, exp'
                        ' the result in the given DIRECTORY')
    parser.add_argument('--jobs', action='store_true',
                        help='Compute the runtime for different number '
                             'of cores')
    parser.add_argument('--lmbd', action='store_true',
                        help='Compute the scaling relatively to lmbd.')
    parser.add_argument('--rcd', action='store_true',
                        help='Uses the random selection in CD')
    parser.add_argument('--seg', action='store_true',
                        help='Uses locally greedy selection in CD')
    parser.add_argument('--met', action='store_true',
                        help='Compute the optimization algorithms')
    parser.add_argument('--no-display', action='store_false',
                        help='Compute the optimization algorithms')
    parser.add_argument('--step', action='store_true',
                        help='Convolutional dictionary learning with signals '
                        'from humans walking.')
    parser.add_argument('--rand', action='store_true',
                        help='Convolutional dictionary learning with randomly '
                        'generated signals.')
    parser.add_argument('--run', type=str, nargs="+", default="all",
                        help='list of jobs to compute')
    parser.add_argument('--optim', type=str, default="dicod",
                        help='Optimizer to test for scaling performances.')
    args = parser.parse_args()

    if args.jobs:
        from utils.iter_njobs import iter_njobs
        algorithm = ALGO_RANDOM if args.rcd else ALGO_GS
        # # Extract njobs in list of str
        # run = []
        # for r in args.run:
        #     try:
        #         run += [int(r)]
        #     except ValueError:
        #         run += [r]
        iter_njobs(T=args.T, max_jobs=args.njobs, n_rep=args.nrep,
                   exp_dir=args.exp, max_iter=5e8, timeout=args.tmax,
                   hostfile=args.hostfile, lgg=False, debug=args.d,
                   algorithm=algorithm, seed=422742,
                   run=args.run, use_seg=args.seg)

    if args.lmbd:
        from utils.scaling_lmbd import scaling_lmbd
        scaling_lmbd(T=args.T, n_jobs=args.njobs, n_rep=args.nrep,
                     exp_dir=args.exp, max_iter=5e9, timeout=args.tmax,
                     hostfile=args.hostfile, lgg=False, optimizer=args.optim,
                     debug=args.d, seed=422742)

    if args.met:
        from utils.compare_methods import compare_met

        compare_met(T=args.T, K=args.K, exp_dir=args.exp, max_iter=5e8,
                    timeout=args.tmax, n_jobs=args.njobs, debug=args.d,
                    hostfile=args.hostfile, display=args.no_display)

    if args.step:
        from utils.step_detect import step_detect
        step_detect(exp_dir=args.exp, max_iter=5e6, timeout=args.tmax,
                    n_jobs=args.njobs, hostfile=args.hostfile,
                    n_epoch=args.nepoch, debug=args.d)

    if args.rand:
        from utils.dict_learn import dict_learn
        dict_learn(exp_dir=args.exp, max_iter=5e6, timeout=args.tmax,
                   n_jobs=args.njobs, hostfile=args.hostfile,
                   n_epoch=args.nepoch, debug=args.d)
