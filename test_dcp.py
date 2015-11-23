

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser('Test for the DICOD algorithm')
    parser.add_argument('--njobs', type=int, default=10,
                        help='# max of process launched')
    parser.add_argument('--niter', type=int, default=100,
                        help='# max of process launched')
    parser.add_argument('-T', type=int, default=150,
                        help='Size of the problem')
    parser.add_argument('-d', type=int, default=0,
                        help='Debug level for the algorithm')
    parser.add_argument('--hostfile', type=str, default=None,
                        help='Hostfile to pass to MPI')
    parser.add_argument('--nrep', type=int, default=10,
                        help='# of repetition for each value of M')
    parser.add_argument('--tmax', type=int, default=20,
                        help='Max time for each algorithm in sec')
    parser.add_argument('--save', type=str, default=None,
                        metavar='DIRECTORY', help='If present, save'
                        ' the result in the given DIRECTORY')
    parser.add_argument('--graph', action='store_true',
                        help='Show a graphical logging')
    parser.add_argument('--jobs', action='store_true',
                        help='Compute the runtime for different number '
                             'of cores')
    parser.add_argument('--met', action='store_true',
                        help='Compute the optimization algorithms')
    parser.add_argument('--no-display', action='store_false',
                        help='Compute the optimization algorithms')
    parser.add_argument('--step', action='store_true',
                        help='Convolutional dicitonary learning')
    args = parser.parse_args()

    graphical_cost = None
    if args.graph:
        graphical_cost = 'Cost'

    if args.jobs:
        from utils.iter_njobs import iter_njobs

        iter_njobs(T=args.T, max_jobs=args.njobs,  n_rep=args.nrep,
                   save_dir=args.save, i_max=5e6, t_max=7200,
                   hostfile=args.hostfile, lgg=False, graphical_cost=None,
                   debug=args.d)

    if args.met:
        from utils.compare_methods import compare_met

        compare_met(T=args.T, save_dir=args.save, i_max=5e6, t_max=args.tmax,
                    n_jobs=args.njobs, hostfile=args.hostfile,
                    graphical_cost=graphical_cost, display=args.no_display,
                    debug=args.d)

    if args.step:
        from utils.step_detect import step_detect
        step_detect(save_dir=args.save, i_max=5e6, t_max=args.tmax,
                    n_jobs=args.njobs, hostfile=args.hostfile,
                    n_iter=args.niter,
                    graphical_cost=graphical_cost, debug=args.d)
