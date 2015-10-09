import numpy as np
from numpy.random import rand
from cdl.multivariate_convolutional_coding_problem import \
    MultivariateConvolutionalCodingProblem
from cdl.dicod import DICOD

def fun_rand_problem(T, S, K, d, lmbd, noise_level):
    rho = K/(d*S)
    t = np.arange(S)/S
    D = [[10*rand()*np.sin(2*np.pi*K*rand()*t +
                           (0.5-rand())*np.pi)
          for _ in range(d)]
         for _ in range(K)]
    D = np.array(D)
    nD = np.sqrt((D**2).sum(axis=-1))[:, :, np.newaxis]
    D /= nD + (nD == 0)
    Z = (rand(K, (T-1)*S+1) < rho)*rand(K, (T-1)*S+1)*10
    print("\nNon zero coefficients: ", len(Z.nonzero()[0]),
          "Sparse prob: ", K/(d*S))
    X = np.array([[np.convolve(zk, dk, 'full') for dk in Dk]
                  for Dk, zk in zip(D, Z)]).sum(axis=0)
    X += noise_level*np.random.normal(size=X.shape)

    z0 = np.zeros((K, (T-1)*S+1))
    pb = MultivariateConvolutionalCodingProblem(
        D, X, z0=z0, lmbd=lmbd)
    pb.compute_DD()
    return pb


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser('Test for the DICOD algorithm')
    parser.add_argument('--n_jobs', type=int, default=10,
                        help='# max of process launched')
    parser.add_argument('-T', type=int, default=1500,
                        help='Size of the problem')
    parser.add_argument('--hostfile', type=str, default=None,
                        help='Hostfile to pass to MPI')
    parser.add_argument('--n_rep', type=int, default=10,
                        help='# of repetition for each value of M')
    args = parser.parse_args()

    i_max = 5e6
    debug = 5
    t_max = 7200
    lgg = False
    graphical_cost = None
    met = None
    rep = 10
    nj = 1
    SAVE_DIR = "save_exp/"
    common_args = dict(logging=lgg, log_rate='log1.6', i_max=i_max,
                       t_max=t_max, graphical_cost=graphical_cost,
                       debug=debug, tol=5e-2)

    T = args.T
    S = 150
    K = 10
    d = 7
    lmbd = 0.1

    noise_level = 1

    for j in range(args.n_rep):
        pb = fun_rand_problem(T, S, K, d, lmbd, noise_level)

        dcp = DICOD(pb, n_jobs=args.n_jobs, hostfile=args.hostfile,
                    **common_args)

        runtimes = []
        n_jobs = np.logspace(np.log2(2), np.log2(75), 10, base=2)[::-1]
        n_jobs = n_jobs[n_jobs <= args.n_jobs]
        n_jobs = n_jobs[::-1]
        for nj in n_jobs:
            dcp.n_jobs = int(round(nj))

            dcp._init_pool()
            dcp.end()
            runtimes += [[dcp.runtime, dcp.runtime2]]
            dcp.reset()
            import time
            time.sleep(1)
            rt = runtimes[-1]
            with open('runtimes_{}.csv'.format(T), 'a') as f:
                f.write('Pb{},{},{},{}\n'.format(
                    j, int(round(nj)), rt[0], rt[1]))
            print('End process ', nj, 'Pb', j)

    dcp.reset()

    dcp._init_pool()
    dcp.end()
