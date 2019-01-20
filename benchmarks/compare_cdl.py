
import pandas
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

from dicod.dicodil import dicodil
from dicod.data import get_mandril
from alphacsc.utils.dictionary import get_lambda_max
from sporco.dictlrn.prlcnscdl import ConvBPDNDictLearn_Consensus

from alphacsc.init_dict import init_dictionary

from joblib import Memory
mem = Memory(location='.')


ResultItem = namedtuple('ResultItem', [
    'n_atoms', 'atom_support', 'reg', 'n_jobs', 'random_state', 'method',
    'z_positive', 'times', 'pobj'])


@mem.cache(ignore=['n_iter'])
def run_one(method, n_atoms, atom_support, reg, z_positive, n_jobs, n_iter,
            random_state):

    X = get_mandril()
    D_init = init_dictionary(X[None], n_atoms, atom_support, D_init='chunk',
                             rank1=False, random_state=random_state)

    if method == 'wohlberg':
        ################################################################
        #            Run parallel consensus ADMM
        #
        lmbd_max = get_lambda_max(X[None], D_init).max()
        print("Lambda max = {}".format(lmbd_max))
        reg_ = reg * lmbd_max

        D_init_ = np.transpose(D_init, axes=(3, 2, 1, 0))
        X_ = np.transpose(X[None], axes=(3, 2, 1, 0))

        options = {
            'Verbose': True,
            'StatusHeader': False,
            'MaxMainIter': n_iter,
            'CBPDN': dict(NonNegCoef=z_positive),
            'CCMOD': dict(ZeroMean=False),
            'DictSize': D_init_.shape,
            }
        opt = ConvBPDNDictLearn_Consensus.Options(options)
        cdl = ConvBPDNDictLearn_Consensus(
            D_init_, X_, lmbda=reg_, nproc=n_jobs, opt=opt, dimK=1, dimN=2)

        _, pobj = cdl.solve()
        print(pobj)

        itstat = cdl.getitstat()
        times = itstat.Time

    elif method == "dicodil":
        pobj, times, D_hat, z_hat = dicodil(
            X, D_init, reg=reg, z_positive=z_positive, n_iter=n_iter, eps=1e-5,
            n_jobs=n_jobs, verbose=2, tol=1e-3)
        pobj = pobj[::2]
        times = np.cumsum(times)[::2]

    else:
        raise NotImplementedError()

    return ResultItem(
        n_atoms=n_atoms, atom_support=atom_support, reg=reg, n_jobs=n_jobs,
        random_state=random_state, method=method, z_positive=z_positive,
        times=times, pobj=pobj)


def run_benchmark():
    n_rep = 5
    n_iter = 500
    n_jobs = 36
    reg = .1
    n_atoms = 64
    atom_support = (12, 12)
    z_positive = True

    results = []

    for method in ['wohlberg', 'dicodil']:
        for random_state in range(n_rep):
            args = (method, n_atoms, atom_support, reg, z_positive, n_jobs,
                    n_iter, random_state)
            if False and method == 'dicodil':
                results.append(run_one.call(*args))
            else:
                results.append(run_one(*args))

    # Save results
    df = pandas.DataFrame(results)
    df.to_pickle("benchmarks_results/compare_cdl.pkl")


def plot_results():
    df = pandas.read_pickle("benchmarks_results/compare_cdl.pkl")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('Compare DICODIL with wohlberg')
    parser.add_argument('--plot', action='store_true',
                        help='Plot the result of the benchmark')
    args = parser.parse_args()

    if args.plot:
        plot_results()
    else:
        run_benchmark()
