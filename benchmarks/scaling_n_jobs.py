import os
import pandas
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

from dicod import dicod
from dicod.utils import check_random_state
from dicod.utils.shape_helpers import get_valid_shape

from alphacsc.utils.dictionary import get_lambda_max

from joblib import Memory
mem = Memory(location='.')

ResultItem = namedtuple('ResultItem', [
    'n_atoms', 'atom_support', 'reg', 'n_jobs', 'n_seg', 'strategy', 'tol',
    'dicod_kwargs', 'seed', 'sparsity', 'pobj'])


def get_problem(n_atoms, atom_support, seed):
    data_dir = os.environ.get("DATA_DIR", "../../data")
    mandril = os.path.join(data_dir, "images/standard_images/mandril_color.tif")

    X = plt.imread(mandril) / 255
    X = X.swapaxes(0, 2)

    rng = check_random_state(seed)

    n_channels, *sig_shape = X.shape
    valid_shape = get_valid_shape(sig_shape, atom_support)

    indices = np.c_[[rng.randint(size_ax, size=(n_atoms))
                     for size_ax in valid_shape]].T
    D = np.empty(shape=(n_atoms, n_channels, *atom_support))
    for k, pt in enumerate(indices):
        D_slice = tuple([Ellipsis] + [
            slice(v, v + size_ax) for v, size_ax in zip(pt, atom_support)])
        D[k] = X[D_slice]
    sum_axis = tuple(range(1, D.ndim))
    D /= np.sqrt(np.sum(D*D, axis=sum_axis, keepdims=True))

    return X, D


@mem.cache(ignore=['timeout', 'max_iter', 'verbose'])
def run_one(n_atoms, atom_support, reg, n_jobs, strategy, tol, seed,
            timeout, max_iter, verbose, dicod_kwargs):
    # Generate a problem
    X, D = get_problem(n_atoms, atom_support, seed)
    lmbd = reg * get_lambda_max(X[None], D).max()

    if strategy == 'lgcd':
        n_seg = 'auto'
        effective_strategy = 'greedy'
    elif strategy in ["greedy", 'random']:
        n_seg = 1
        effective_strategy = strategy
    else:
        raise NotImplementedError(f"Bad strategy name {strategy}")

    z_hat, *_, pobj, cost = dicod(
        X, D, reg=lmbd, n_seg=n_seg, strategy=effective_strategy,
        n_jobs=n_jobs, timing=True, tol=tol, timeout=timeout,
        max_iter=max_iter, verbose=verbose, **dicod_kwargs)

    sparsity = len(z_hat.nonzero()[0]) / z_hat.size

    return ResultItem(n_atoms=n_atoms, atom_support=atom_support, reg=reg,
                      n_jobs=n_jobs, n_seg=n_seg, strategy=strategy,
                      tol=tol, dicod_kwargs=dicod_kwargs, seed=seed,
                      sparsity=sparsity, pobj=pobj)


def run_scaling_benchmark(max_n_jobs, n_rep=1):
    tol = 1e-3
    n_atoms = 5
    atom_support = (8, 8)

    verbose = 1
    timeout = 9000
    max_iter = int(1e8)

    dicod_kwargs = dict(z_positive=False, use_soft_lock=True)

    reg_list = np.logspace(-3, np.log10(.5), 10)[::-1]

    list_n_jobs = np.round(np.logspace(0, np.log10(20), 10)).astype(int)
    list_n_jobs = [int(v * v) for v in np.unique(list_n_jobs)[::-1]]
    print(list_n_jobs)

    results = []
    for reg in reg_list:
        for n_jobs in list_n_jobs:
            for strategy in ['greedy', 'lgcd']:  # , 'random']:
                for seed in range(n_rep):
                    res = run_one(n_atoms, atom_support, reg, n_jobs, strategy,
                                  tol, seed, timeout, max_iter, verbose,
                                  dicod_kwargs)
                    results.append(res)

    df = pandas.DataFrame(results)
    df.to_pickle("scaling_benchmark.plk")
    print(df)


if __name__ == "__main__":
    run_scaling_benchmark(max_n_jobs=400, n_rep=5)
