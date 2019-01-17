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
    'n_atoms', 'atom_support', 'reg', 'grid', 'tol', 'seed',
    'sparsity', 'pobj'])


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


@mem.cache(ignore=['verbose'])
def run_one_grid(n_atoms, atom_support, reg, n_jobs, grid, tol, seed,
                 verbose):
    # Generate a problem
    X, D = get_problem(n_atoms, atom_support, seed)
    reg_ = reg * get_lambda_max(X[None], D).max()

    if grid:
        w_world = 'auto'
    else:
        w_world = n_jobs

    dicod_kwargs = dict(z_positive=False, use_soft_lock=True, timeout=None,
                        max_iter=int(1e7))
    z_hat, *_, pobj, cost = dicod(
        X, D, reg=reg_, n_seg='auto', strategy='greedy', w_world=w_world,
        n_jobs=n_jobs, timing=True, tol=tol, verbose=verbose, **dicod_kwargs)

    sparsity = len(z_hat.nonzero()[0]) / z_hat.size

    return ResultItem(n_atoms=n_atoms, atom_support=atom_support, reg=reg,
                      grid=grid, tol=tol, seed=seed, sparsity=sparsity,
                      pobj=pobj)


def run_scaling_grid(n_rep=1):
    tol = 5e-3
    n_atoms = 5
    atom_support = (8, 8)

    reg_list = np.logspace(-3, np.log10(.5), 10)[::-1][2:3]

    list_n_jobs = np.round(np.logspace(0, np.log10(20), 10)).astype(int)
    list_n_jobs = [int(v * v) for v in np.unique(list_n_jobs)[::-1]]

    results = []
    for reg in reg_list:
        for n_jobs in list_n_jobs:
            for grid in [True, False]:
                for seed in range(n_rep):
                    try:
                        res = run_one_grid(
                            n_atoms, atom_support, reg,
                            n_jobs, grid, tol, seed, 1)
                        results.append(res)
                    except ValueError as e:
                        print(e)
                        continue

    df = pandas.DataFrame(results)
    df.to_pickle("benchmarks_results/scaling_grid.pkl")


def plot_scaling_benchmark():
    df = pandas.read_pickle("benchmarks_results/scaling_n_jobs.pkl")
    import matplotlib.lines as mlines
    handles = {}

    colors = ['C0', 'C1', 'C2']
    n_jobs = df['n_jobs'].unique()
    regs = df['reg'].unique()
    for reg, c in zip(regs, colors):
        for strategy, style in [('Greedy', '--'), ('LGCD', '-')]:
            s = strategy.lower()
            this_res = df[(df['reg'] == reg) & (df['strategy'] == s)]
            runtimes = []
            runtime_std = []
            for n in n_jobs:
                pobj = this_res[this_res['n_jobs'] == n]['pobj'].values
                end_times = [rt[-1][1] for rt in pobj if rt is not None]
                runtimes.append(np.mean(end_times))
                runtime_std.append(np.std(end_times))
            runtimes, runtime_std = np.array(runtimes), np.array(runtime_std)

            plt.loglog(n_jobs, runtimes, label=f"{strategy}_{reg:.2f}",
                       linestyle=style, c=c)
            plt.fill_between(n_jobs, runtimes - runtime_std,
                             runtimes + runtime_std, alpha=.1)
            color_handle = mlines.Line2D(
                [], [], linestyle='-', c=c, label=f"$\lambda = {reg:.2f}$")
            style_handle = mlines.Line2D(
                [], [], linestyle=style, c='k', label=f"{strategy}")
            handles[strategy] = style_handle
            handles[str(reg)] = color_handle
    plt.xlim((1, 400))
    plt.xticks(n_jobs, n_jobs, fontsize=14)
    plt.yticks(fontsize=14)
    plt.minorticks_off()
    plt.xlabel("# cores $M$", fontsize=18)
    plt.ylabel("Runtime [sec]", fontsize=18)

    keys = list(handles.keys())
    keys.sort()
    handles = [handles[k] for k in keys]
    plt.legend(handles=handles, ncol=2, fontsize=16)
    plt.tight_layout()
    plt.savefig("benchmarks_results/scaling_n_jobs.pdf")
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('Benchmark scaling performance for DICOD')
    parser.add_argument('--plot', action="store_true",
                        help='Plot the result of the benchmark')
    args = parser.parse_args()

    if args.plot:
        plot_scaling_benchmark()
    else:
        run_scaling_grid(n_rep=5)
