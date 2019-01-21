import os
import pandas
import datetime
import numpy as np
from joblib import Memory
import matplotlib.pyplot as plt
from collections import namedtuple


from dicod import dicod
from dicod.data import simulate_data
from dicod.utils.dictionary import get_lambda_max


COLOR = ['C0', 'C1', 'C2']
SAVE_DIR = "benchmarks_results"

list_n_times = [151]  # , 750]
strategies = ['lgcd', 'random', 'greedy'][::2]
names = ["LGCD", "Randomized", "Greedy"][::2]
styles = ["o-", "h-", "s-"][::2]


mem = Memory(location='.')


ResultItem = namedtuple('ResultItem', [
    'reg', 'n_jobs', 'strategy', 'tol', 'seed', 'pobj'])


@mem.cache(ignore=['common_args'])
def run_one(T, L, K, d, noise_level, seed_pb, n_jobs, reg, tol, strategy,
            common_args):

    X, D_hat = simulate_data(T, L, K, d, noise_level, seed=seed_pb)
    lmbd_max = get_lambda_max(X, D_hat)
    reg_ = reg * lmbd_max

    n_seg = 1
    strategy_ = strategy
    if strategy == 'lgcd':
        n_seg = 'auto'
        strategy_ = "greedy"

    *_, pobj, _ = dicod(X, D_hat, reg_, n_jobs=n_jobs, tol=tol,
                        strategy=strategy_, n_seg=n_seg, **common_args)
    print(pobj)

    return ResultItem(reg=reg, n_jobs=n_jobs, strategy=strategy, tol=tol,
                      seed=seed_pb, pobj=pobj)


def run_scaling_1d_benchmark(T=151, max_jobs=75, n_rep=10, reg=0.1, tol=1e-5,
                             timeout=7200, strategy='lgcd', seed=None):
    '''Run DICOD strategy for a certain problem with different value
    for n_jobs and store the runtime in csv files if given a save_dir.

    Parameters
    ----------
    T: int, optional (default: 300)
        Size of the generated problems
    max_jobs: int, optional (default: 75)
        The strategy will be run on problems with a number
        of cores varying from 5 to max_jobs in a log2 fashion
    n_rep: int, optional (default: 10)
        Number of different problem solved for all the different
        number of cores.
    max_iter: int, optional (default: 5e6)
        maximal number of iteration run by DICOD
    timeout: int, optional (default: 7200)
        maximal running time for DICOD. The default timeout
        is 2 hours
    strategy: enum, optional (default: ALGO_GS)
        Algorithm used to select the update for the coordinate descent. It
        should be either ALGO_GS (greedy selection) or ALGO_RANDOM (random
        selection).
    seed: int, optional (default:None)
        seed the rng of numpy to obtain fixed set of problems

    '''

    L = 150
    K = 10
    d = 7
    noise_level = 1

    rng = np.random.RandomState(seed)
    file_name = f'runtimes_n_jobs_{T}_{strategy}.csv'
    file_name = os.path.join(SAVE_DIR, file_name)

    common_args = dict(timing=False, timeout=timeout, max_iter=int(5e8),
                       verbose=2)

    n_jobs = np.logspace(0, np.log2(75), 10, base=2)
    n_jobs = [int(round(nj)) for nj in n_jobs if nj <= max_jobs]
    n_jobs = np.unique(n_jobs)
    n_jobs = n_jobs[::-1]

    seeds = [rng.randint(4294967295) for _ in range(n_rep)]
    results = []

    for j, seed_pb in enumerate(seeds):
        for nj in n_jobs:
            for reg in [.1, .01, .001]:
                for tol in [1e-3, 1e-4]:  # , 1e-5]:
                    for strategy in ['lgcd', 'greedy']:
                        res = run_one(T, L, K, d, noise_level, seed_pb, nj,
                                      reg, tol, strategy, common_args)
                        runtime = res.pobj[-1][1]
                        print('=' * 79)
                        print('[{}] PB{}: End process with {} jobs  in {:.2f}s'
                              .format(datetime.datetime.now()
                                      .strftime("%I:%M"), j, nj, runtime))
                        print('\n' + '=' * 79)
                        results.append(res)
    results = pandas.DataFrame(results)
    results.to_pickle("benchmarks_results/scaling_1d.pkl")


def plot_scaling_1d_benchmark():
    # compute the width of the bars
    n_group = len(list_n_times)
    n_bar = len(strategies)
    width = 1 / ((n_bar + 1) * n_group - 1)

    fig = plt.figure('comparison CD', figsize=(6, 3.5))
    ax_bar = fig.subplots()
    xticks, labels = [], []
    for i, n_times in enumerate(list_n_times):
        fig_scaling = plt.figure(f'Scaling T={n_times}', figsize=(6, 3))
        ax_scaling = fig_scaling.subplots()
        handles = []
        xticks.append(((i + .5) * (n_bar + 1)) * width)
        labels.append(f"$T = {n_times}L$")
        for j, (strategy, name, style) in enumerate(zip(strategies, names,
                                                    styles)):
            col_name = ['pb', 'n_jobs', 'runtime', 'runtime1']
            csv_name = (f"benchmarks_results/runtimes_n_jobs_"
                        f"{n_times}_{strategy}.csv")

            try:
                df = pandas.read_csv(csv_name, names=col_name)
            except FileNotFoundError:
                print(f"Not found {csv_name}")
                continue

            runtimes_1 = df[df['n_jobs'] == 1]['runtime'].values

            position = (i * (n_bar + 1) + j + 1) * width

            handles.append(ax_bar.bar(position, height=np.mean(runtimes_1),
                                      width=width, color=COLOR[j], label=name,
                                      hatch='//' if strategy == 'lgcd' else '')
                           )

            ax_bar.plot(
                np.ones_like(runtimes_1) * position,
                runtimes_1, '_', color='k')

            n_jobs = df['n_jobs'].unique()
            n_jobs.sort()

            runtimes_scale = []
            runtimes_scale_mean = []
            for n in n_jobs:
                runtimes_scale.append(df[df['n_jobs'] == n]['runtime'].values)
                runtimes_scale_mean.append(np.mean(runtimes_scale[-1]))
            runtimes_scale_mean = np.array(runtimes_scale_mean)
            if strategy != 'random':

                t = np.logspace(0, np.log2(2 * n_jobs.max()), 3, base=2)
                R0 = runtimes_scale_mean.max()

                # Linear and quadratic lines
                p = 1 if strategy == 'lgcd' else 2
                ax_scaling.plot(t, R0 / t ** p, 'k--', linewidth=1)
                tt = 2
                bbox = None  # dict(facecolor="white", edgecolor="white")
                if strategy == 'lgcd':
                    ax_scaling.text(tt, 1.4 * R0 / tt, "linear", rotation=-14,
                                    bbox=bbox, fontsize=12)
                    name_ = "DiCoDiLe$_Z$"
                else:
                    ax_scaling.text(tt, 1.4 * R0 / tt**2, "quadratic",
                                    rotation=-25, bbox=bbox, fontsize=12)
                    name_ = "DICOD"
                ax_scaling.plot(n_jobs, runtimes_scale_mean, style,
                                label=name_, zorder=10, markersize=8)
                # for i, n in enumerate(n_jobs):
                #     x = np.array(runtimes_scale[i])
                #     ax_scaling.plot(np.ones(value.shape) * n, value, 'k_')

        if n_times == 150:
            y_lim = (.5, 1e3)
        else:
            y_lim = (2, 2e4)
        ax_scaling.vlines(n_times / 4, *y_lim, 'g', '-.')
        ax_scaling.set_ylim(y_lim)
        ax_scaling.set_xscale('log')
        ax_scaling.set_yscale('log')
        ax_scaling.set_xlim((1, 75))
        ax_scaling.grid(True, which='both', axis='x', alpha=.5)
        ax_scaling.grid(True, which='major', axis='y', alpha=.5)
        # ax_scaling.set_xticks(n_jobs)
        # ax_scaling.set_xticklabels(n_jobs, fontsize=12)
        ax_scaling.set_ylabel("Runtime [sec]", fontsize=12)
        ax_scaling.set_xlabel("# workers $W$", fontsize=12)
        ax_scaling.legend(fontsize=14)
        fig_scaling.tight_layout()
        fig_scaling.savefig(f"benchmarks_results/scaling_T{n_times}.pdf",
                            dpi=300, bbox_inches='tight', pad_inches=0)

    ax_bar.set_ylabel("Runtime [sec]", fontsize=12)
    ax_bar.set_yscale('log')
    ax_bar.set_xticks(xticks)
    ax_bar.set_xticklabels(labels, fontsize=12)
    ax_bar.set_ylim(1, 2e4)
    ax_bar.legend(bbox_to_anchor=(-.02, 1.02, 1., .3), loc="lower left",
                  handles=handles, ncol=3, fontsize=14, borderaxespad=0.)
    fig.tight_layout()
    fig.savefig("benchmarks_results/CD_strategies_comparison.png", dpi=300,
                bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('')
    parser.add_argument('--plot', action="store_true",
                        help='Plot the results of the benchmarl')
    args = parser.parse_args()

    if args.plot:
        plot_scaling_1d_benchmark()
    else:
        seed = 422742
        run_scaling_1d_benchmark(seed=seed, n_rep=5)
