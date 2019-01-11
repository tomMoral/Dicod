import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


SAVE_FILE_PATTERN = "scaling_n_jobs_T{}{}{}.{{}}"


def plot_scaling_n_jobs(data_file_name, dir, config):
    params = re.compile(r"_([0-9]+)(.*).csv")
    T, alg = params.findall(data_file_name)[0]
    save_file_name = SAVE_FILE_PATTERN.format(T, alg, config['suffix'])
    save_file_name = os.path.join(dir, save_file_name)
    T = int(T)

    # Read and parse the data
    with open(data_file_name) as f:
        lines = f.readlines()
    arr = defaultdict(lambda: [])
    for l in lines:
        r = list(map(float, l.split(',')[1:]))
        arr[r[0]] += [r]

    # Create a new figure
    fig, ax = plt.subplots(1, num="scaling_jobs_{}_{}".format(T, alg))

    # Display the different runtimes
    l, L = 1e6, 0
    for k, v in arr.items():
        V = np.mean(v, axis=0)[1]
        ax.scatter(k, V, color="b")
        l, L = min(l, V), max(L, V)

    # Plot the linear and quadratic scaling
    n_jobs = np.asarray(list(arr.keys()), dtype=int)
    n_jobs.sort()

    m, M = n_jobs.min(), n_jobs.max()
    t = np.logspace(np.log2(m), np.log2(2 * M), 200, base=2)
    R0 = np.mean(arr[m], axis=0)[1]
    ax.plot(t, R0 * m / t, 'k--', linewidth=2)
    ax.plot(t, R0 * (m / t)**2, 'r--', linewidth=2)
    tt = 8
    ax.text(tt, .4 * R0 * (m / tt)**2, "quadratic", rotation=-22,
            fontsize=20)
    ax.text(tt, R0 * m / tt, "linear", rotation=-14, bbox=dict(
        facecolor="white", edgecolor="white"), fontsize=20)

    # Plot the theoretical scaling
    th_scaling = R0 / (t * t * np.maximum(1 - 2 * (t / T)**2 * (
        1 + 2 * (t / T)**2)**(t / 2 - 1), 1e-5))

    ax.plot(t, th_scaling, "g-.", label="theoretical speedup",
            linewidth=2)

    # Plot the numerical optimum of the theoretical scaling
    break_p = np.where((th_scaling[2:] > th_scaling[1:-1]) &
                       (th_scaling[:-2] > th_scaling[1:-1]))[0] + 1
    ax.vlines(t[break_p], .1, 100000, "g", linestyle="-", linewidth=3)
    ax.text(.9 * t[break_p], .7 * R0 * m / tt, "$M^*$", rotation=0,
            bbox=dict(facecolor="w", edgecolor="w"))

    # Format the plot
    fig.set_size_inches(config['size'])
    fig.patch.set_alpha(0)
    ax.set_title(f"$T={T}W$", fontsize="xx-large")
    ax.legend(fontsize="xx-large", loc=3)
    ax.minorticks_off()

    ax.set_xscale('log')
    ax.set_xlabel("# cores $M$", fontsize="xx-large")
    ax.set_xticks(n_jobs)
    ax.set_xticklabels(n_jobs)
    ax.xaxis.labelpad = 20

    ax.set_yscale('log')
    ax.set_ylim((.2 * l, 1.7 * L))
    ax.set_ylabel("Runtime (s)", fontsize="xx-large")
    ax.yaxis.labelpad = 30

    plt.tight_layout()

    for ext in ['pdf', 'png']:
        plt.savefig(save_file_name.format(ext), dpi=150)