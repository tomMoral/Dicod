import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt


SAVE_FILE_PATTERN = "compare_methods_{{}}_{}{}.{{}}"


def plot_compare_methods(data_file_name, dir, config):
    params = re.compile(r"T[0-9]+_K[0-9]+_njobs[0-9]+")
    params = params.findall(data_file_name)[0]
    n_jobs = int(re.compile(r"njobs([0-9]+)").findall(params)[0])

    save_file_name = SAVE_FILE_PATTERN.format(params, config['suffix'])
    save_file_name = os.path.join(dir, save_file_name)

    style = {}
    style['CD'] = ('rd-', 1)
    style['RCD'] = ('gH-', 1)
    style['FCSC'] = ('k.-', 1)
    style['Fista'] = ('y*-', 1)
    style['DICOD$_{{{}}}$'.format(n_jobs // 2)] = ('ms-', 1)
    style['DICOD$_{{{}}}$'.format(n_jobs)] = ('bo-', 10)
    style['LGCD$_{{{}}}$'.format(n_jobs)] = ('c^-', 1)
    style['LGCD$_{{{}}}$'.format(n_jobs * 10)] = ('co-', 1)

    with open(data_file_name, "rb") as f:
        cost_curves = pickle.load(f)

    base_cost = cost_curves["CD"].pobj[0]
    c_min = min([c.pobj[-1] for c in cost_curves.values()])
    c_min -= config['eps']

    T_min = 1e-1
    time_lim = [T_min, 1e-1]
    cost_lim = [config['eps'] * .7, base_cost * 1.7]
    it_lim = [7e-1, 0]

    for name, (s, zorder) in style.items():
        try:
            it, t, cost = cost_curves[name]
        except KeyError:
            print("Did not find {} in cost_curves".format(name))
            continue
        t = np.maximum(t, T_min)

        cost_lim[0] = min(cost_lim[0], cost[-1] * .9)
        time_lim[0] = min(time_lim[0], t[1] * .9)
        time_lim[1] = max(time_lim[1], t[-1] * 1.1)
        it_lim[1] = max(it_lim[1], it[-1] * 1.1)
        c_min = min(c_min, np.min(cost))

        if name in style:
            if name == "Fista":
                name = "FISTA"
            plt.figure('time')
            plt.loglog(t, cost - c_min, s, label=name, linewidth=2,
                       markersize=9, zorder=zorder)
            plt.figure('iter')
            plt.loglog(it, cost - c_min, s, label=name, linewidth=2,
                       markersize=9, zorder=zorder)

    # Format the figures
    for title, xlabel, lim in [('iter', '# iteration $q$', it_lim),
                               ('time', 'Running Time (s)', time_lim)]:
        fig = plt.figure(title)
        fig.set_size_inches(config['size'])
        fig.patch.set_alpha(0)
        plt.hlines([config['eps']], lim[0], lim[1], linestyles='--',
                   colors='k')
        plt.legend(**config['legend'])
        plt.xlabel(xlabel, fontsize=18)
        plt.ylabel('Cost $E(Z^{(q)}) - E(Z^*)$', fontsize=18)
        plt.xlim(lim)
        plt.ylim((cost_lim[0], cost_lim[1] * 2))
        plt.xticks(size=14)
        plt.yticks(size=14)
        plt.tight_layout()

        for ext in ['pdf', 'png']:
            plt.savefig(save_file_name.format(title, ext), dpi=150)
