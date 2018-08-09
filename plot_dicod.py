import os
import re
import glob
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

CONFIG_METHOD = {
    'default': {
        'size': (12, 6), 'sns_style': None, 'suffix': "",
        'eps': 1e-4, 'legend': dict(fontsize=16, loc=3, ncol=1, frameon=False)
    },
    'presentation': {
        'size': (12, 7), 'sns_style': 'darkgrid', 'suffix': "_seaborn",
        'eps': 1e-4, 'legend': dict(fontsize=16, loc=3, ncol=1, frameon=False)
    },
    'small': {
        'size': (6.4, 4.8), 'sns_style': 'darkgrid', 'suffix': "_small",
        'eps': 1e-4, 'legend': dict(fontsize=16, loc=4, ncol=1, frameon=True)
    },
    'conference': {
        'size': (12, 8), 'sns_style': None, 'suffix': "_conf",
        'eps': 1e-4, 'legend': dict(fontsize=16, loc=3, ncol=1, frameon=False)
    }
}

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser('Test for the DICOD algorithm')
    parser.add_argument('--dir', type=str, default=None,
                        metavar='DIRECTORY', help='If present, save'
                        ' the result in the given DIRECTORY')
    parser.add_argument('--jobs', action='store_true',
                        help='Plot the runtime for different number '
                             'of cores')
    parser.add_argument('--lmbd', action='store_true',
                        help='Plot the scaling relatively to lmbd.')
    parser.add_argument('--met', action='store_true',
                        help='Plot the comparison of optimization algorithms')
    parser.add_argument('--config', type=str, default='default',
                        help='Configuration of the plot.')
    args = parser.parse_args()

    if args.met:
        data_file_pattern = os.path.join(args.dir, "cost_curves*.pkl")
        save_file_pattern = os.path.join(args.dir,
                                         "compare_algos_{{}}_{}{}.pdf")

        config = CONFIG_METHOD[args.config]
        if config['sns_style'] is not None:
            sns.set_style(config['sns_style'])
        method_files = glob.glob(data_file_pattern)
        for data_file_name in method_files:
            params = re.compile(r"T[0-9]+_K[0-9]+_njobs[0-9]+")
            params = params.findall(data_file_name)[0]
            n_jobs = int(re.compile(r"njobs([0-9]+)").findall(params)[0])
            save_file_name = save_file_pattern.format(params, config['suffix'])

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
                legend = plt.legend(**config['legend'])
                plt.xlabel(xlabel, fontsize=18)
                plt.ylabel('Cost $E(Z^{(q)}) - E(Z^*)$', fontsize=18)
                plt.xlim(lim)
                plt.ylim((cost_lim[0], cost_lim[1] * 2))
                plt.xticks(size=14)
                plt.yticks(size=14)
                plt.tight_layout()
                plt.savefig(save_file_name.format(title), dpi=150)

            plt.show()
