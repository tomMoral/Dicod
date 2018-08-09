import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt


from utils.plots import plot_scaling_n_jobs
from utils.plots import plot_compare_methods

PLOT_CONFIGS = {
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

    # Load figure config
    config = PLOT_CONFIGS[args.config]
    if config['sns_style'] is not None:
        sns.set_style(config['sns_style'])

    plots = []
    if args.met:
        plots.append(("cost_curves*.pkl", plot_compare_methods))

    if args.jobs:
        plots.append(("runtimes_*.csv", plot_scaling_n_jobs))

    for pattern, plot_func in plots:
        data_file_pattern = os.path.join(args.dir, pattern)
        data_files = glob.glob(data_file_pattern)

        for data_file_name in data_files:
            plot_func(data_file_name, args.dir, config)
        plt.show()
