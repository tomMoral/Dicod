import os
import PIL
import pandas
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

from dicod.dicodil import dicodil
from alphacsc.utils.dictionary import get_lambda_max
from sporco.dictlrn.prlcnscdl import ConvBPDNDictLearn_Consensus

from alphacsc.init_dict import init_dictionary

from joblib import Memory
mem = Memory(location='.')


DATA_DIR = os.environ.get("DATA_DIR", "../../data")
IMAGE_PATH = "images/standard_images/mandril_color.tif"

IMAGE_PATH = os.path.join(DATA_DIR, IMAGE_PATH)


ResultItem = namedtuple('ResultItem', [
    'n_atoms', 'atom_support', 'reg', 'n_jobs', 'random_state', 'method',
    'z_positive', 'times', 'pobj'])


@mem.cache(ignore=['n_iter'])
def run_one(method, n_atoms, atom_support, reg, z_positive, n_jobs, n_iter,
            random_state):

    X = plt.imread(IMAGE_PATH)
    X = X / 255
    X = X.swapaxes(0, 2)
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
            'MaxMainIter': 3 * n_iter,
            'CBPDN': dict(NonNegCoef=z_positive),
            'CCMOD': dict(ZeroMean=False),
            'DictSize': D_init_.shape,
            }
        opt = ConvBPDNDictLearn_Consensus.Options(options)
        cdl = ConvBPDNDictLearn_Consensus(
            D_init_, X_, lmbda=reg_, nproc=n_jobs, opt=opt, dimK=1, dimN=2)

        cdl.solve()

        itstat = cdl.getitstat()
        times = itstat.Time
        pobj = itstat.ObjFun

    elif method == "dicodil":
        pobj, times, D_hat, z_hat = dicodil(
            X, D_init, reg=reg, z_positive=z_positive, n_iter=n_iter, eps=1e-4,
            n_jobs=n_jobs, verbose=2)
        pobj = pobj[::2]
        times = np.cumsum(times)[::2]

    else:
        raise NotImplementedError()

    return ResultItem(
        n_atoms=n_atoms, atom_support=atom_support, reg=reg, n_jobs=n_jobs,
        random_state=random_state, method=method, z_positive=z_positive,
        times=times, pobj=pobj)


n_iter = 100
n_jobs = 36
reg = .1
n_atoms = 64
atom_support = (12, 12)
z_positive = True


results = []


for method in ['wohlberg', 'dicodil']:
    for random_state in range(5):
        results.append(run_one(method, n_atoms, atom_support, reg, z_positive,
                               n_jobs, n_iter, random_state))


################################################################
#             Save the results
#
df = pandas.DataFrame(results)
df.to_pickle("benchmarks_results/compare_cdl.plk")
print(df)
