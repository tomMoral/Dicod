#!/scratch/thmoreau/miniconda3/bin/python
import os
import numpy as np
import matplotlib.pyplot as plt
from joblib import Memory

from alphacsc.init_dict import init_dictionary
from alphacsc.utils.dictionary import get_lambda_max

from dicod import dicod
from dicod.data import get_mandril
from dicod.utils.segmentation import Segmentation
from dicod.utils.shape_helpers import get_valid_shape
from dicod.utils.csc import compute_objective, reconstruct

mem = Memory(location='.')
hostfile = os.environ.get("MPI_HOSTFILE", None)


@mem.cache
def run_without_soft_lock(n_atoms=25, atom_support=(12, 12), reg=.01,
                          tol=5e-2, n_jobs=100, random_state=60):
    rng = np.random.RandomState(random_state)

    X = get_mandril()
    n_channels, *sig_shape = X.shape
    D_init = init_dictionary(X[None], n_atoms, atom_support, D_init='chunk',
                             rank1=False, random_state=rng)
    lmbd_max = get_lambda_max(X[None], D_init).max()
    reg_ = reg * lmbd_max

    z_hat, *_ = dicod(
        X, D_init, reg_, max_iter=1000000, n_jobs=n_jobs, strategy='greedy',
        tol=tol, hostfile=hostfile, verbose=1, use_soft_lock=False,
        z_positive=False, timing=False)
    pobj = compute_objective(X, z_hat, D_init, reg_)
    z_hat = np.clip(z_hat, -1e3, 1e3)
    print("[DICOD] final cost : {}".format(pobj))

    X_hat = reconstruct(z_hat, D_init)
    X_hat = np.clip(X_hat, 0, 1)
    return X_hat, pobj


reg = .01
tol = 5e-2
n_atoms = 25
w_world = 7
n_jobs = w_world * w_world
random_state = 60
atom_support = (16, 16)

X_hat, pobj = run_without_soft_lock(n_atoms, atom_support, reg, tol, n_jobs,
                                    random_state)


# Compute the worker segmentation for the image,
n_channels, *sig_shape = X_hat.shape
valid_shape = get_valid_shape(sig_shape, atom_support)
workers_segments = Segmentation(n_seg=(w_world, w_world),
                                signal_shape=valid_shape,
                                overlap=0)

fig = plt.figure("recovery")

ax = plt.subplot()
ax.imshow(X_hat.swapaxes(0, 2))
for i_seg in range(workers_segments.effective_n_seg):
    seg_bounds = np.array(workers_segments.get_seg_bounds(i_seg))
    seg_bounds = seg_bounds + np.array(atom_support) / 2
    ax.vlines(seg_bounds[1], *seg_bounds[0], linestyle='--')
    ax.hlines(seg_bounds[0], *seg_bounds[1], linestyle='--')
ax.axis('off')

plt.tight_layout()

fig.savefig(f"benchmarks_results/soft_lock_M{n_jobs}_"
            f"support{atom_support[0]}.pdf", dpi=300)
print("done")
