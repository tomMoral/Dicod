import PIL
import numpy as np
import matplotlib.pyplot as plt

from dicod.dicodil import dicodil
from dicod.utils.viz import plot_atom_and_coefs

from alphacsc.init_dict import init_dictionary


n_atoms = 65
atom_support = (48, 48)
reg = .1
random_state = 42

# Load the data
PIL.Image.MAX_IMAGE_PIXELS = 617967525
X = plt.imread("../../data/images/hubble/STScI-H-2016-39-a-Medium.jpg")
X = X / 255
X = X.swapaxes(0, 2)

D_init = init_dictionary(X[None], n_atoms, atom_support, D_init='chunk',
                         rank1=False, random_state=random_state)


dicod_kwargs = dict(tol=.01, use_soft_lock=True)
pobj, times, D_hat, z_hat = dicodil(
    X, D_init, reg=reg, z_positive=True, n_iter=10, n_jobs=400, eps=1e-3,
    verbose=2)


# Save the atoms
prefix = f"K{n_atoms}_L{atom_support}_reg{reg}_seed{random_state}_dicodil_"
prefix = prefix.replace(" ", "")
np.save(f"hubble/{prefix}D_hat.npy", D_hat)
np.save(f"hubble/{prefix}z_hat.npy", z_hat)

plot_atom_and_coefs(D_hat, z_hat, prefix)
