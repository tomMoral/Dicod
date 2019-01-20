
import numpy as np
from scipy import sparse

from dicod.data import get_hubble
from dicod.dicodil import dicodil
from dicod.utils.viz import plot_atom_and_coefs

from alphacsc.init_dict import init_dictionary


n_atoms = 25
random_state = 42


for size in ['Large', 'Medium']:
    # Load the data
    X = get_hubble(size=size)

    for reg in [.1, .3, .05]:
        for L in [32, 28]:

            D_init = init_dictionary(
                X[None], n_atoms, (L, L), D_init='chunk',
                rank1=False, random_state=random_state)

            dicod_kwargs = dict(tol=1e-1, use_soft_lock=True)
            pobj, times, D_hat, z_hat = dicodil(
                X, D_init, reg=reg, z_positive=True, n_iter=50, n_jobs=400,
                eps=1e-4, verbose=2)

            # Save the atoms
            prefix = (f"K{n_atoms}_L{L}_reg{reg}"
                      f"_seed{random_state}_dicodil_{size}_")
            prefix = prefix.replace(" ", "")
            np.save(f"hubble/{prefix}D_hat.npy", D_hat)
            z_hat[z_hat < 1e-2] = 0
            z_hat_save = [sparse.csr_matrix(z) for z in z_hat]
            np.save(f"hubble/{prefix}z_hat.npy", z_hat_save)

            plot_atom_and_coefs(D_hat, z_hat, prefix)
