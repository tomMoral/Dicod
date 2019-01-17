import numpy as np
import matplotlib.pyplot as plt

from .csc import reconstruct


def plot_atom_and_coefs(D_hat, z_hat, prefix):
    n_atoms = D_hat.shape[0]

    E = np.sum(z_hat > 0, axis=(1, 2))
    i0 = E.argsort()[::-1]

    n_cols = 5
    n_rows = int(np.ceil(n_atoms / n_cols))
    fig = plt.figure(figsize=(3*n_cols + 2, 3*n_rows + 2))
    for i in range(n_rows):
        for j in range(n_cols):
            if n_cols * i + j >= n_atoms:
                continue
            k = i0[n_cols * i + j]
            ax = plt.subplot2grid((n_rows, n_cols), (i, j))
            scale = 1 / D_hat[k].max() * .99
            Dk = np.clip(scale * D_hat[k].swapaxes(0, 2), 0, 1)
            ax.imshow(Dk)
            ax.axis('off')
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                        wspace=.1, hspace=.1)

    fig.savefig(f"hubble/{prefix}dict.pdf", dpi=300)

    fig = plt.figure()
    plt.imshow(z_hat.sum(axis=0).T > 0, cmap='gray')
    plt.axis('off')
    fig.tight_layout()
    fig.savefig(f"hubble/{prefix}z_hat.pdf")

    fig = plt.figure()
    X_hat = np.clip(reconstruct(z_hat, D_hat), 0, 1)
    plt.imshow(X_hat.swapaxes(0, 2))
    plt.axis('off')
    fig.tight_layout()
    fig.savefig(f"hubble/{prefix}X_hat.pdf")
