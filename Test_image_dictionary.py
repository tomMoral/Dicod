import numpy as np
from time import time
from sys import stdout as out
from dicod.dicod2d import DICOD2D
from scipy.signal import fftconvolve
from joblib import Parallel, delayed
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt


def init_D(im, K, h_dic, w_dic, overlap=0, method='PCA'):
    chan, h_im, w_im = im.shape
    X = []
    for i in range(h_im//(h_dic-overlap)-overlap):
        for j in range(w_im//(w_dic-overlap)-overlap):
            h0 = int(i*(h_dic-overlap))
            w0 = int(j*(w_dic-overlap))
            X += [im[:, h0:h0+h_dic, w0:w0+w_dic].flatten()]
    print("Dictionary intialization with {} patches".format(len(X)))
    if method == 'PCA':
        from sklearn.decomposition import PCA
        pca = PCA(K)
        pca.fit(X)
        D = pca.components_
    else:
        from sklearn.cluster import KMeans
        km = KMeans(K, n_jobs=-1)
        km.fit(X)
        D = km.cluster_centers_
    print("End dictionary intialization")
    return D.reshape((K, chan, h_dic, w_dic))


def print_D(D):
    import matplotlib.pyplot as plt
    K, chan, h_dic, w_dic = D.shape
    w_d = int(np.sqrt(K))
    if K % w_d != 0:
        for i in range(w_d // 2):
            if K % int(w_d-i) == 0:
                break
        if i == w_d // 3 - 1:
            i = -1
        w_d -= i
    h_d = int(np.ceil(K / w_d))
    imD = np.empty((chan, h_d*h_dic, w_d*w_dic))
    for h in range(h_d):
        for w in range(w_d):
            imD[:, h*h_dic:(h+1)*h_dic, w*w_dic:(w+1)*w_dic] = D[h*w_d+w]
            brk = h*w_d+w+1 >= K
            if brk:
                break
        if brk:
            break
    plt.imshow(((imD.swapaxes(0, 2)+.5)*255).astype(np.uint8),
               interpolation='none')


def conv_DU(A_kk, d_k):
    return [fftconvolve(A_kk, d_kd, 'valid') for d_kd in d_k]


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser('Test dictionary learning for images')
    parser.add_argument('--display', action='store_true',
                        help='if set, display the resulting image')
    parser.add_argument('--ip', action='store_true',
                        help='if set, open an ipython terminal after')
    parser.add_argument('--hostfile', type=str, default=None,
                        help='set hostfile for MPI')
    args = parser.parse_args()

    # Construct the problem
    K = 64
    sigma = 10
    w_dic = 12
    h_dic = 12
    try:
        I = plt.imread("../datasets/images/standard_images/mandril_color.tif")
        # I = plt.imread("../datasets/images/standard_images/palace.png")
    except FileNotFoundError:
        I = plt.imread("../data/images/standard_images/mandril_color.tif")

    im0 = I.swapaxes(0, 2)
    m, M, tp = im0.min(), im0.max(), im0.dtype
    noise = np.random.normal(size=im0.shape)*sigma/255
    im = im0 + noise
    im = 2*((im - m)/(M - m)) - 1
    # assert -1 <= im.min() < im.max() <= 1

    D = init_D(im, K, h_dic, w_dic, overlap=0, method='KM')
    from dicod.multivariate_convolutional_coding_problem_2d import \
        MultivariateConvolutionalCodingProblem2D as mccp
    pb = mccp(D, im, lmbd=.001)
    nD = np.sqrt(np.mean(pb.D*pb.D, axis=1).sum(axis=-1).sum(axis=-1))
    rm_D = set()
    for k in range(K):
        for k0 in range(k+1, K):
            if (abs(pb.DD[k, k0]/(nD[k]*nD[k0])).max() >= .99 and
                    k not in rm_D):
                rm_D.add(k0)
    rm_D = list(rm_D)
    rm_D.sort(reverse=True)
    D = list(D)
    for i in rm_D:
        del D[i]
    print("Number of dictionary : {}".format(len(D)))
    D = np.array(D)
    pb = mccp(D, im, lmbd=.001)

    # Intiate the algorithm
    w_world = 6
    n_jobs = 36  # *w_world
    dcp = DICOD2D(debug=5, n_jobs=n_jobs, w_world=w_world, tol=1e-2, use_seg=4,
                  timeout=90, max_iter=n_jobs*1e7, hostfile=args.hostfile,
                  logging=True)

    # dcp.fit(pb)
    # print("cost: ", dcp.cost)

    # Dicitonary learning tryout
    ########################################
    for it in range(5):
        dcp.fit(pb)
        Dp = np.copy(D)
        out.flush()
        print('Iter {}: Cost={:.4f}; set_dict={}; nnzero={}'
              ''.format(it, dcp.cost, set(pb.pt.nonzero()[0]),
                        len(pb.pt.nonzero()[0])/(K*dcp.L)))
        out.flush()
        with Parallel(n_jobs=-1) as para:
            cdu = delayed(conv_DU)
            t = time()
            for j in range(10):
                dD = 0
                for k in range(len(D)):

                    dA = para(cdu(A_kk, d_k)
                              for d_k, A_kk in zip(pb.D, dcp.A[k]))
                    dA = np.sum(dA, axis=0)
                    nA = np.sqrt((dcp.A[k, k]*dcp.A[k, k]).sum())
                    nA += nA == 0
                    U = (dcp.B[k]-dA)/nA+pb.D[k]
                    nU = np.mean(U*U, axis=0).sum(axis=-1).sum(axis=-1)
                    U /= nU
                    dD += abs(D[k] - U).sum()
                    D[k] = U
                print(dD)
                if dD < 1e-4:
                    break
        dcp.tol *= .9
        print("{}: Update Dictionary in {:.2f}s - end dD: {:.2f}"
              "".format(j, time()-t, dD))
        print('Ok:', abs(D).max())
        print("tol: ", dcp.tol)
        out.flush()

        if dcp.cost > 1e6:
            break
    ###########################################

    # import IPython
    # IPython.embed()
    # import sys
    # sys.exit(0)

    def show_rec(pt=pb.pt, sup=None):
        if sup is not None:
            pt = pb.pt*(abs(pb.pt) < sup)
        imr = np.clip((pb.reconstruct(pt)+1)/2*(M-m)+m, m, M).astype(tp)
        plt.imshow(imr.swapaxes(0, 2))
        plt.axis('off')
        ll = np.linspace(-.5, dcp.w_cod-.5, w_world+1)
        plt.hlines(ll, -.5, dcp.h_cod-.5)
        h_world = n_jobs // w_world
        ll = np.linspace(-.5, dcp.h_cod-.5, h_world+1)
        plt.vlines(ll, -.5, dcp.w_cod-.5)
        plt.subplots_adjust(0, 0, 1, 1)
        # plt.savefig('../../output.pdf')
        # plt.close()
        return imr

    if args.display:
        try:
            imr = show_rec()
            res = (imr-im0)*255
            MSRE = np.sqrt(np.mean(res * res))
            PSNR = 20*np.log(255 / MSRE)
            print("MSRE: {:.3f}; PSNR: {:.3f}".format(MSRE, PSNR))

        except Exception as e:
            print(e.__traceback__)
            import IPython
            IPython.embed()
    if args.ip:
        import IPython
        IPython.embed()
