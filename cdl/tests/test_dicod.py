import numpy as np
from cdl.multivariate_convolutional_coding_problem import\
    MultivariateConvolutionalCodingProblem
from cdl.dicod import DICOD, ALGO_GS, ALGO_RANDOM


def test_dicod_simple():
    D = np.random.normal(size=(1, 10, 2))
    z = np.zeros((1, 171))
    z[0, [30, 50, 65, 85, 100, 130]] = 1
    x = np.array([[np.convolve(zk, dk, 'full') for dk in Dk]
                  for Dk, zk in zip(D, z)]).sum(axis=0)
    pb = MultivariateConvolutionalCodingProblem(
            D, x, lmbd=0.01)

    for algo, name in [(ALGO_GS, "Gauss-Southwell"),
                       (ALGO_RANDOM, "Random")]:
        dicod = DICOD(pb, n_jobs=2, use_seg=2, algorithm=algo,
                      debug=3)
        dicod.fit(pb)

        # Assert we recover the right support
        assert np.all(pb.pt.nonzero()[1] == z.nonzero()[1]), (
            "DICOD algo {} failed to recover the right support"
            "\n{}".format(name, pb.pt.nonzero()))
