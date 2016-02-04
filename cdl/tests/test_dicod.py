import numpy as np
import pytest


from cdl.multivariate_convolutional_coding_problem import\
    MultivariateConvolutionalCodingProblem
from cdl.dicod import DICOD, ALGO_GS, ALGO_RANDOM
from cdl.multivariate_convolutional_coding_problem_2d import \
    MultivariateConvolutionalCodingProblem2D
from cdl.dicod2d import DICOD2D
from scipy.signal import fftconvolve


from faulthandler import dump_traceback_later
from faulthandler import cancel_dump_traceback_later


@pytest.yield_fixture
def exit_on_deadlock():
    dump_traceback_later(timeout=3, exit=True)
    yield
    cancel_dump_traceback_later()

slow = pytest.mark.skipif(
    not pytest.config.getoption("--runslow"),
    reason="need --runslow option to run"
)

ids = ["GaussS, n_jobs=1, seg=1",
       "GaussS, n_jobs=2, seg=1",
       "GaussS, n_jobs=3, seg=1",
       "GaussS, n_jobs=4, seg=1",
       "GaussS, n_jobs=2, seg=2",
       "GaussS, n_jobs=2, seg=4",
       "GaussS, n_jobs=2, seg=8",
       "Random, n_jobs=1, seg=1",
       "Random, n_jobs=2, seg=1",
       "Random, n_jobs=3, seg=1",
       # "Random, n_jobs=2, seg=2",
       # "Random, n_jobs=2, seg=4",
       # "Random, n_jobs=2, seg=8"
       ]
param_array = [
    (ALGO_GS, 1, 1),
    (ALGO_GS, 2, 1),
    (ALGO_GS, 3, 1),
    (ALGO_GS, 4, 1),
    (ALGO_GS, 2, 2),
    (ALGO_GS, 2, 4),
    (ALGO_GS, 2, 8),
    (ALGO_RANDOM, 1, 1),
    (ALGO_RANDOM, 2, 1),
    (ALGO_RANDOM, 3, 1),
    # (ALGO_RANDOM, 2, 2),
    # (ALGO_RANDOM, 2, 4),
    # (ALGO_RANDOM, 2, 8),
]


@pytest.mark.parametrize("algo,n_jobs,n_seg", param_array, ids=ids)
def test_dicod_simple(exit_on_deadlock, algo, n_jobs, n_seg):
    K = 3
    D = np.random.normal(size=(K, 2, 5))
    D /= np.sqrt((D*D).sum(axis=-1))[:, :, None]
    z = np.zeros((K, 100))
    z[0, [0, 12, 23, 30, 42, 50, 65, 85, 95]] = 1
    z[1, 67] = 2
    x = np.array([[np.convolve(zk, dk, 'full') for dk in Dk]
                  for Dk, zk in zip(D, z)]).sum(axis=0)
    pb = MultivariateConvolutionalCodingProblem(
            D, x, lmbd=0.002)

    dicod = DICOD(pb, n_jobs=n_jobs, use_seg=n_seg,
                  algorithm=algo, debug=5, patience=1000)
    dicod.fit(pb)

    pt = pb.pt*(abs(pb.pt) > pb.lmbd)

    # Assert we recover the right support
    print(pb.pt.reshape(1, -1).nonzero()[1], '\n',
          pt.reshape(1, -1).nonzero()[1], '\n',
          z.reshape(1, -1).nonzero()[1])
    assert (np.all(pt.reshape(1, -1).nonzero()[1] ==
                   z.reshape(1, -1).nonzero()[1]) or
            pb.cost(z) >= dicod.cost), (
        "Cost pt: ", dicod.cost, "Cost z: ", pb.cost(z))
    assert abs(pb.cost(pb.pt) - dicod.cost)/dicod.cost < 1e-6


@pytest.mark.parametrize("algo,n_jobs,n_seg", param_array, ids=ids)
def test_dicod_simple_2d(algo, n_jobs, n_seg):
    K = 3
    D = np.random.normal(size=(K, 2, 1, 5))
    D /= np.sqrt((D*D).sum(axis=-1))[:, :, :, None]
    z = np.zeros((K, 1, 100))
    z[0, 0, [0, 12, 23, 30, 42, 50, 65, 85, 95]] = 1
    z[1, 0, 67] = 2
    x = np.array([[fftconvolve(zk, dk, 'full') for dk in Dk]
                  for Dk, zk in zip(D, z)]).sum(axis=0)
    pb = MultivariateConvolutionalCodingProblem2D(
            D, x, lmbd=0.002)

    dicod = DICOD2D(pb, n_jobs=n_jobs, w_world=n_jobs, use_seg=n_seg,
                    algorithm=algo, debug=5, patience=1000)
    dicod.fit(pb)

    pt = pb.pt*(abs(pb.pt) > pb.lmbd)

    # Assert we recover the right support
    print(pb.pt.reshape(1, -1).nonzero()[1], '\n',
          z.reshape(1, -1).nonzero()[1])
    assert (np.all(pt.reshape(1, -1).nonzero()[1] ==
                   z.reshape(1, -1).nonzero()[1]) or
            pb.cost(z) >= dicod.cost), (
        "Cost pt: ", dicod.cost, "Cost z: ", pb.cost(z))
    assert abs(pb.cost(pb.pt) - dicod.cost)/dicod.cost < 1e-6


param_corner = [
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1),
    (-5, 7),
    (-20, 0),
]


@slow
@pytest.mark.parametrize("h_pad, w_pad", param_corner)
def test_dicod_2d_corner(h_pad, w_pad):
    dim = 2
    K = 3
    h_dic = 10
    w_dic = 10
    h_cod = 100
    w_cod = 100
    n_jobs = 4
    w_world = 2

    D = np.random.normal(size=(K, dim, h_dic, w_dic))
    D /= np.sqrt((D*D).sum(axis=-1).sum(axis=-1))[:, :, None, None]
    z = np.zeros((K, h_cod, w_cod))
    x = np.array([[fftconvolve(zk, dk, 'full') for dk in Dk]
                  for Dk, zk in zip(D, z)]).sum(axis=0)
    pb = MultivariateConvolutionalCodingProblem2D(
            D, x, lmbd=0.002)
    dicod = DICOD2D(pb, n_jobs=n_jobs, w_world=w_world, use_seg=1,
                    algorithm=ALGO_GS, debug=5, patience=200,
                    i_max=1e7, t_max=15)
    for _ in range(3):
        D = np.random.normal(size=(K, dim, h_dic, w_dic))
        z = np.zeros((K, 1, h_cod, w_cod))
        z[0, 0, h_cod//2+h_pad, w_cod//2+w_pad] = 1
        x = np.array([[fftconvolve(zk[0], dk, 'full') for dk in Dk]
                      for Dk, zk in zip(D, z)]).sum(axis=0)
        pb = MultivariateConvolutionalCodingProblem2D(
                D, x, lmbd=0.002)
        dicod.fit(pb)

        pt = pb.pt*(abs(pb.pt) > pb.lmbd)

        print(pb.pt.reshape(1, -1).nonzero()[1], '\n',
              pt.reshape(1, -1).nonzero()[1], '\n',
              z.reshape(1, -1).nonzero()[1])
        assert np.all(pt.reshape(1, -1).nonzero()[1] ==
                      z.reshape(1, -1).nonzero()[1]), (
            "Cost pt: ", pb.cost(pb.pt), "Cost z: ", pb.cost(z))
    assert abs(pb.cost(pb.pt) - dicod.cost)/dicod.cost < 1e-6
