import numpy as np


from toolboxTom.optim.paralel_solver import ParalelSolver
from toolboxTom.logger import Logger
log = Logger('ConvolutionalDicitonaryLearning', 10)

from .convolutionalPursuit import ConvolutionalPursuit
from convDL.multivariate_convolutional_coding_problem \
    import MultivariateConvolutionalCodingProblem


class OnlineDicitonaryLearning(object):
    """Online dictionary learning"""
    def __init__(self, n_dict=15, K=100, optim=ConvolutionalPursuit,
                 batch_size=1, graph=None, debug=0, **kwargs):
        super(OnlineDicitonaryLearning, self).__init__()
        self.n_dict = n_dict
        self.K = K
        self.optim = optim
        self.batch_size = batch_size
        self.graph = graph
        self.kwargs = kwargs
        if debug:
            debug -= 1
            log.set_level(10)
        self.debug = debug

    def fit(self, X, D0=None, lmbd=1, **kwargs):
        if X[0].ndim == 1:
            X = [x[np.newaxis] for x in X]

        self.n_train = len(X)
        self.n_dim = len(X[0])

        if D0 is None:
            D0 = np.random.random((self.n_dict, self.n_dim, self.K))
        self.D = np.zeros(D0.shape)

        self.problems = []
        for i, x in enumerate(X):
            pb = MultivariateConvolutionalCodingProblem(
                D=self.D, x=x, lmbd=lmbd)
            self.problems += [pb]
        self.problems = np.array(self.problems)

        self.update_D(dD=D0)

        self.kwargs.update(**kwargs)
        solver = ParalelSolver(optim=self.optim, name='SparseCoder',
                               debug=self.debug, **self.kwargs)

        n_iter = 100
        Cp = 1e30
        compt_up = 0
        self.pZ = [None]*len(self.problems)
        for i in range(n_iter):
            log.progress(levl=10, iteration=i, i_max=n_iter)
            i0 = np.random.randint(0, len(self.problems),
                                   self.batch_size)
            self.Z = solver.solve(self.problems[i0])
            for ii, z in zip(i0, self.Z):
                self.pZ[ii] = z
            C = np.sum(solver.scores)
            if self.graph is not None:
                log.graphical_cost(name=self.graph, curve='dictionary',
                                   cost=C)
            if C >= Cp:
                compt_up += 1
            else:
                compt_up = 0
                Cp = C
            if compt_up > 20:
                break
            grad_D = np.zeros(self.D.shape)
            for pb, z in zip(self.problems, self.pZ):
                if z is not None:
                    grad_D += pb.grad_D(z)/z.shape[1]
            self.update_D(-1e0*grad_D)

        self.Z = solver.solve(self.problems)

    def update_D(self, dD):
        D, DD = None, None
        for pb in self.problems:
            D, DD = pb.update_D(dD, D=D, DD=DD)
        self.D = D

    def cost(self):
        cost = 0
        for pb, z in zip(self.problems, self.Z):
            cost += pb.cost(z)
        return cost
