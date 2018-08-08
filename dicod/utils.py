import numpy as np
from collections import namedtuple


CostCurve = namedtuple('CostCurve', ['iterations', 'times', 'pobj'])


def get_log_rate(lr):
    """Return a function which define the next iteration to log.

    Parameter
    ---------
    lr : string or numeric
        String representing the log rate. Should be in:
        {'log[0-9]+', 'lin[0-9]+', 'none'} or a numerical value.
    """
    if lr[:3] == 'log':
        inc = float(lr[3:]) if len(lr[3:]) > 0 else 2
        log_rate = (lambda s: lambda t: s*t + (t == 0))(inc)
    elif lr[:3] == 'lin':
        inc = int(lr[3:]) if len(lr[3:]) > 0 else 10
        log_rate = (lambda s: lambda t: t+s)(inc)
    elif lr == 'none':
        def log_rate():
            return 1e300
    elif type(lr) is int or type(lr) is float:
        inc = log_rate
        log_rate = (lambda s: lambda t: s*t + (t == 0))(inc)
    else:
        assert False, "{} is not a log rate".format(lr)

    return log_rate


def l1(x):
    return np.sum(np.abs(x))


def l2(x):
    return x.ravel().dot(x.ravel())
