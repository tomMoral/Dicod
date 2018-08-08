
def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true",
                     help="run slow tests.")
    parser.addoption("--max_workers", type=int, default=4,
                     help="maximal number of worker that can be used in the "
                     "tests.")
    parser.addoption("--deadlock_timeout", type=int, default=120,
                     help="dump traceback and exit after this delay.")
