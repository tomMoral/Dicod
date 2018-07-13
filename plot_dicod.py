if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser('Test for the DICOD algorithm')
    parser.add_argument('--exp', type=str, default=None,
                        metavar='DIRECTORY', help='If present, save'
                        ' the result in the given DIRECTORY')
    parser.add_argument('--jobs', action='store_true',
                        help='Compute the runtime for different number '
                             'of cores')
    parser.add_argument('--lmbd', action='store_true',
                        help='Compute the scaling relatively to lmbd.')
    parser.add_argument('--met', action='store_true',
                        help='Compute the optimization algorithms')
    args = parser.parse_args()
