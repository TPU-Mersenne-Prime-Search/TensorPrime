import sys

# Benchmarking
# import . as bench
# Lucas-Lehmer test
import mersenne.lucas as ll
# FFT
# import . as fft
from PRPTest import *


def main():
    # Command line arguments
    args = sys.argv[1:]

    # Argument option to load from file?

    # Determine which function is wanted,
    # Run relevant function

    if args[0] == "-bench":
        pass

    if args[0] == "-fft":
        pass

    if args[0] == "-ll":
        # Type check passed arguments
        inval = int(args[1])
        outval = ll.naive_lucas_lehmer(inval)
        print(outval)

    if args[0] == "-prp":
        p = int(args[1])
        start_time = time.time()
        is_probable_prime = probable_prime(p)
        end_time = time.time()
        print("{} tested in {} sec: {}".format(p, end_time - start_time,
                                               "probably prime!" if is_probable_prime else "composite"))


main()
