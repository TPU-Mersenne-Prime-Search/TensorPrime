import sys
import time as time

# Benchmarking
# import . as bench
# Lucas-Lehmer test
import mersenne.lucas as ll
# FFT
# import . as fft
# IBDWT
import IBDWT as ibdwt
# Global variables
import config

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
    
    if args[0] == "-ibdwt":
        config.initialize_constants(int(args[1]), int(args[2]))
        print(config.exponent)
        print(ibdwt.squaremod_with_ibdwt(int(args[3])))


def probable_prime(power):
    s = 3
    for i in range(power):
        s *= s
        s = s % ((1 << power) - 1)
    if s == 9:
        return True
    return False


main()
