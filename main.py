import sys
import argparse
import time

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
    parser = argparse.ArgumentParser()

    #in order to add more arguments to the parser, attempt a similar declaration to below. Anthing without a dash is becomes ordinal and required
    parser.add_argument("-p","--prime",type = int, help="seed for the mersenne prime being tested", default = None)
    parser.add_argument("--ll", action = "store_true")
    parser.add_argument("--fft",type = str, default = None)
    parser.add_argument("--bench",action = "store_true", help = "perform testing etc")
    args = vars(parser.parse_args())
    if not args["prime"]:
	      raise ValueError("runtime requires a prime number for testing!")
	      exit()

    #args is a dictionary in python types, in a per flag key-value mapping, which can be accessed via, 
    #eg, flags["prime"], which will return the integer passed in.
    #If you want specific behavior for the options, eg prime is none, exit()""
    # Command line arguments

    # Argument option to load from file?

    # Determine which function is wanted,
    # Run relevant function
    
    if args["bench"] is not None:
        pass
    
    if args["fft"] is not None:
        pass

    if args["prime"] is not None:
        p = int(args["prime"])
        start_time = time.time()
        is_probable_prime = probable_prime(p)
        end_time = time.time()
        print("{} tested in {} sec: {}".format(p, end_time - start_time,
                                               "probably prime!" if is_probable_prime else "composite"))
    
    if args[0] == "-ibdwt":
        config.initialize_constants(int(args[1]), int(args[2]))
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
