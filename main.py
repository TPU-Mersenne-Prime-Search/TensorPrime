import argparse
import sys
import time

# Benchmarking
# import . as bench
# Lucas-Lehmer test
from log_helper import init_logger
import mersenne.lucas as ll
# FFT
# import . as fft
# IBDWT
import IBDWT as ibdwt
# PrP Test
from prptest import probable_prime
from prptestGEC import probable_primeGEC
# Global variables
import config


def main():
    parser = argparse.ArgumentParser()

    # in order to add more arguments to the parser, attempt a similar declaration to below. Anthing without a dash is becomes ordinal and required
    parser.add_argument("-p", "--prime", type=int,
                        help="seed for the mersenne prime being tested", default=None)
    parser.add_argument("--ll", action="store_true")
    parser.add_argument("--fft", type=str, default=None)
    parser.add_argument("--bench", action="store_true",
                        help="perform testing etc")
    parser.add_argument("--nogec", action="store_true", default=False)
    args = vars(parser.parse_args())
    if not args["prime"]:
        raise ValueError("runtime requires a prime number for testing!")
        exit()

    # args is a dictionary in python types, in a per flag key-value mapping, which can be accessed via,
    # eg, flags["prime"], which will return the integer passed in.
    # If you want specific behavior for the options, eg prime is none, exit()""
    # Command line arguments

    # Argument option to load from file?

    # Determine which function is wanted,
    # Run relevant function
    
    m_logger = init_logger(args)   # logging functionality specific to our runtime

    if args["bench"] is not None:
        pass

    if args["fft"] is not None:
        pass

    if args["nogec"] is False:
        config.GEC_enabled = True
  
    if args["prime"] is not None:
        p = int(args["prime"])
        start_time = time.time()
        probable_primeGEC(p)
        end_time = time.time()
        print("{} tested in {} sec: {}".format(p, end_time - start_time,
                                               "probably prime!" if is_probable_prime else "composite"))

main()
