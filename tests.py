import unittest

import numpy as np
import pandas as pd
from math import log2, floor

from prptest import probable_prime
import IBDWT
import config

# Demo class to showcase syntax for writing test cases


class TestNumpyFunctionality(unittest.TestCase):

    # Demo test to check that Numpy's dot product works as expected
    def test_dot_prod(self):
        x = np.array([2, 3])
        y = np.array([3, 2])
        self.assertEqual(12, np.dot(x, y))


class TestProbablePrimes(unittest.TestCase):

    # Test for probable prime test correctness
    def test_prp_primes(self):
        test_exponents = [7, 13, 17, 61, 89]
        for i in range(len(test_exponents)):
            config.initialize_constants(
                test_exponents[i], 2**(floor(log2(test_exponents[i]))))
            self.assertEqual(probable_prime(test_exponents[i]), True)

    def test_prp_composites(self):
        test_exponents = [6, 12, 20, 100, 300]
        for i in range(len(test_exponents)):
            config.initialize_constants(
                test_exponents[i], 2**(floor(log2(test_exponents[i]))))
            self.assertEqual(probable_prime(test_exponents[i]), False)


class TestIBDWT(unittest.TestCase):

    # Test the implementation of the IBDWT for correctness
    def test_ibdwt(self):
        num_to_square = 78314567209
        test_exponent = 37
        config.initialize_constants(
            test_exponent, 2**(floor(log2(test_exponent))))
        self.assertEqual(IBDWT.squaremod_with_ibdwt(
            num_to_square, test_exponent), 58368107274)


# Add new test classes above this comment

def tests_main():
    unittest.main()


if __name__ == "__main__":
    tests_main()
