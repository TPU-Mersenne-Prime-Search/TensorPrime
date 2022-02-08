import unittest

import numpy as np
import pandas as pd
from math import log2, floor

from prptest import probable_prime
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
    def test_prp(self):
        test_exponents = [(7, True), (13, True), (17, True), (61, True), (89, True), (6, False), (12, False), (20, False), (100, False), (300, False)]
        for i in range(len(test_exponents)):
            config.initialize_constants(test_exponents[i], 2**(floor(log2(test_exponents[i]))))
            self.assertEqual(probable_prime(test_exponents[i][0]), test_exponents[i][1])


# Add new test classes above this comment

def tests_main():
    unittest.main()


if __name__ == "__main__":
    tests_main()
