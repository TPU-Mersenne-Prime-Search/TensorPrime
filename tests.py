import unittest

import numpy as np
import pandas as pd

from mersenne.lucas import naive_lucas_lehmer
from prptest import probable_prime

# Demo class to showcase syntax for writing test cases


class TestNumpyFunctionality(unittest.TestCase):

    # Demo test to check that Numpy's dot product works as expected
    def test_dot_prod(self):
        x = np.array([2, 3])
        y = np.array([3, 2])
        self.assertEqual(12, np.dot(x, y))


# Test known mersenne prime numbers to verify algorithms are working correctly
class TestKnownPrimes(unittest.TestCase):

    # Naive, CPU-based Lucas-Lehmer
    def test_first_10_naive(self):
        known_primes = pd.read_csv("known_primes.csv", sep=",")
        for index, row in known_primes[:10].iterrows():
            p = int(row['exponent'])
            print("checking {}".format(p), end="\r")
            self.assertTrue(naive_lucas_lehmer(p))


class TestProbablePrimes(unittest.TestCase):

    # Test for probable prime test correctness
    def test_prp(self):
        known_powers = [7, 13, 17, 61, 89]
        known_composite = [6, 12, 20, 100, ]
        for i in range(5):
            prime = known_powers[i]
            comp = known_composite[i]
            self.assertTrue(probable_prime(prime))
            self.assertFalse(probable_prime(comp))


# Add new test classes above this comment

def tests_main():
    unittest.main()


if __name__ == "__main__":
    tests_main()
