import unittest

import numpy as np
import pandas as pd

from prptest import probable_prime

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
        known_powers = [7, 13, 17, 61, 89]
        known_composite = [6, 12, 20, 100, 300]
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
