import unittest

import numpy as np

# Demo class to showcase syntax for writing test cases
class TestNumpyFunctionality(unittest.TestCase):
    
    # Demo test to check that Numpy's dot product works as expected
    def test_dot_prod(self):
        x = np.array([2, 3])
        y = np.array([3, 2])
        self.assertEqual(12, np.dot(x, y))


if __name__ == "__main__":
    unittest.main()