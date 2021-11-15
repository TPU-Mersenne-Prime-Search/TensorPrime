# Unit Testing
Unit testing for this program is facilitated by the `unittest` Python package. Unit tests should be run and verified passing before opening a pull request (setting this up as a GitHub action is a work in progress). An example of how this works in Colab can be found at [this example](https://colab.research.google.com/github/caam37830/book/blob/master/09_computing/unittest.ipynb).

## Usage
Please create a new test every time you add a new non-trivial feature to the program. As shown in the example linked above, the `unittest` library allows the creation of a class which takes a `unittest.TestCase` object as an argument. Below is an example of the general structure of a test case.

```
import unittest

class TestSomeFeature(unittest.TestCase):

    # This test will fail if the inputs to assertEqual() are not equal
    def test_equality(self):
        a = 2 + 2
        b = 4
        self.assertEqual(a, b)
    
    # This test will fail if the input to assertTrue() is not true
    def test_boolean(self):
        some_value = True
        self.assertTrue(some_value)
```

To run the tests above, one can call `unittest.main()`