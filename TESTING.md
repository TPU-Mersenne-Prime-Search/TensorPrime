# Unit Testing
Please create a new test every time you add a new non-trivial feature to the program!

Unit testing for this program is facilitated by the `unittest` Python package. Unit tests should be verified passing before opening a pull request. Documentation of `unittest` can be found at [this Colab link](https://colab.research.google.com/github/caam37830/book/blob/master/09_computing/unittest.ipynb).

## Usage
To run existing tests manually from the command line, change directory to the root of the git repo and run `python tests.py`. The output should look something like this:
```
$ python tests.py
.
----------------------------------------------------------------------
Ran 2 tests in 0.003s

OK
```

As shown in the example linked above, the `unittest` library allows the creation of a class which takes a `unittest.TestCase` object as an argument. Below is an example of the general structure of a test case. Functions defined within that class are tests which will run as part of that test case. To create a new test, add a class definition to `tests.py`, following the general format as seen below.

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

## CI
A GitHub action will automatically run all the tests in tests.py when either
1. A PR is opened to merge into the `master` branch
2. A push is made to the `master` branch

The outcome of these tests will be reported in "Conversation" section of the PR.
