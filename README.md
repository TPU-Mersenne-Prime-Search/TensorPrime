# TensorPrime

This program was developed as part of a two-term senior capstone project at Portland State University. Our goal was to develop a program which uses the Tensor Processing Unit to accelerate the Probable Prime Test, an algorithm for testing the (probable) primality of large [Mersenne numbers](https://mathworld.wolfram.com/MersenneNumber.html).

Currently due to precision issues with the TPU the known primes we are able to check range from `2^7 - 1` to `2^4423 - 1` inclusive. Exponents `2` and `3` are not able to be checked with our program due to the Prime/composite logic checking only if the residue at the end is `9`, which is not possible for those exponents.

See our project [Wiki](https://github.com/TPU-Mersenne-Prime-Search/TensorPrime/wiki) for more details about the project.

# Quickstart
Open [Google Colab](https://github.com/TPU-Mersenne-Prime-Search/TensorPrime) and select `File -> Open Notebook` and enter `https://github.com/TPU-Mersenne-Prime-Search/TensorPrime` in the "GitHub" tab. Select `BootstrapTensorPrime.ipynb`.

Currently due to precision limitations of the `bfloat16` data type used by the TPU for matrix multiplication operations, the largest known mersenne prime this program is able to check is `2^4423 - 1`. Because of this the PrimeNet assignments this program supports will not work on the TPU architecture. To run TensorPrime on the TPU, the following steps are recommended.

1. Enter the desired branch into the form in `BootstrapTensorPrime.ipynb` and run the cell. Follow the prompts to connect to your Google Drive account. As Colab VMs do not provide persistent storage Google Drive is necessary to store persistent information about the program such as user settings.
2. Add a new code cell after Drive is mounted. The command `!python3 main.py -p <exponent> --siglen <signal length>` will run the probable prime test on the desired exponent with the specified signal length.

## Example

To check `2^4423 - 1` for primality:

```
!python3 main.py -p 4423 --siglen 1024
```
