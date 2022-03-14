# TensorPrime

This program was developed as part of a two-term senior capstone project at Portland State University. Our goal was to develop a program which uses the Tensor Processing Unit to accelerate the Probable Prime Test, an algorithm for testing the (probable) primality of large [Mersenne numbers](https://mathworld.wolfram.com/MersenneNumber.html).

See our project [Wiki](https://github.com/TPU-Mersenne-Prime-Search/TensorPrime/wiki) for more details about the project.

# Quickstart
Open [Google Colab](https://github.com/TPU-Mersenne-Prime-Search/TensorPrime) and select `File -> Open Notebook` and enter `https://github.com/TPU-Mersenne-Prime-Search/TensorPrime` in the "GitHub" tab. Select `BootstrapTensorPrime.ipynb`.

Currently due to precision limitations of the `bfloat16` data type used by the TPU for matrix multiplication operations, the highest exponent this
