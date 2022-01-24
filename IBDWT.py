# This file holds the more math-heavy code; namely, the IBDWT implementation

import numpy as np

# Takes the number to be squaremodded (int), the exponent of the marsene prime being
# tested (int),
# the signal length of the transform (int), the bit_array corresponding to the bases used in
# the variable base representation (array of ints of size signal_length), and the weight_array corresponding to the weighted
# transform (array of ints of size signal_length). Outputs (num_to_square ^ 2) mod (2 ^ exponent) - 1.
# This is the highest-level method in this file, and is the primary one to be called
# by main. The intended use within the core loop is to pass in pre-calculated
# bit_array and weight_array (for optimization purposes), but they can be omitted if
# it's just being called for testing purposes.
# Note that the only three things this method does directly (without a helper method)
# is the square operation, the rounding, and the mod operation at the end. Everything
# else is handled by subsidiary methods.
def squaremod_with_ibdwt(num_to_square, prime_exponent, signal_length, bit_array = None, weight_array = None):
    # Checks if bit_array or weight_array need to be made, and makes them if so
    if (bit_array == None):
        bit_array = determine_bit_array(prime_exponent, signal_length)
    if (weight_array == None):
        weight_array = determine_weight_array(prime_exponent, signal_length)

    # These do the pre-square prep
    signal_to_square = signalize(num_to_square, bit_array)
    transformed_signal = weighted_transform(signal_to_square, weight_array)
    
    # This is the squaring
    squared_transformed_signal = transformed_signal * transformed_signal
    
    # These do the post-square processing
    squared_signal = inverse_weighted_transform(squared_transformed_signal, weight_array)
    rounded_signal = np.round(squared_signal)
    squared_num = designalize(rounded_signal, bit_array)
    final_result = int(squared_num % (2**prime_exponent - 1))

    return final_result

# Takes the marsene exponent and signal length of the transform as ints and outputs the
# corresponding bit_array. Should be called by main to store the bit_array outside
# the core loop.
def determine_bit_array(exponent, signal_length):
    bit_array = [int(0)] * signal_length
    for i in range(1, signal_length+1):
        bit_array[i-1] = int(np.ceil((exponent * i) / signal_length) - np.ceil(exponent*(i-1) / signal_length))
    return bit_array

# Takes the marsene exponent and signal length of the transform as ints and outputs the
# corresponding weight_array. Should be called by main to store the weight_array
# outside the core loop.
def determine_weight_array(exponent, signal_length):
    weight_array = [int(0)] * signal_length
    for i in range(0, signal_length):
        weight_array[i] = 2**(np.ceil(exponent * i/signal_length) - (exponent * i/signal_length))
    return weight_array

# Takes a number to be turned into a signal and the bit_array corresponding to the
# bases used in the variable base representation. Returns an array of integers
# corresponding to the coeffecients of the variable base representation of num.
def signalize(num, bit_array):
    signalized_num = [int(0)]*len(bit_array)
    for i in range(0, len(bit_array)):
        signalized_num[i] = num % int(2**bit_array[i])
        num = num // int(2**bit_array[i])
    return signalized_num

# Takes an array of integers to be transformed and an array corresponding to the desired
# weighting. Outputs the weighted FFT of signal_to_transform as an array of floats.
def weighted_transform(signal_to_transform, weight_array):
    weighted_signal = np.multiply(signal_to_transform, weight_array)
    transformed_weighted_signal = np.fft.fft(weighted_signal)
    return transformed_weighted_signal

# Takes an array of integers to be inverse-transformed and an array corresponding to its
# weighting. Outputs the inverse weighted FFT of transformed_weighted_signal (that is,
# the de-weighted, de-transformed signal) as an array of floats.
# This just inverts weighted_transform().
def inverse_weighted_transform(transformed_weighted_signal, weight_array):
    weighted_signal = np.real(np.fft.ifft(transformed_weighted_signal))
    signal = np.divide(weighted_signal, weight_array)
    return signal

# Takes a signalized number and its corresponding bit_array and turns it back into an
# integer.
# This just inverts signalize().
def designalize(signal, bit_array):
    resultant_number = 0
    base = 0
    for i in range(0,len(bit_array)):
        resultant_number += signal[i] * (2**base)
        base += bit_array[i]
    return resultant_number