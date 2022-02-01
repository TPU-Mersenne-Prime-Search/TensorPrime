# This file holds the more math-heavy code; namely, the IBDWT implementation

import config

import numpy as np
import tensorflow as tf

# Takes the number to be squaremodded (int), the exponent of the marsene prime being
# tested (int),
# the signal length of the transform (int), the bit_array corresponding to the bases used in
# the variable base representation (array of ints of size signal_length), and the weight_array
# corresponding to the weighted transform (array of ints of size signal_length). Outputs
# (num_to_square ^ 2) mod ((2 ^ exponent) - 1). 
# This is the highest-level method in this file, and should be the only one called by
# main(). prime_exponent and signal_length can be omitted if config.initialize_constants
# has already been run but should not be omitted if you're testing the ibdwt method.
# bit_array and weight_array can also be omitted if config.initialize_constants has been
# run, but they can also be omitted if you're testing the ibdwt method (it'll just impact
# performance speed). 
# Note that the only three things this method does directly (without a helper method)
# is the square operation, the rounding, and the mod operation at the end. Everything
# else is handled by subsidiary methods.
def squaremod_with_ibdwt(num_to_square, prime_exponent = None, signal_length = None, 
                         bit_array = None, weight_array = None):
    
    if prime_exponent is None:
        prime_exponent = config.exponent    
    if signal_length is None:
        signal_length = config.signal_length
    if bit_array is None:
        bit_array = config.bit_array
    if weight_array is None:
        weight_array = config.weight_array

    if (prime_exponent == None or signal_length == None or num_to_square < 0):
        return -1

    # If tensorflow doesn't find a TPU, this'll run on the CPU instead
    with tf.device('/TPU:0'):

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
        rounded_signal = np.round(squared_signal).astype(int)
        squared_num = designalize(rounded_signal, bit_array)
        if (config.prime == None):
            final_result = int(squared_num % (1 << (prime_exponent) - 1))
        else:
            final_result = int(squared_num % config.prime)

        return final_result

# Takes the marsene exponent and signal length of the transform as ints and outputs the
# corresponding bit_array. Should be called by main to store the bit_array outside
# the core loop.
def determine_bit_array(exponent = config.exponent, signal_length = config.signal_length):
    bit_array = [int(0)] * signal_length
    for i in range(1, signal_length+1):
        bit_array[i-1] = int(np.ceil((exponent * i) / signal_length) - np.ceil(exponent*(i-1) / signal_length))        
    
    return bit_array

# Takes the marsene exponent and signal length of the transform as ints and outputs the
# corresponding weight_array. Should be called by main to store the weight_array
# outside the core loop.
def determine_weight_array(exponent = config.exponent, signal_length = config.signal_length):
    weight_array = [int(0)] * signal_length
    for i in range(0, signal_length):
        weight_array[i] = 2 ** (np.ceil(exponent * i/signal_length) - (exponent * i/signal_length))
    return weight_array

# Takes a number to be turned into a signal and the bit_array corresponding to the
# bases used in the variable base representation. Returns an array of integers
# corresponding to the coeffecients of the variable base representation of num.
def signalize(num, bit_array = config.bit_array):
    if (config.two_to_the_bit_array == None):    
        signalized_num = [int(0)]*len(bit_array)
        for i in range(0, len(bit_array)):
            signalized_num[i] = num % int(1 << bit_array[i])
            num = num // int(1 << bit_array[i])
    else:
        signalized_num = [int(0)]*config.signal_length
        for i in range(0, config.signal_length):
            signalized_num[i] = num % config.two_to_the_bit_array[i]
            num = num // config.two_to_the_bit_array[i]
    
    return signalized_num

# Takes a signalized number and its corresponding bit_array and turns it back into an
# integer.
# This just inverts signalize().
def designalize(signal, bit_array = config.bit_array):
    resultant_number = 0
    if (config.base_array is None):
        base = 0
        for i in range(0,len(bit_array)):
            resultant_number += signal[i] * (1 << base)
            base += bit_array[i]
    else:
        # To-Do: rig this to the TPU
        # Except the base array can't be converted to floats, so it can't be I think  
        resultant_number = np.dot(signal, config.base_array)
    return resultant_number

# Takes an array of integers to be transformed and an array corresponding to the desired
# weighting. Outputs the weighted FFT of signal_to_transform as an array of floats.
def weighted_transform(signal_to_transform, weight_array = config.weight_array):
    weighted_signal = np.multiply(signal_to_transform, weight_array)
    transformed_weighted_signal = tf.signal.fft(weighted_signal)
    return transformed_weighted_signal

# Takes an array of integers to be inverse-transformed and an array corresponding to its
# weighting. Outputs the inverse weighted FFT of transformed_weighted_signal (that is,
# the de-weighted, de-transformed signal) as an array of floats.
# This just inverts weighted_transform().
def inverse_weighted_transform(transformed_weighted_signal, weight_array = config.weight_array):
    weighted_signal = tf.math.real(tf.signal.ifft(transformed_weighted_signal))
    if (config.inverse_weight_array == None):
        signal = tf.math.divide(weighted_signal, weight_array)
    else:
        signal = tf.math.multiply(weighted_signal, config.inverse_weight_array)

    return signal