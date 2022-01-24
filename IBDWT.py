#This file holds the more math-heavy code; for now, just the IBDWT

from signal import signal
import numpy as np

import math

def square_by_ibdwt(num_to_square, exponent, signal_length, bit_array = None, weight_array = None):
    if (bit_array == None):
        bit_array = determine_bit_array(exponent, signal_length)
    if (weight_array == None):
        weight_array = determine_weight_array(exponent, signal_length)

    signal_to_square = signalize(num_to_square, bit_array)
    transformed_signal = weighted_transform(signal_to_square, weight_array)
    
    squared_transformed_signal = transformed_signal * transformed_signal
    
    squared_signal = inverse_weighted_transform(squared_transformed_signal, weight_array)
    squared_signal = np.round(squared_signal)

    squared_num = designalize(squared_signal, bit_array)
    return squared_num

def determine_bit_array(exponent, signal_length):
    bit_array = [int(0)] * signal_length
    for i in range(1, signal_length+1):
        bit_array[i-1] = int(np.ceil((exponent * i) / signal_length) - np.ceil(exponent*(i-1) / signal_length))
    return bit_array

def determine_weight_array(exponent, signal_length):
    weight_array = [int(0)] * signal_length
    for i in range(0, signal_length):
        weight_array[i] = 2**(np.ceil(exponent * i/signal_length) - (exponent * i/signal_length))
    return weight_array

def signalize(num_to_square, bit_array):
    signalized_num = [0]*len(bit_array)
    for i in range(0, len(bit_array)):
        signalized_num[i] = num_to_square % int(2**bit_array[i])
        num_to_square = num_to_square // int(2**bit_array[i])
    return signalized_num

def weighted_transform(signal_to_transform, weight_array):
    weighted_signal = np.multiply(signal_to_transform, weight_array)
    transformed_weighted_signal = np.fft.fft(weighted_signal)
    return transformed_weighted_signal

def inverse_weighted_transform(transformed_weighted_signal, weight_array):
    weighted_signal = np.real(np.fft.ifft(transformed_weighted_signal))
    signal = np.divide(weighted_signal, weight_array)
    return signal

def designalize(signal, bit_array):
    resultant_number = 0
    base = 0
    for i in range(0,len(bit_array)):
        resultant_number += signal[i] * (2**base)
        base += bit_array[i]
    return resultant_number