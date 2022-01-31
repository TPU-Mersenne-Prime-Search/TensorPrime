# This module is for holding the global variables that need to be accessed by
# multiple other files

# Needed to determine the bit_array and weight_array
import IBDWT as ibdwt

# These will be provided by main() when it calls initialize_constants()
exponent = None
signal_length = None

# These will be calculated by initialize_constants() based on the above 
prime = None
bit_array = None
two_to_the_bit_array = None
base_array = None
weight_array = None
inverse_weight_array = None

def initialize_constants(prime_exponent, sig_length):
    global exponent
    global signal_length
    global prime
    global bit_array
    global two_to_the_bit_array
    global base_array
    global weight_array
    global inverse_weight_array

    exponent = prime_exponent
    signal_length = sig_length
    
    prime = 2 ** exponent - 1
    
    bit_array = ibdwt.determine_bit_array(exponent, signal_length)
    two_to_the_bit_array = [int(0)] * signal_length
    for i in range(0, signal_length):
        two_to_the_bit_array[i] = 2**bit_array[i]
    
    base = 0
    base_array = [int(0)] * signal_length
    for i in range(0, signal_length):
        base_array[i] = 2**base
        base += bit_array[i]
    
    weight_array = ibdwt.determine_weight_array(exponent, signal_length)
    inverse_weight_array = [0.0] * signal_length
    for i in range(0, signal_length):
        inverse_weight_array[i] = 1 / weight_array[i]