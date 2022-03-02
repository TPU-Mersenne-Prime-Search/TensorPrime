# This module is for holding the global variables that need to be accessed by
# multiple other files

import time

# Needed to determine the bit_array and weight_array
import numpy as np

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

# Constants for GEC
GEC_enabled = False
GEC_iterations = 2000000

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