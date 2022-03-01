# This module is for holding the global variables that need to be accessed by
# multiple other files

import time

# Needed to determine the bit_array and weight_array
import numpy as np
import tensorflow as tf

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

    # Check time of IBDWT array initialization
    array_start = time.time()
    
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

    # Check time after array initialization
    array_end = time.time()
    # Get total array initialization time
    array_time = array_end - array_start
    print("IBDWT arrays initialized in ", array_time, " seconds.")

    # Wrapping this in a try block lets it work on systems without a TPU; should make
    # it easier to test locally, but might not be wanted
    try:
        print("Configuring TPU...")
        config_start = time.time()
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        config_end = time.time()
        config_time = config_end - config_start
        print("TPU configuration complete.")
        print("Total time: ", config_time)
        print()
    except ValueError:
        pass
        
settings = None
        
# Returns settings in the form of a dictionary
def getSettings():
    file = open("settings.txt", "r")
    global settings

    # Dictionary contains all settings as key value pairs
    settings = {}
    lines = file.readlines()

    for l in lines:
        # Only process arguments
        if l[0] == "-":
            # Reform
            cuts = l.partition(":")
            idv = cuts[0][1:]
            value = cuts[2].strip()
            
            # Convert values
            if value == "T":
                value = True
            elif value == "F":
                value = False
            elif value[len(value)-1] == "i":
                value = int(value[:len(value)-1])
            
            '''
            else:
                print("odd value: ", value)
            '''
            
            # Add to dictionary
            settings.update({idv: value})

    file.close()
    # Dictionary to variables?
    # return settings
