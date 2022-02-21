import config
import numpy as np
import os

path = "saves/"
# BootstrapTensorPrime

# Saves data required to re-initialize the search
def save(signal, iteration):

    # Get highest save
    maxsaves = config.settings["SaveCount"]
    for i in range(maxsaves):
        if os.path.exists(path + "save" + i + ".txt"):
            continue
        # Remove oldest if maximum
        
        
        # Increment saves
        for x in range(i):
            setval = i - x
            os.rename(path + "save" + str(setval - 1) + ".txt", 
                    path + "save" + str(setval) + ".txt")
        break
    
    # Pack data
    
    # Exponent, Signal length are required to re-build
    config.exponent
    config.signal_length
    
    # signal and iteration are required to resume within a prime check
    signal
    iteration
    
    packed = [iteration, config.exponent, config.signal_length]
    
    # Save data
    #np.save("save0", packed, allow_pickle=False)
    np.savez("save0", [config.exponent, iteration, config.signal_length], signal)
    

def load():
    # Only attempt to load if there is a file to read.
    if not os.path.exists(path + "save0.txt"):
        return False
    
    # Load latest save file
    filedat = np.load(path + "save0")
    
    iteration = filedat['arr_0'][0]
    exponent = filedat['arr_0'][1]
    signal_length = filedat['arr_0'][2]
    
    signal = filedat['arr_1']
    
    
    # The signal length may be the cause of the problem
    # which would require it to be initialized from args
    # and NOT from the save.
