import config
#import numpy as np
import jax.numpy as np
import os

dirname = os.path.dirname(__file__)
pathed = os.path.join(dirname, "saves\save")
extension = ".npy"
#path = "saves/" + "save"
# BootstrapTensorPrime

# Saves data required to re-initialize the search
def save(exponent, siglen, signal, iteration):
    global pathed
    global extension

    # Get highest save
    maxsaves = config.settings["SaveCount"]
    for i in range(maxsaves):
        # Count to highest stored
        if i < maxsaves-1 and os.path.exists(pathed + str(i) + extension):
            continue
            
        # Remove oldest saves if maximum is reached
        if i == maxsaves-1 and os.path.exists(pathed + str(maxsaves-1) + extension):
            os.remove(pathed + str(maxsaves-1) + extension)
            os.remove(pathed + "signal" + str(maxsaves-1) + extension)
        
        # Increment saves
        for x in range(i):
            setval = i - x
            os.rename(pathed + str(setval - 1) + extension, 
                    pathed + str(setval) + extension)
            os.rename(pathed + "signal" + str(setval - 1) + extension, 
                    pathed +  "signal" + str(setval) + extension)
        break
    
    # Pack data
    # Exponent, Signal length are required to re-build
    # signal and iteration are required to resume within a prime check
    packed = [exponent, siglen, iteration]
    
    # Save data
    np.save(pathed + "0", packed)
    np.save(pathed + "signal0", signal)
    #.savez(pathed + "0", [config.exponent, iteration, config.signal_length], signal)
    

def load():
    global pathed
    global extension
    
    # Only attempt to load if there is a file to read.
    if not os.path.exists(pathed + "0" + extension):
        return None
    
    # Load latest save file
    filedat = np.load(pathed + "0" + extension, allow_pickle = True)
    filedat.append(np.load(pathed + "signal0" + extension, allow_pickle = True))
    
    # The signal length may be the cause of the problem
    # which would require it to be initialized from args
    # and NOT from the save.
    # This is not handled.
    
    #exponent = filedat[0]
    #signal_length = filedat[1]
    #iteration = filedat[2]
    #signal = filedat[3]
    #vals = [exponent, signal_length, iteration, signal]
    ids = ["prime", "siglen", "iteration", "signal"]
    
    return zip(ids, filedat)
