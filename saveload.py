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
def save(exponent, siglen, signal, iteration, d = None, d_p = None):
    global pathed
    global extension

    # Get highest save
    maxsaves = config.settings["SaveCount"]
    # Extra work for Gerbicz error checking
    GEC = config.settings["GECEnabled"]
    
    for i in range(maxsaves):
        # Count to highest stored
        if i < maxsaves-1 and os.path.exists(pathed + str(i) + extension):
            continue
            
        # Remove oldest saves if maximum is reached
        if i == maxsaves-1 and os.path.exists(pathed + str(maxsaves-1) + extension):
            os.remove(pathed + str(maxsaves-1) + extension)
            os.remove(pathed + "signal" + str(maxsaves-1) + extension)
            if GEC:
                os.remove(pathed + "GEC" + str(maxsaves-1) + extension)
                
        
        # Increment saves
        for x in range(i):
            setval = i - x
            os.rename(pathed + str(setval - 1) + extension,
                    pathed + str(setval) + extension)
            os.rename(pathed + "signal" + str(setval - 1) + extension,
                    pathed +  "signal" + str(setval) + extension)
            if GEC:
                os.rename(pathed + "GEC" + str(setval - 1) + extension,
                        pathed +  "GEC" + str(setval) + extension)
        break
    
    # Pack data
    # Exponent, Signal length are required to re-build
    # signal and iteration are required to resume within a prime check
    packed = [exponent, siglen, iteration]
    
    # Save data
    np.save(pathed + "0", packed)
    np.save(pathed + "signal0", signal)
    
    if GEC:
        GECvals = np.zeros((2, siglen))
        GECvals.at[0] = d
        GECvals.at[1] = d_prev
        np.save(pathed + "GEC0", GECvals)
    
    

def load(source):
    global pathed
    global extension
    
    if source == -1:
        source = 0
    ext = str(source) + extension
    
    # Only attempt to load if there is a file to read.
    if not os.path.exists(pathed + ext):
        print("File does not exist.")
        return None
    
    # Load latest save file
    filedat = np.load(pathed + ext, allow_pickle = True)
    signal = np.load(pathed + "signal" + ext, allow_pickle = True)
    
        
    
    # The signal length may be the cause of the problem
    # which would require it to be initialized from args
    # and NOT from the save.
    # This is not handled.
    
    exponent = filedat[0]
    signal_length = filedat[1]
    iteration = filedat[2]
    vals = [exponent, signal_length, iteration, signal, None, None]
    ids = ["prime", "siglen", "iteration", "signal", "d", "d_prev"]
    
    
    # Extra work for Gerbicz error checking
    GEC = config.settings["GECEnabled"]
    
    if GEC:
        if not os.path.exists(pathed + "GEC" + ext):
            print("No GEC file found, Disabling.")
            config.settings["GECEnabled"] = False
        else:
            GECs = np.load(pathed + "GEC" + ext, allow_pickle = True)
            vals[4] = GECs[0]
            vals[5] = GECs[1]
    
    return zip(ids, vals)


def clean(start = 0):
    global pathed
    global extension
    maxsaves = config.settings["SaveCount"]
    
    for i in range(start, maxsaves):
        num = str(i)
        if os.path.exists(pathed + num + extension):
            os.remove(pathed + num + extension)
            os.remove(pathed + "signal" + num + extension)
        else:
            print("Cleaned all savefiles.")
            break
        
        
        
        
        
        
    
