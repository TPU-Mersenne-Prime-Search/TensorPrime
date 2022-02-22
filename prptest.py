import config
import saveload
import IBDWT as ibdwt
import time


def probable_prime(power, startPos=0, s=3):
    start = time.time()
    
    printIter = config.settings["PrintIter"]
    timestamp = config.settings["Timestamps"]
    saveIter = config.settings["SaveIter"]
    
    for i in range(startPos, power):
    
        if i % saveIter == 0:
            saveload.save(s, i)
        # s *= s
        # s = s % ((1 << power) - 1)
        s = ibdwt.squaremod_with_ibdwt(s)
        
        if timestamp and i % printIter == 0:
            time_elapsed = time.time() - start
            print("Time elapsed at iteration ", i, ": ", time_elapsed, ". S = ", s)
            
        
            
    if s == 9:
        return True
    return False