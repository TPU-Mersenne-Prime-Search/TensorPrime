import config
import saveload
import IBDWT as ibdwt
import time


def probable_prime(power, startPos=0, s=3):
    start = time.time()
    
    timestamp = config.settings["Timestamps"]
    # Uses counters to avoid modulo check
    saveIter = config.settings["SaveIter"]
    saveIcount = saveIter
    printIter = config.settings["PrintIter"]
    printIcount = printIter
    
    for i in range(startPos, power):
    
        if saveIcount == 0:
            saveload.save(s, i)
            saveIcount = saveIter
            print(i)
        # s *= s
        # s = s % ((1 << power) - 1)
        s = ibdwt.squaremod_with_ibdwt(s)
        
        if timestamp:
            if  printIcount == 0:
                time_elapsed = time.time() - start
                print("Time elapsed at iteration ", i, ": ", time_elapsed, ". S = ", s)
                printIcount = printIter
            printIcount -= 1
            
        #Iterate counters
        saveIcount -= 1
            
    if s == 9:
        return True
    return False