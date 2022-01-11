import sys

# Benchmarking
#import . as bench
# Lucas-Lehmer test
import mersenne.lucas as ll
# FFT
#import . as fft




def main():
    # Command line arguments
    args = sys.argv[1:]
    
    # Argument option to load from file?
    
    # Determine which function is wanted,
    # Run relevant function
    
    if args[0] == "-bench":
        pass
    
    if args[0] == "-fft":
        pass
    
    if args[0] == "-ll":
        # Type check passed arguments
        inval = int(args[1])
        outval = ll.naive_lucas_lehmer(inval)
        print(outval)
    

main()