import argparse

parser = argparse.ArgumentParser()

#in order to add more arguments to the parser, attempt a similar declaration to below. Anthing without a dash is becomes ordinal and required
parser.add_argument("-p","--prime",type = int, help="seed for the mersenne prime being tested", default = None)


flags = vars(parser.parse_args())
if not flags["prime"]:
	raise ValueError("runtime requires a prime number for testing!")
	exit()

#flags is a dictionary in python types, in a per flag key-value mapping, which can be accessed via, 
#eg, flags["prime"], which will return the integer passed in.
#If you want specific behavior for the options, eg prime is none, exit()""
