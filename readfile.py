

file = open("settings.txt", "r")

# Dictionary contains all settings as key value pairs
settings = {}
lines = file.readlines()

for l in lines:
    # Only attempt arguments
    if l[0] == "-":
        # Reform
        cuts = l.partition(":")
        idv = cuts[0][1:]
        value = cuts[2].strip()
        
        if value == "T":
            value = True
        elif value == "F":
            value = False
        else:
            print("odd value: ", value)
        
        # Convert to dictionary
        settings.update({idv: value})

# Dictionary to variables?

print(settings)
file.close()


