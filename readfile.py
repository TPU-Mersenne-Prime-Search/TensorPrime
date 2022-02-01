
# Returns settings in the form of a dictionary
def getSettings():
    file = open("settings.txt", "r")

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
            else:
                print("odd value: ", value)
            
            # Add to dictionary
            settings.update({idv: value})

    file.close()
    # Dictionary to variables?
    return settings


