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
      
      # Add to dictionary
      settings.update({idv: value})

  file.close()
  # Dictionary to variables?
  # return settings
