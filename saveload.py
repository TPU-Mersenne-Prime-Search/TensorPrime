import os
import logging

import jax.numpy as np

from config import config

# dirname = os.path.dirname(__file__)
# pathed = os.path.join(dirname, r"saves\save")
pathed = '.'
extension = ".npy"


# Saves data required to re-initialize the search
def save(exponent, siglen, signal, iteration, d=None, d_p=None):
    # Get highest save
    max_saves = int(config.get("TensorPrime", "SaveCount"))

    # Extra work for Gerbicz error checking
    GEC = config.getboolean("TensorPrime", "GECEnabled")

    for i in range(max_saves):
        # Count to highest stored
        if i + 1 < max_saves and os.path.exists(os.path.join(pathed, f"{exponent}-{i}{extension}")):
            continue

        # Remove oldest saves if maximum is reached
        if i + 1 == max_saves and os.path.exists(os.path.join(
                pathed, f"{exponent}-{max_saves - 1}{extension}")):
            os.remove(os.path.join(
                pathed, f"{exponent}-{max_saves - 1}{extension}"))

        # Increment saves
        for x in range(i):
            setval = i - x
            os.rename(os.path.join(pathed, f"{exponent}-{setval - 1}{extension}"),
                      os.path.join(pathed, f"{exponent}-{setval}{extension}"))
        break

    # Pack data
    # Exponent, Signal length are required to re-build
    # signal and iteration are required to resume within a prime check
    packed = [exponent, siglen, iteration, signal]

    if GEC:
        GECvals = np.zeros((2, siglen))
        GECvals.at[0].set(d)
        GECvals.at[1].set(d_p)
        packed.push(GECvals)

    # Save data
    np.save(os.path.join(pathed, f"{exponent}-{0}"), packed)


def load(source, exponent):
    if source == -1:
        source = 0
    file = os.path.join(pathed, f"{exponent}-{source}{extension}")

    # Only attempt to load if there is a file to read.
    if not os.path.exists(file):
        logging.info("File does not exist.")
        return None

    # Load latest save file
    filedat = np.load(file, allow_pickle=True)

    # The signal length may be the cause of the problem
    # which would require it to be initialized from args
    # and NOT from the save.
    # This is not handled.

    vals = {"prime": filedat[0], "fft": filedat[1],
            "iteration": filedat[2], "signal": filedat[3]}

    # Extra work for Gerbicz error checking
    GEC = config.getboolean("TensorPrime", "GECEnabled")

    if GEC:
        if len(filedat) == 4:
            logging.warning("No GEC file found, Disabling.")
            config.set("TensorPrime", "GECEnabled", False)
        else:
            vals["d"] = filedat[4]
            vals["d_prev"] = filedat[5]

    return vals


def clean(exponent, start=0):
    max_saves = int(config.get("TensorPrime", "SaveCount"))

    for i in range(start, max_saves):
        file = os.path.join(pathed, f"{exponent}-{i}{extension}")
        if os.path.exists(file):
            os.remove(file)
        else:
            logging.info("Cleaned all savefiles.")
            break
