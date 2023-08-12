import logging
from configparser import ConfigParser
from configparser import Error as ConfigParserError


def config_read():
    """Reads the configuration file."""
    config = ConfigParser()
    config.optionxform = lambda option: option
    localfile = "settings.txt"
    try:
        config.read([localfile])
    except ConfigParserError:
        logging.exception(f"ERROR reading {localfile!r} file:")
    if not config.has_section("PrimeNet"):
        # Create the section to avoid having to test for it later
        config.add_section("PrimeNet")
    return config


config = config_read()
