from configparser import ConfigParser, Error as ConfigParserError
import logging


def config_read():
    """Reads the configuration file."""
    config = ConfigParser()
    config.optionxform = lambda option: option
    localfile = "settings.txt"
    try:
        config.read([localfile])
    except ConfigParserError as e:
        logging.exception("ERROR reading '{0}' file:".format(localfile))
    if not config.has_section("PrimeNet"):
        # Create the section to avoid having to test for it later
        config.add_section("PrimeNet")
    return config


config = config_read()
