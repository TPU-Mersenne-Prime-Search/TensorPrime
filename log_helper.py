import logging


def init_logger(filename):
    logging.basicConfig(
        level=logging.DEBUG, format='[%(threadName)s %(asctime)s]  %(levelname)s: %(message)s', filename=filename)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(console)
