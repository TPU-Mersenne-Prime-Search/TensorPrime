import logging


def init_logger(filename):
    logging.basicConfig(
        level=logging.INFO, format='[%(threadName)s %(asctime)s]  %(levelname)s: %(message)s', filename=filename)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s]  %(levelname)s: %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
