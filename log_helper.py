import logging
import time


def create_mersenne_logger(file_name, logger_name, level=logging.INFO):
    m_logger = logging.Logger(logger_name, level=level)

    fh = logging.FileHandler(filename=file_name, mode="a", delay=False)
    fmt = logging.Formatter("%(levelname)s:%(name)s:%(asctime)s %(message)s")
    # documentation online describes valid default field names for formatter

    fh.setFormatter(fmt)
    m_logger.addHandler(fh)

    return m_logger


def time_name(name: str = None):
    if name is None:
        name = ".log"
    return time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "_" + name


def init_logger(opts: dict) -> logging.Logger:
    # opts is presumed to be the cli parsed flags and settings
    keys = opts.keys()

    logger_name = (opts["loggername"] if "loggername" in keys
                   else "tensorprime")
    file_name = time_name(opts["filename"] if "filename" in keys
                          else "tensorprime.log")

    return create_mersenne_logger(file_name, logger_name, level=logging.INFO)
