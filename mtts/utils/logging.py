import logging
import os
import sys

nesting_level = 0
is_start = None


def get_logger(file_name, use_error_log=False, log_dir=None):

    logger = logging.getLogger(file_name)
    logging_level = getattr(logging, 'INFO')
    logger.setLevel(logging_level)

    if log_dir is None:
        log_dir = os.path.join("./", "log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_dir = os.path.join(log_dir, "log.txt")
    else:
        log_dir = os.path.join(log_dir, "log.txt")

    formatter = logging.Formatter(fmt='%(asctime)s %(filename)s: %(levelname)s: %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging_level)
    stdout_handler.setFormatter(formatter)

    logger.addHandler(stdout_handler)

    fh = logging.FileHandler(log_dir)
    fh.setLevel(logging_level)
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    if use_error_log:
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.WARNING)
        stderr_handler.setFormatter(formatter)
        logger.addHandler(stderr_handler)
    logger.propagate = False
    return logger
