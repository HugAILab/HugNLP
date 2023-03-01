import sys
import logging
import datasets
import transformers


def init_logger(log_file, log_level, dist_rank):
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    datasets.utils.logging.disable_propagation()
    # transformers.utils.logging.enable_propagation()

    logger = logging.getLogger('')
    log_format = logging.Formatter(
        fmt=
        '[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    logger.setLevel(log_level)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # transformer_logger = logging.getLogger('transformers')
    # transformer_logger.handlers = []
    # transformer_logger.propagate = True

    if dist_rank in [-1, 0]:
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
        logging.getLogger('transformers').addHandler(file_handler)
