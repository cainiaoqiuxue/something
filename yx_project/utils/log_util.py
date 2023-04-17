import logging


def get_logger(log_level='info', log_path=None):
    log_level_dict = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'error': logging.ERROR,
        'critical': logging.CRITICAL,
        'warning': logging.WARNING
    }
    log_level = log_level_dict.get(log_level, logging.INFO)
    logger = logging.getLogger(__name__)
    log_format = "%(asctime)s[%(levelname)s] - %(filename)s: %(message)s"
    logging.basicConfig(level=log_level, format=log_format)

    if log_path:
        log_formatter = logging.Formatter(log_format)
        normal_handler = logging.FileHandler(log_path, encoding='utf8')
        normal_handler.setLevel(log_level)
        normal_handler.setFormatter(log_formatter)
        logger.addHandler(normal_handler)

    return logger