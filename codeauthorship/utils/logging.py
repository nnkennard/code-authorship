import logging


LOGGING_NAMESPACE = 'codeauthorship'


def configure_logger():
    # Create logger.
    logger = logging.getLogger(LOGGING_NAMESPACE)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    # Also log to console.
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def get_logger():
    return logging.getLogger(LOGGING_NAMESPACE)
