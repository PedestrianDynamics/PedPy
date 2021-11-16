import logging

logger = logging.getLogger("JPSreport")
FORMAT = "[%(asctime)s] [%(levelname)s]: %(message)s"
logging.basicConfig(format=FORMAT)


def log_debug(msg, **kwargs):
    logger.debug(msg, **kwargs)


def log_info(msg, **kwargs):
    logger.info(msg, **kwargs)


def log_warning(msg, **kwargs):
    logger.warning(msg, **kwargs)


def log_error(msg, **kwargs):
    logger.error(msg, **kwargs)
