import logging


def get_logger(level: int = 20) -> logging.Logger:
    """get_logger.

    Parameters
    ----------
    level : int
        - default: 20 (logging.INFO)

    Returns
    -------
    logging.Logger

    """
    logger = logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s (%(filename)s:%(lineno)d)",
    )
    return logger
