import logging

logger = logging.getLogger("NewsByte")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("[%(levelname)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

DEBUG_MODE = True

if DEBUG_MODE:
    logger.setLevel(logging.DEBUG)
    ch.setLevel(logging.DEBUG)
