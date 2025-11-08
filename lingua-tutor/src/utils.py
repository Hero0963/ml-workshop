# src/utils.py
import time
from functools import wraps
from loguru import logger


def timer_decorator(func):
    """
    A decorator that logs the start, end, and execution time of a function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Starting '{func.__name__}'...")
        start_time = time.time()

        result = func(*args, **kwargs)

        end_time = time.time()
        duration = end_time - start_time
        logger.success(f"Finished '{func.__name__}' in {duration:.4f} seconds.")
        return result

    return wrapper
