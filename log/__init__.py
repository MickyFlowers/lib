# config.py
import logging
import colorlog
import os


def setup_logging():
    # Get logger level from environment variable
    logger_level = os.getenv("LOGGER_LEVEL", "INFO")

    # Set logger level
    logger = logging.getLogger()
    logger.setLevel(logger_level.upper())
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = colorlog.ColoredFormatter(
        "%(log_color)s[%(asctime)s] [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
        secondary_log_colors={},
        style="%",
    )

    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)


# 初始化日志配置
setup_logging()
