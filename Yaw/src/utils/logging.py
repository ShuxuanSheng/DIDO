# Copyright 2004-present Facebook. All Rights Reserved.

# Put some color in you day!
import logging
import sys


try:
    import coloredlogs
    log_format = "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s"
    coloredlogs.install(level=logging.INFO, fmt=log_format)
except ImportError:
    pass

logging.basicConfig(
    stream=sys.stdout,
    format=log_format,
    level=logging.INFO
)
#
# logging.basicConfig(
#     stream=sys.stdout,
#     format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
#     level=logging.INFO,
# )

