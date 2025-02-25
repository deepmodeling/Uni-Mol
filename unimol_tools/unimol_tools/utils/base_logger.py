# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import datetime
import logging
import os
import sys
import threading
from logging.handlers import TimedRotatingFileHandler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class PackagePathFilter(logging.Filter):
    """A custom logging filter for adding the relative path to the log record."""

    def filter(self, record):
        """add relative path to record"""
        pathname = record.pathname
        record.relativepath = None
        abs_sys_paths = map(os.path.abspath, sys.path)
        for path in sorted(abs_sys_paths, key=len, reverse=True):  # longer paths first
            if not path.endswith(os.sep):
                path += os.sep
            if pathname.startswith(path):
                record.relativepath = os.path.relpath(pathname, path)
                break
        return True


class Logger(object):
    """A custom logger class that provides logging functionality to console and file."""

    _instance = None
    _lock = threading.Lock()

    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    LOG_FORMAT = "%(asctime)s | %(relativepath)s | %(lineno)s | %(levelname)s | %(name)s | %(message)s"

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance

    def __init__(self, logger_name='None'):
        """
        :param logger_name: (str) The name of the logger (default: 'None')
        """
        self.logger = logging.getLogger(logger_name)
        logging.root.setLevel(logging.NOTSET)
        self.log_file_name = 'unimol_tools_{0}.log'.format(
            datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        )

        cwd_path = os.path.abspath(os.getcwd())
        self.log_path = os.path.join(cwd_path, "logs")

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.backup_count = 5

        self.console_output_level = 'INFO'
        self.file_output_level = 'INFO'

        self.formatter = logging.Formatter(self.LOG_FORMAT, self.DATE_FORMAT)

    def get_logger(self):
        """
        Get the logger object.

        :return: logging.Logger - a logger object.

        """
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(self.formatter)
            console_handler.setLevel(self.console_output_level)
            console_handler.addFilter(PackagePathFilter())
            self.logger.addHandler(console_handler)

            file_handler = TimedRotatingFileHandler(
                filename=os.path.join(self.log_path, self.log_file_name),
                when='D',
                interval=1,
                backupCount=self.backup_count,
                delay=True,
                encoding='utf-8',
            )
            file_handler.setFormatter(self.formatter)
            file_handler.setLevel(self.file_output_level)
            self.logger.addHandler(file_handler)
        return self.logger


# add highlight formatter to logger
class HighlightFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno == logging.WARNING:
            record.msg = "\033[93m{}\033[0m".format(record.msg)  # 黄色高亮
        return super().format(record)


logger = Logger('Uni-Mol Tools').get_logger()
logger.setLevel(logging.INFO)

# highlight warning messages in console
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setFormatter(HighlightFormatter(Logger.LOG_FORMAT, Logger.DATE_FORMAT))
