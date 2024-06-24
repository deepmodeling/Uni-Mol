# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function
import logging
import os
import sys
import datetime
from logging.handlers import TimedRotatingFileHandler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class PackagePathFilter(logging.Filter):
    """A custom logging filter for adding the relative path to the log record."""
    def filter(self, record):
        """add relative path to record
        """
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
    def __init__(self, logger_name='None'):
        """
        :param logger_name: (str) The name of the logger (default: 'None')
        """
        self.logger = logging.getLogger(logger_name)
        logging.root.setLevel(logging.NOTSET)
        self.log_file_name = 'uniqsar_{0}.log'.format(
            datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        )

        cwd_path = os.path.abspath(os.getcwd())
        self.log_path = os.path.join(cwd_path, "logs")

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.backup_count = 5

        self.console_output_level = 'INFO'
        self.file_output_level = 'INFO'
        self.DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
        self.formatter = logging.Formatter(
            "%(asctime)s | %(relativepath)s | %(lineno)s | %(levelname)s | %(name)s | %(message)s",
            self.DATE_FORMAT
        )

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
                    encoding='utf-8'
                )
            file_handler.setFormatter(self.formatter)
            file_handler.setLevel(self.file_output_level)
            self.logger.addHandler(file_handler)
        return self.logger

logger = Logger('Uni-Mol Tools').get_logger()
logger.setLevel(logging.INFO)