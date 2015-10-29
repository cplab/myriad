"""
..module:: myriad_testing
    :platform: Linux
    :synopsis: Testing utilities for myriad

.. moduleauthor:: Pedro Rittner <pr273@cornell.edu>
"""

import logging
import unittest
import io
import sys

from termcolor import cprint


def trim_spaces(string: str) -> str:
    """ Removes insignificant whitespace for comparison purposes"""
    if string is None:
        return ""
    else:
        return " ".join(string.split())


def _print_log(lines: str):
    """ Prints log lines, coloring where appropriate """
    # If empty, don't print anything - better than printing an empty line
    if lines is None or lines == '':
        return
    for line in lines.split("\n"):
        if line.count("DEBUG"):
            cprint(line, color="white", on_color="on_grey", file=sys.stderr)
        elif line.count("INFO"):
            cprint(line, color="blue", on_color="on_grey", file=sys.stderr)
        elif line.count("WARNING"):
            cprint(line, color="yellow", on_color="on_grey", file=sys.stderr)
        elif line.count("ERROR"):
            cprint(line, color="red", on_color="on_grey", file=sys.stderr)
        elif line.count("CRITICAL"):
            cprint(line, color="white", on_color="on_red", file=sys.stderr)
        else:
            print(line, file=sys.stderr)

def set_external_loggers(
        logger_name: str,
        *args,
        log_filename=None,
        log_level=logging.DEBUG,
        log_format: str="%(name)s - %(levelname)s - %(message)s"):
    """ Sets external loggers to be activated by the wrapped test case """
    # Create logger with level
    log = logging.getLogger(logger_name)
    log.setLevel(log_level)
    # Create log handler
    log_stream = io.StringIO()
    log_handler = logging.StreamHandler(stream=log_stream)
    log_handler.setLevel(log_level)
    # Create and set log formatter
    log_formatter = logging.Formatter(log_format)
    log_handler.setFormatter(log_formatter)
    # Set log handler
    log.addHandler(log_handler)
    # Add handler/formatter to module(s) we're testing
    for external_log in args:
        external_log.addHandler(log_handler)
        external_log.setLevel(log_level)

    def decorator(cls):
        """ Decorator for wrapping the target class """
        setattr(cls, "log_stream", log_stream)
        setattr(cls, "log_filename", log_filename)
        return cls
    return decorator


_ERRORS_SEEN = set()
_FAILURES_SEEN = set()


class MyriadTestCase(unittest.TestCase):
    """
    TestCase class used to facilitate keeping logs of failed/error'd tests
    """

    @classmethod
    def setUpClass(cls):
        """ Setup class state for logging"""
        cls.curr_errors = 0
        cls.curr_failures = 0
        if hasattr(cls, "log_stream"):
            cls.log_file = getattr(cls, "log_stream")
        else:
            cls.log_file = io.StringIO()
        cls.log_list = []

    def assertTrimStrEquals(self, str1: str, str2: str):
        """ Asserts two strings are equal after trimming whitespaces """
        self.assertEqual(trim_spaces(str1), trim_spaces(str2))

    def _format_log(self,
                    is_err: bool,
                    err_rslt: (unittest.TestResult, str)) -> str:
        """ Formats the log string in a pretty way """
        desc_str = "Log for " + ("Error @ " if is_err else "Failure @ ")
        header_str = desc_str + err_rslt[0].__repr__() + ' '
        hyphens = '=' * max(0, (80 - len(header_str)) // 2)
        header_str = hyphens + header_str + hyphens + '\n'
        foot_str = ('=' * min(80, len(header_str))) + '\n'
        log_contents = self.log_file.getvalue().strip(u"\x00")  # UTF8 format
        if len(hyphens) < 2:
            return foot_str + header_str + foot_str.replace('=', '-') +\
                log_contents + '\n' + err_rslt[1] + foot_str
        else:
            return header_str + log_contents + '\n' + err_rslt[1] + foot_str

    def run(self, result=None):
        """ Show log output on failed tests """
        # Get new result from running the test
        new_result = super().run(result)
        new_errors = len(new_result.errors) if new_result else 0
        new_failures = len(new_result.failures) if new_result else 0
        # If we've added an error/failure, append the formatted log to list
        if new_errors > self.curr_errors:
            new_error = result.errors[self.curr_errors]
            if new_error not in _ERRORS_SEEN:
                self.log_list.append(self._format_log(True, new_error))
                self.curr_errors = new_errors
                _ERRORS_SEEN.add(new_error)
        elif new_failures > self.curr_failures:
            new_failure = result.failures[self.curr_failures]
            if new_failure not in _FAILURES_SEEN:
                self.log_list.append(self._format_log(False, new_failure))
                self.curr_failures = new_failures
                _FAILURES_SEEN.add(new_failure)
        # Truncate the StringIO whether an error/failure occured or not
        self.log_file.truncate(0)
        return new_result

    @classmethod
    def tearDownClass(cls):
        """ Prints the log to file or stderr if no file is specified """
        filename = getattr(cls, "log_filename")
        if len(cls.log_list) == 0:
            return
        if filename:
            with open(filename, "wt") as log_file:
                log_file.writelines(cls.log_list)
            print("\nError log written to", filename, file=sys.stderr)
        else:
            print('\n', file=sys.stderr)
            for line in cls.log_list:
                _print_log(line)
