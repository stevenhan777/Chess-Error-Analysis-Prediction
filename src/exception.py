"""
src/exception.py
Custom exception class that captures full traceback details.
"""

import sys
import traceback


def _get_error_details(error, error_detail: sys):
    """Extract file name, line number, and message from a sys.exc_info() tuple."""
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is not None:
        file_name   = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        file_name   = "unknown"
        line_number = -1

    return (
        f"Error in '{file_name}', line {line_number}: {str(error)}"
    )


class ChessAnalysisException(Exception):
    """
    Wrapper exception that includes file + line information.

    Usage
    -----
    try:
        ...
    except Exception as e:
        raise ChessAnalysisException(e, sys) from e
    """

    def __init__(self, error_message, error_detail: sys):
        super().__init__(str(error_message))
        self.error_message = _get_error_details(error_message, error_detail)

    def __str__(self):
        return self.error_message
