import sys

def error_message_detail(error, error_detail: sys):
    """
    Build detailed error messages with file name, line number, and error text.
    Handles both caught exceptions and manual raises.
    """
    exc_type, exc_obj, exc_tb = error_detail.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        error_message = (
            f"Error in python script [{file_name}] "
            f"line no [{exc_tb.tb_lineno}] "
            f"error message [{str(error)}]"
        )
    else:
        # Fallback: if no traceback is available
        error_message = f"Error message [{str(error)}]"
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
