# We use this custom exception handling in the project to handle all the errors that will come into the project, simply we can say that we are handling all the errors that will come into the project in a single place.

import sys

# Sys module in python provides various functions and variables that are used to manipulate different parts of the python runtime environment. It allows operating on the python interpreter as it provides access to some variables used or maintained by the interpreter and to functions that interact strongly with the interpreter.
# Read more about sys module here: https://docs.python.org/3/library/sys.html
from src.logger import logging


def error_message_detail(error,error_detail:sys):
    _,_,exec_tb = error_detail.exc_info()
    file_name = exec_tb.tb_frame.f_code.co_filename
    error_message = f"Error occured in python script name {file_name} on line number {exec_tb.tb_lineno} and error is {str(error)}"
    
    return error_message
    
class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_detail= error_detail)
        #self.error_detail = error_detail        
        
    def __str__(self):
        return f"{self.error_message}"

# Read more about custom exception handling here: https://www.programiz.com/python-programming/user-defined-exception

if __name__ == '__main__':
    try:
        a = 10
        b = 0
        c = a/b
        print(c)
    except Exception as e:
        logging.error(e)
        raise CustomException(e,error_detail=sys)