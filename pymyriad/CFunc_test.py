"""
Tester module for CTypes and ast_parse.
Author: Alex Davies
"""

import ast_function_assembler


    

        

if __name__ == '__main__':

    def tester_fun(a, b):
        return a + b

    myriadfun = ast_function_assembler.pyfun_to_cfun(tester_fun)
    print(myriadfun)
    
