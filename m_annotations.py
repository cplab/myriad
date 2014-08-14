"""
Function annotation enforcer.

@author Pedro Rittner
"""

from functools import wraps
from inspect import getcallargs


def enforce_annotations(fun):
    """
    Function annotation to enforce function argument and return types.
    """
    @wraps(fun)
    def _wrapper(*args, **kwargs):
        """ Check each annotated argument type for type correctness """
        for arg, val in getcallargs(fun, *args, **kwargs).items():
            if arg in fun.__annotations__:
                templ = fun.__annotations__[arg]
                msg_args = {'arg': arg, 'f': fun, 't1': type(val), 't2': templ}
                msg = """Argument mismatch in call to {f}:
                \'{arg}\' is of type {t1}, expected type {t2}"""
                if val is not None and not issubclass(val.__class__, templ):
                    raise TypeError(msg.format(**msg_args))
        # Call wrapped function and get return value
        return_val = fun(*args, **kwargs)
        if 'return' in fun.__annotations__:
            templ = fun.__annotations__['return']
            msg_args = {'f': fun, 't1': type(return_val), 't2': templ}
            msg = """Return type mismatch in call to {f}:
            Call return type {t1} does not match expected type {t2}"""
            if (return_val is not None
                    and not issubclass(return_val.__class__, templ)):
                raise TypeError(msg.format(**msg_args))
        return return_val
    return _wrapper
