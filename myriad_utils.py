"""
Collection of internal utility metaclasses and function annotations.

@author Pedro Rittner
"""

from functools import wraps
from inspect import getcallargs
from types import FunctionType


class wrap_file_function(object):
    """
    Wrap a function which takes a file or a str as it's first argument.
    If a str is provided, replace the first argument of the wrapped function
    with a file handle, and close the file afterwards

    Example:

    @wrap_file_function('w')
    def write_hi(f):
        f.write('hi!\n')

    # This will write to already open file handle.
    f = open('f1.txt', 'w')
    write_hi(f)
    f.close()

    # This will open file f2.txt with mode 'w', write to it, and close it.
    write_hi('f2.txt')
    """

    def __init__(self, *args):
        self.modes = args if args else ('r',)

    def __call__(self, func):
        def wrapped(*args, **kwargs):
            close = []  # Files that should be closed
            files = []  # File handles that should be passed to func
            num_files = len(self.modes)
            filep = None
            try:
                for i, mode in enumerate(self.modes):
                    filep = args[i]
                    if isinstance(filep, str):
                        filep = open(filep, mode)
                        close.append(filep)
                    files.append(filep)

                # Replace the files in args when calling func
                args = files + list(args[num_files:])

                # Make function call and return value
                return func(*args, **kwargs)
            finally:
                for filep in close:
                    filep.close()
        return wrapped


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


class TypeEnforcer(type):
    """
    Metaclass designed to setup runtime typechecking for all subclasses.
    """

    def __new__(mcs, name, bases, attrs):
        """ Overwrites all methods with enforced annotated wrappers """
        for attr_name, attr_value in list(attrs.items()):
            if isinstance(attr_value, FunctionType):
                attrs[attr_name] = enforce_annotations(attr_value)

        return super(TypeEnforcer, mcs).__new__(mcs, name, bases, attrs)


class _Foo(object, metaclass=TypeEnforcer):
    """ Dummy class to test enforcement """

    def __init__(self, a: int=0):
        self.a = a


def main():
    f = _Foo(5.)
    print(f.a)


if __name__ == "__main__":
    main()
