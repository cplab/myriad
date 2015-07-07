"""
Collection of internal utility metaclasses and function annotations.

TODO: Use logging module for logging
"""

__author__ = ["Pedro Rittner"]

from functools import wraps
from inspect import getcallargs
from types import FunctionType


def assert_list_type(m_list: list, m_type: type):
    """ Raises an error if m_list's members are not all of m_type. """
    msg = "Invalid argument(s) type(s): expected {0}"
    if not all(issubclass(type(e), m_type) for e in m_list):
        raise TypeError(msg.format(str(m_type)))


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


class IndexedSet(object):

    def __init__(self, elems=None):
        """
        Initializes an indexed set with the given elements.
        """
        self._my_dict = dict()
        self._curr_indx = -1
        if elems is None:
            return

        for index, elem in enumerate(elems):
            self._curr_indx += 1
            self._my_dict[index] = elem

    def append(self, n_elem):
        """
        Appends an element to the end of the set.

        TODO: If the element is already in the set, it is moved to the end.
        """
        if n_elem not in self._my_dict.values():
            self._curr_indx += 1
            self._my_dict[self._curr_indx] = n_elem

    def prepend(self, n_elem):
        """
        Prepends an element to the start of the set.

        TODO: If the element is already in the set, it is moved to the start.
        """
        if n_elem not in self._my_dict.values():
            items = list(self._my_dict.items())
            self._my_dict = {0: n_elem}
            for orig_indx, elem in items:
                self._my_dict[orig_indx+1] = elem

    def __getitem__(self, key):
        if type(key) is not int:
            msg = "IndexedSet indices must be integers, not {0}."
            raise TypeError(msg.format(type(key)))
        elif key > self._curr_indx or key < 0:
            raise IndexError("Index out of bounds.")
        return self._my_dict[key]

    def __repr__(self):
        _lst = list(range(self._curr_indx+1))
        for indx, value in self._my_dict.items():
            _lst[indx] = value
        return str(_lst)


def remove_header_parens(lines: list) -> list:
    """ Removes lines where top-level parentheses are found """
    open_index = (-1, -1)  # (line number, index within line)
    flattened_list = list('\n'.join(lines))

    # Find initial location of open parens
    linum = 0
    for indx, char in enumerate(flattened_list):
        if char == '\n':
            linum += 1
        elif char == '(':
            open_index = (linum, indx)
            break

    # print("open parenthese index: ", open_index)

    # Search for matching close parens
    close_index = (-1, -1)
    open_br = 0
    linum = 0
    for indx, char in enumerate(flattened_list[open_index[1]:]):
        if char == '\n':
            linum += 1
        elif char == '(':
            open_br += 1
            # print("Found ( at index ", indx, " open_br now ", open_br)
        elif char == ')':
            open_br -= 1
            # print("Found ) at index ", indx, " open_br now ", open_br)
        # Check if we're matched
        if open_br == 0:
            close_index = (linum, indx)
            break

    # print("close parentheses index: ", close_index)

    if open_index[0] == close_index[0]:
        return lines[1:]
    else:
        return lines[open_index[0]:][close_index[0]:]
