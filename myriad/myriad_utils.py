"""
Collection of internal utility metaclasses and function annotations.
:author Pedro Rittner
#TODO: Use logging module for logging
"""
from functools import wraps
from inspect import getcallargs
from copy import copy
from collections import OrderedDict


def assert_list_type(m_list: list, m_type: type):
    """ Raises an error if m_list's members are not all of m_type. """
    msg = "Invalid argument(s) type(s): expected {0}"
    if not all(issubclass(type(e), m_type) for e in m_list):
        raise TypeError(msg.format(str(m_type)))


def get_all_subclasses(cls):
    """ Gets all subclasses of a class """
    all_subclasses = []
    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))
    return all_subclasses


class wrap_file_function(object):
    """
    Wrap a function which takes a file or a str as it's first argument.
    If a str is provided, replace the first argument of the wrapped function
    with a file handle, and close the file afterwards.

    .. code-block:: python

        @wrap_file_function('w')
        def write_hi(f):
            f.write('Hello, World!')
        # The following will write to already open file handle:
        f = open('f1.txt', 'w')
        write_hi(f)
        f.close()
        # The following opens file f2.txt for writing, writes, then closes it:
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
            if (return_val is not None and
                    not issubclass(return_val.__class__, templ)):
                raise TypeError(msg.format(**msg_args))
        return return_val
    return _wrapper


class OrderedSet(object):
    """
    Set that remembers the order elements were added.
    """

    def __init__(self, contents: list=None):
        if contents is None:
            contents = []
        self._backing_set = set(contents)
        self._elements_ordered = contents

    @property
    def backing_set(self):
        """ Returns a shallow copy of the backing set """
        return copy(self._backing_set)

    def __iter__(self):
        return self._elements_ordered.__iter__()

    def __eq__(self, other):
        if hasattr(other, "backing_set"):
            return self._backing_set == other.backing_set
        else:
            raise TypeError("Invalid comparison type for OrderedSet: ",
                            other.__class__)

    def add(self, item):
        """ Adds the item to the backing set """
        if item in self._backing_set:
            raise ValueError("Item '${0}' already in OrderedSet".format(item))
        self._backing_set.add(item)
        self._elements_ordered.append(item)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return self._backing_set.__hash__()

    def __len__(self):
        return len(self._backing_set)

    def __contains__(self, item):
        return self._backing_set.__contains__(item)

    def isdisjoint(self, other):
        """
        Return True if the set has no elements in common with other.
        OrderedSets are disjoint if and only if their intersection is the empty
        set.
        """
        return self._backing_set.isdisjoint(other.backing_set)

    def __le__(self, other):
        return self._backing_set <= other.backing_set

    def issubset(self, other):
        """
        Test whether every element in the set is in other.
        """
        return self.__le__(other)

    def __lt__(self, other):
        """
        Test whether the set is a proper subset of other, that is, set <= other
        and set != other.
        """
        return self._backing_set < other.backing_set

    def __ge__(self, other):
        return self._backing_set >= other.backing_set

    def issuperset(self, other):
        """ Test whether every element in other is in the set. """
        return self.__ge__(other)

    def __gt__(self, other):
        return self._backing_set > other.backing_set

    def __or__(self, other):
        new_backing_set = copy(self._backing_set)
        new_backing_set_list = copy(self._elements_ordered)
        for other_val in other.backing_set:
            if other_val not in new_backing_set:
                new_backing_set.add(other_val)
                new_backing_set_list.append(other_val)
        return OrderedSet(new_backing_set_list)

    def union(self, other):
        """
        Return a new OrderedSet with elements from the set and all others.
        """
        return self.__or__(other)

    def __and__(self, other):
        new_backing_set = set()
        new_backing_set_list = list()
        for our_val in self._elements_ordered:
            if our_val in other:
                new_backing_set.add(our_val)
                new_backing_set_list.append(our_val)
        return OrderedSet(new_backing_set_list)

    def intersection(self, other):
        """ Return a new set with elements common to the set and all others """
        return self.__and__(other)

    def __sub__(self, other):
        new_backing_set = set()
        new_backing_set_list = list()
        for our_val in self._elements_ordered:
            if our_val not in other:
                new_backing_set.add(our_val)
                new_backing_set_list.append(our_val)
        return OrderedSet(new_backing_set_list)

    def difference(self, other):
        """
        Return a new set with elements in the set that are not in the others.
        """
        return self.__sub__(other)

    def __xor__(self, other):
        new_backing_set_list = list()
        # Get all elements together first, then only add unique ones
        union_set = self.union(other)
        for union_val in union_set:
            in_ours = union_val in self
            in_other = union_val in other
            if in_ours ^ in_other:
                new_backing_set_list.append(union_val)
        return OrderedSet(new_backing_set_list)

    def __repr__(self):
        return str(self._elements_ordered)

    def __str__(self):
        return str(self._elements_ordered)

    def symmetric_difference(self, other):
        """
        Return a new set with elements in either the set or other but not both.
        """
        return self.__xor__(other)


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
    # If the indices match (same line), just skip the first line altogether
    if open_index[0] == close_index[0]:
        return lines[1:]
    else:
        return lines[open_index[0]:][close_index[0]:]


def indent_fix(source_lines: str) -> str:
    """ Returns the source lines as a single string, left-indented """
    if source_lines is None:
        return None
    # Empty string indented correctly is just the empty string
    elif source_lines == "":
        return ""
    # Single-line statements need only be trimmed
    elif source_lines.count('\n') <= 1:
        return " ".join(source_lines.split())

    # Multi-line statements need to have tabs converted to 4 spaces
    result_str = source_lines.replace('\t', '    ')
    # Split the string into lines and get the first line (ignoring empty lines)
    as_lines = [line for line in result_str.splitlines() if len(line) > 0]
    # and count the number of spaces until the 1st non-whitespace character
    count = 0
    for char in as_lines[0]:
        if char != ' ':
            break
        count += 1
    # Now we remove the count whitespaces from each line and return
    return "\n".join([line[count:] for line in as_lines])


def indent_fix_lines(source_lines: list) -> list:
    """ Returns the source lines as a list of strings, left-indented """
    # Empty list fixed is just the empty string
    if len(source_lines) == 0:
        return ""
    # Single-line strings can just be stripped of leading/lingering whitespace
    elif len(source_lines) == 1:
        return source_lines[0].strip()
    # Join list to pass to indent_fix
    result_str = indent_fix("".join(source_lines))
    return result_str


def filter_odict_values(to_filter: OrderedDict, *args) -> OrderedDict:
    """ Filters out values of type *args from dictionary """
    new_dict = OrderedDict()
    for key, value in to_filter.items():
        if value:
            accept = True
            for v_class in args:
                accept = not issubclass(value.__class__, v_class)
                if accept is False:
                    break
            if accept:
                new_dict[key] = value
    return new_dict
