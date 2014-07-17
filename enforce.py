#!/usr/bin/python3

from functools import wraps
from inspect import getcallargs


def enforce_annotations(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        for arg, val in getcallargs(f, *args, **kwargs).items():
            if arg in f.__annotations__:
                templ = f.__annotations__[arg]
                msg = "Argument {arg} to {f} doesn't match annotation type {t}"
                if not issubclass(val.__class__, templ.__class__):
                    raise ValueError(msg.format(arg=arg, f=f, t=templ))
        return_val = f(*args, **kwargs)
        if 'return' in f.__annotations__:
            templ = f.__annotations__['return']
            msg = "Return value of {f} does not match annotation type {t}"
            if not issubclass(val.__class__, templ.__class__):
                    raise ValueError(msg.format(arg=arg, f=f, t=templ))
        return return_val
    return wrapper


@enforce_annotations
def f(x: int, y: float) -> float:
    return x+y


def main():
    print(f(1, y=2.2))
    print(f(1, y=2))

if __name__ == "__main__":
    main()
