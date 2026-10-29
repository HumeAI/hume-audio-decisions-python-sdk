# Maintained by hand and listed in .fernignore, so SDK regeneration leaves it in place.

import operator


def round_div(numerator: int, denominator: int) -> int:
    return (2 * numerator + denominator) // (2 * denominator)


def ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def check_integer(name: str, value: int) -> int:
    # operator.index also accepts NumPy integers, which are not subclasses of int. Its result is a
    # plain int, which arithmetic needs because a small NumPy integer can overflow.
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {value!r}.") from None
