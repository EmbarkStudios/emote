"""
Supporting types used for working with the memory
"""

from typing import Generic, TypeVar


# Number is *either* an int or a float, but *not* covariant.
# For example: Sequence[int | float] accepts [int, float]
#              Sequence[Number] only accept [int, int, ...] or
#              [float, float, ...]

Number = TypeVar("Number", int, float)


# Technically far too general, but there is no good support for
# multidimensional arrays
class Matrix(Generic[Number]):
    pass


SampleResult = dict[str, Matrix]
SamplePoint = tuple[int, int, int]
