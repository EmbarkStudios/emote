"""
Supporting types used for working with the memory
"""

from typing import Dict, Generic, Tuple, TypeVar


# Number is *either* an int or a float, but *not* covariant.
# For example: Sequence[Union[int, float]] accepts [int, float]
#              Sequence[Number] only accept [int, int, ...] or
#              [float, float, ...]

Number = TypeVar("Number", int, float)


# Technically far too general, but there is no good support for
# multidimensional arrays
class Matrix(Generic[Number]):
    pass


SampleResult = Dict[str, Matrix]
SamplePoint = Tuple[int, int, int]
