from typing import Union
import casadi as cs
#import numpy.typing as ntp

Vector = Union[cs.MX, cs.SX]
Scalar = Union[float, cs.MX]
Matrix = Union[cs.MX, cs.SX]
Angle = Vector
TangentVector = Vector