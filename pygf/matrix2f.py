import ctypes

from .float2 import float2
from .genMat2 import genMat2


class matrix2f(genMat2):

    _type_ = ctypes.c_float

    @staticmethod
    def subvec_type()->type:
        return float2
