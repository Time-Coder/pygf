import ctypes

from .float4 import float4
from .genMat4 import genMat4


class matrix4f(genMat4):

    _type_ = ctypes.c_float

    @staticmethod
    def subvec_type()->type:
        return float4
