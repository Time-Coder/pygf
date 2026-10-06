import ctypes

from .float3 import float3
from .genMat3 import genMat3


class matrix3f(genMat3):

    _type_ = ctypes.c_float

    @staticmethod
    def subvec_type()->type:
        return float3
