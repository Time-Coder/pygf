import ctypes

from .double4 import double4
from .genMat4 import genMat4


class matrix4d(genMat4):

    _type_ = ctypes.c_double

    @staticmethod
    def subvec_type()->type:
        return double4
