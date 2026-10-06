import ctypes

from .double3 import double3
from .genMat3 import genMat3


class matrix3d(genMat3):

    _type_ = ctypes.c_double

    @staticmethod
    def subvec_type()->type:
        return double3
