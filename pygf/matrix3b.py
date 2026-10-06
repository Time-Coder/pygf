import ctypes

from .bool3 import bool3
from .genMat3 import genMat3


class matrix3b(genMat3):

    _type_ = ctypes.c_bool

    @staticmethod
    def subvec_type()->type:
        return bool3
