import ctypes

from .bool4 import bool4
from .genMat4 import genMat4


class matrix4b(genMat4):

    _type_ = ctypes.c_bool

    @staticmethod
    def subvec_type()->type:
        return bool4
