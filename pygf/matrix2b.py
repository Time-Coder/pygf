import ctypes

from .bool2 import bool2
from .genMat2 import genMat2


class matrix2b(genMat2):

    _type_ = ctypes.c_bool

    @staticmethod
    def subvec_type()->type:
        return bool2
