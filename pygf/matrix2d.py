import ctypes

from .double2 import double2
from .genMat2 import genMat2


class matrix2d(genMat2):

    _type_ = ctypes.c_double

    @staticmethod
    def subvec_type()->type:
        return double2
