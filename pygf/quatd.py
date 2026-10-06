import ctypes

from .genQuat import genQuat


class quatd(genQuat):

    _fields_ = [
        ('w', ctypes.c_double),
        ('x', ctypes.c_double),
        ('y', ctypes.c_double),
        ('z', ctypes.c_double)
    ]

    @property
    def dtype(self)->type:
        return ctypes.c_double
