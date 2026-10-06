import ctypes

from .genQuat import genQuat


class quatf(genQuat):

    _fields_ = [
        ('w', ctypes.c_float),
        ('x', ctypes.c_float),
        ('y', ctypes.c_float),
        ('z', ctypes.c_float)
    ]

    @property
    def dtype(self)->type:
        return ctypes.c_float
