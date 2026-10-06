import ctypes

from .genQuat import genQuat


class quatb(genQuat):

    _fields_ = [
        ('w', ctypes.c_bool),
        ('x', ctypes.c_bool),
        ('y', ctypes.c_bool),
        ('z', ctypes.c_bool)
    ]

    @property
    def dtype(self)->type:
        return ctypes.c_bool
