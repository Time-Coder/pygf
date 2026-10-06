import ctypes

from .genQuat import genQuat
from .half import from_bits, half_fields, to_bits


class quath(genQuat):
    """A quaternion whose four components are stored as binary16.

    genQuat's own w/x/y/z properties reach the field through
    ``ctypes.Structure.__getattribute__(self, "w")``, which cannot work here:
    the fields are private (``_hw`` and friends) because a ctypes Structure
    cannot carry both a field and a converting property under one name.
    """

    _fields_ = half_fields(4)

    @property
    def dtype(self)->type:
        return ctypes.c_uint16

    @property
    def w(self)->float:
        return from_bits(ctypes.Structure.__getattribute__(self, "_hw"))

    @w.setter
    def w(self, w:float)->None:
        ctypes.Structure.__setattr__(self, "_hw", to_bits(w))
        self._update_data()

    @property
    def x(self)->float:
        return from_bits(ctypes.Structure.__getattribute__(self, "_hx"))

    @x.setter
    def x(self, x:float)->None:
        ctypes.Structure.__setattr__(self, "_hx", to_bits(x))
        self._update_data()

    @property
    def y(self)->float:
        return from_bits(ctypes.Structure.__getattribute__(self, "_hy"))

    @y.setter
    def y(self, y:float)->None:
        ctypes.Structure.__setattr__(self, "_hy", to_bits(y))
        self._update_data()

    @property
    def z(self)->float:
        return from_bits(ctypes.Structure.__getattribute__(self, "_hz"))

    @z.setter
    def z(self, z:float)->None:
        ctypes.Structure.__setattr__(self, "_hz", to_bits(z))
        self._update_data()
