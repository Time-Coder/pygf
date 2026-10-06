from __future__ import annotations

import ctypes
import math
from typing import Any, Tuple, TypeAlias, Union, cast

from .genType import MathForm, genType
from .genVec import genVec
from .genVec3 import genVec3
from .helper import Number, is_number


class genQuat(genType, ctypes.Structure):

    def __init__(self, *args):
        genType.__init__(self)

        # Routed through self[i] rather than ctypes.Structure.__init__ so that a
        # half quaternion converts on the way in. Structure.__init__ assigns the
        # raw bit pattern of a half field, which would make the identity
        # quaternion (1, 0, 0, 0) read back as w == 1.4e-45.
        if len(args) == 0:
            self[0], self[1], self[2], self[3] = 1.0, 0.0, 0.0, 0.0
            return

        if len(args) == 1:
            arg = args[0]
            if not isinstance(arg, genQuat):
                raise TypeError(f"invalid argument type(s) for {self.__class__.__name__}()")

            for i in range(4):
                self[i] = arg[i]

            return

        if len(args) == 2:
            w = args[0]
            v = args[1]

            if not is_number(w) or not isinstance(v, genVec):
                raise TypeError(f"invalid argument type(s) for {self.__class__.__name__}()")

            self[0], self[1], self[2], self[3] = w, v.x, v.y, v.z
            return

        if len(args) == 4:
            for i, value in enumerate(args):
                self[i] = value

            return

        raise ValueError(f"invalid arguments for {self.__class__.__name__}()")

    @property
    def math_form(self)->MathForm:
        return MathForm.Quat

    @property
    def w(self)->float:
        # The ctypes field descriptor lands on the concrete subclass that declares
        # _fields_, which sits before genQuat in the MRO, so instance reads never
        # reach this property; super() cannot see the descriptor either, so going
        # through Structure is what makes the body work when it is called.
        return float(ctypes.Structure.__getattribute__(self, "w"))

    @w.setter
    def w(self, w:float)->None:
        ctypes.Structure.__setattr__(self, 'w', w)
        self._update_data()

    @property
    def x(self)->float:
        # The ctypes field descriptor lands on the concrete subclass that declares
        # _fields_, which sits before genQuat in the MRO, so instance reads never
        # reach this property; super() cannot see the descriptor either, so going
        # through Structure is what makes the body work when it is called.
        return float(ctypes.Structure.__getattribute__(self, "x"))

    @x.setter
    def x(self, x:float)->None:
        ctypes.Structure.__setattr__(self, 'x', x)
        self._update_data()

    @property
    def y(self)->float:
        # The ctypes field descriptor lands on the concrete subclass that declares
        # _fields_, which sits before genQuat in the MRO, so instance reads never
        # reach this property; super() cannot see the descriptor either, so going
        # through Structure is what makes the body work when it is called.
        return float(ctypes.Structure.__getattribute__(self, "y"))

    @y.setter
    def y(self, y:float)->None:
        ctypes.Structure.__setattr__(self, 'y', y)
        self._update_data()

    @property
    def z(self)->float:
        # The ctypes field descriptor lands on the concrete subclass that declares
        # _fields_, which sits before genQuat in the MRO, so instance reads never
        # reach this property; super() cannot see the descriptor either, so going
        # through Structure is what makes the body work when it is called.
        return float(ctypes.Structure.__getattribute__(self, "z"))

    @z.setter
    def z(self, z:float)->None:
        ctypes.Structure.__setattr__(self, 'z', z)
        self._update_data()

    @property
    def xyz(self)->genVec3:
        vec_type = genVec.vec_type(self.dtype, 3)
        return vec_type(self.x, self.y, self.z)

    @xyz.setter
    def xyz(self, xyz:genVec3)->None:
        if not isinstance(xyz, genVec3):
            raise TypeError(f'must be genVec3, not {type(xyz)}')

        ctypes.Structure.__setattr__(self, 'x', xyz.x)
        ctypes.Structure.__setattr__(self, 'y', xyz.y)
        ctypes.Structure.__setattr__(self, 'z', xyz.z)
        self._update_data()

    @staticmethod
    def quat_type(dtype:type)->type:
        return genType.gen_type(MathForm.Quat, dtype, (4,))

    @property
    def shape(self)->Tuple[int]:
        return (4,)

    def __len__(self)->int:
        return 4

    def __getitem__(self, index:int)->float:
        if index == 0:
            return self.w
        elif index == 1:
            return self.x
        elif index == 2:
            return self.y
        elif index == 3:
            return self.z
        else:
            raise IndexError("index out of range")

    def __setitem__(self, index:int, value:float)->None:
        if index == 0:
            self.w = value
        elif index == 1:
            self.x = value
        elif index == 2:
            self.y = value
        elif index == 3:
            self.z = value
        else:
            raise IndexError("index out of range")

    def _op(self, operator:str, other:Union[float, bool, int, genType])->Union[genQuat, genVec]:
        if operator == "**" or (operator in ["/", "//", "%"] and isinstance(other, genType)):
            raise TypeError(f"unsupported operand type(s) for {operator}: '{self.__class__.__name__}' and '{other.__class__.__name__}'")

        if operator == "*" and isinstance(other, genType):
            if not isinstance(other, (genQuat, genVec)):
                raise TypeError(f"unsupported operand type(s) for {operator}: '{self.__class__.__name__}' and '{other.__class__.__name__}'")

            if isinstance(other, genVec) and len(other) != 3:
                raise TypeError(f"unsupported operand type(s) for {operator}: '{self.__class__.__name__}' and '{other.__class__.__name__}'")

            result_dtype = self._bin_op_dtype(operator, self.dtype, other.dtype, False)
            quat_type = self.quat_type(result_dtype)
            if isinstance(other, genQuat):
                result:genQuat = quat_type()
                result.w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
                result.x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
                result.y = self.w * other.y + self.y * other.w + self.z * other.x - self.x * other.z
                result.z = self.w * other.z + self.z * other.w + self.x * other.y - self.y * other.x
                return result
            else: # if isinstance(other, genVec):
                vec_quat:genQuat = quat_type(0, other)
                temp_quat:genQuat = self * vec_quat * quat_type(self.w, -self.xyz) / math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
                return temp_quat.xyz

        if isinstance(other, genType) and not isinstance(other, genQuat):
            raise TypeError(f"unsupported operand type(s) for {operator}: '{self.__class__.__name__}' and '{other.__class__.__name__}'")

        # See genMat._op: the base preserves the math form, so the result is one.
        return cast("Union[genQuat, genVec]", genType._op(self, operator, other))

    def _iop(self, operator:str, other:Union[float, bool, int, genType])->genQuat:
        if operator == "**" or (operator in ["/", "//", "%"] and isinstance(other, genType)):
            raise TypeError(f"unsupported operand type(s) for {operator}=: '{self.__class__.__name__}' and '{other.__class__.__name__}'")

        if operator == "*" and isinstance(other, genType):
            if not isinstance(other, genQuat):
                raise TypeError(f"unsupported operand type(s) for {operator}: '{self.__class__.__name__}' and '{other.__class__.__name__}'")

            w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
            x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
            y = self.w * other.y + self.y * other.w + self.z * other.x - self.x * other.z
            z = self.w * other.z + self.z * other.w + self.x * other.y - self.y * other.x
            self.w = w
            self.x = x
            self.y = y
            self.z = z

            return self

        if isinstance(other, genType) and not isinstance(other, genQuat):
            raise TypeError(f"unsupported operand type(s) for {operator}=: '{self.__class__.__name__}' and '{other.__class__.__name__}'")

        return cast(genQuat, genType._iop(self, operator, other))

    def __iter__(self):
        # Through self[i], not the _fields_ names: a half field is a c_uint16
        # holding a bit pattern, and reading the field directly would yield that
        # integer instead of the float the component represents.
        for i in range(4):
            yield self[i]

    def __contains__(self, value:Any):
        return any(component == value for component in self)

QuatType: TypeAlias = Union[genQuat, Tuple[Number, Number, Number, Number]]
