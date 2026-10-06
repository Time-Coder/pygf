from __future__ import annotations

import ctypes
from collections.abc import Callable
from typing import Any, Tuple, TypeAlias, Union, cast

from .genType import MathForm, genType
from .genVec import VecType, genVec
from .helper import is_number


class genMatIterator:

    def __init__(self, mat:genMat):
        self.__mat:genMat = mat
        self.__current_index:int = 0

    def __iter__(self)->genMatIterator:
        # An iterator has to be iterable, otherwise iter() on an iterator that
        # genMat.__iter__ already handed out fails.
        return self

    def __next__(self)->genVec:
        if self.__current_index >= self.__mat.rows:
            raise StopIteration()

        result:genVec = cast(genVec, self.__mat[self.__current_index])
        self.__current_index += 1
        return result


class genMat(genType, ctypes.Array):

    _type_ = ctypes.c_double
    _length_ = 0

    def __len__(self)->int:
        # genType declares __len__ and comes first in the MRO, so ctypes.Array's
        # implementation has to be reached explicitly.
        return ctypes.Array.__len__(self)

    def __init__(self, *args):
        genType.__init__(self)
        n:int = min(self.rows, self.cols)
        for i in range(n):
            self[i, i] = 1

        i: int = 0
        n_data: int = len(self)
        n_args: int = len(args)

        if n_args == 0:
            return

        if n_args == 1:
            arg = args[0]
            if is_number(arg):
                for i in range(n):
                    self[i, i] = arg
                return

            if isinstance(arg, genMat):
                for i in range(self.rows):
                    for j in range(self.cols):
                        self[i, j] = arg[i, j]
                return

        for i_arg, arg in enumerate(args):
            if is_number(arg):
                ctypes.Array.__setitem__(self, i, arg)

                i += 1
                if i == n_data:
                    if n_args != 1 and i_arg != n_args - 1:
                        raise ValueError(f"invalid arguments for {self.__class__.__name__}()")

                    return

            elif isinstance(arg, genVec):
                sub_n_arg: int = len(arg)
                for sub_i_arg, value in enumerate(arg):
                    ctypes.Array.__setitem__(self, i, value)

                    i += 1
                    if i == n_data:
                        if n_args != 1 and (i_arg != n_args - 1 or sub_i_arg != sub_n_arg - 1):
                            raise ValueError(f"invalid arguments for {self.__class__.__name__}()")

                        return

            else:
                raise TypeError(f"invalid argument type(s) for {self.__class__.__name__}()")

        raise ValueError(f"invalid arguments for {self.__class__.__name__}()")

    @property
    def math_form(self)->MathForm:
        return MathForm.Mat

    @property
    def rows(self)->int:
        # These were shape[1]/shape[0], i.e. transposed. Every genMat is square
        # (mat_type reads shape[0] and ignores the rest), so the swap never
        # showed -- but __getitem__ and __setitem__ mix the two conventions
        # against ctypes' flat storage, so the names have to mean what they say.
        return self.shape[0]

    @property
    def cols(self)->int:
        return self.shape[1]

    @property
    def dtype(self)->type:
        return self._type_

    @staticmethod
    def mat_type(dtype:type, shape:Tuple[int, ...]):
        # genType.gen_type names a matrix `matrix{shape[0]}{suffix}`, so a
        # non-square request used to come back silently truncated to shape[0]x
        # shape[0]. That is how outerProduct(float3, float2) built a 2x2 and then
        # wrote three rows into it. Matrices here are square by design, so say so.
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError(f"only square matrices are supported, got {shape}")

        return genType.gen_type(MathForm.Mat, dtype, shape)

    # ctypes types these as fixed slot wrappers, so the tuple index that genMat
    # adds on top of the C signature can never match; the override is deliberate.
    def __getitem__(self, index:Union[int,Tuple[int, int]])->Union[int,bool,float,genVec]:  # ty: ignore
        if isinstance(index, int):
            result_type = genVec.vec_type(self.dtype, self.cols)
            result:genVec = result_type(*(ctypes.Array.__getitem__(self, index*self.cols + j) for j in range(self.cols)))
            result._mat_start_index = self.cols * index
            result._related_mat = self
            return result
        elif isinstance(index, tuple):
            return ctypes.Array.__getitem__(self, index[0]*self.cols + index[1])

    def __setitem__(self, index:Union[int,Tuple[int, int]], value:Union[float,int,bool,genVec])->None:  # ty: ignore
        if isinstance(index, int):
            for j in range(self.cols):
                ctypes.Array.__setitem__(self, self.cols*index + j, cast(genVec, value)[j])
        elif isinstance(index, tuple):
            ctypes.Array.__setitem__(self, index[0]*self.cols + index[1], value)

    def __iter__(self)->genMatIterator:
        return genMatIterator(self)

    def at(self, row:int, col:int)->float:
        """One element by (row, col).

        __getitem__ has to advertise genVec as well because an int index yields a
        row, so element-wise callers use this to get a plain scalar.
        """
        return cast(float, self[row, col])

    def put(self, row:int, col:int, value:float)->None:
        """Store one element by (row, col)."""
        self[row, col] = value

    def __contains__(self, value:Any)->bool:
        # Every element, not every row: comparing self[i] to a scalar asked a row
        # whether it equalled a number, and range(len(self)) overran the rows
        # anyway, since len() is the flat element count.
        if is_number(value):
            return any(self.at(i, j) == value
                       for i in range(self.rows) for j in range(self.cols))
        elif isinstance(value, genVec) and len(value) == self.cols:
            for i in range(self.rows):
                if self[i] == value:
                    return True

        return False

    def _op(self, operator:str, other:Union[float, bool, int, genType])->Union[genMat, genVec]:
        if operator == "**" or (operator in ["/", "//", "%"] and isinstance(other, genType)):
            raise TypeError(f"unsupported operand type(s) for {operator}: '{self.__class__.__name__}' and '{other.__class__.__name__}'")

        if operator == "*" and isinstance(other, genType):
            if not isinstance(other, (genMat, genVec)) or self.cols != (other.rows if isinstance(other, genMat) else len(other)):
                raise TypeError(f"unsupported operand type(s) for {operator}: '{self.__class__.__name__}' and '{other.__class__.__name__}'")

            result_dtype = self._bin_op_dtype(operator, self.dtype, other.dtype, False)
            result_shape = (self.rows, other.cols) if isinstance(other, genMat) else (self.rows,)
            result_type = self.gen_type(other.math_form, result_dtype, result_shape)
            result = result_type()
            if isinstance(result, genMat):
                # result is a matrix, so the product was matrix * matrix.
                right = cast(genMat, other)
                for i in range(result.rows):
                    for j in range(result.cols):
                        value = 0
                        for k in range(self.cols):
                            value += self.at(i, k) * right.at(k, j)

                        result.put(i, j, value)
            elif isinstance(result, genVec):
                for i in range(len(result)):
                    value = 0
                    for k in range(self.cols):
                        value += self[i, k] * other[k]

                    result[i] = value

            return result

        # The base builds the result through _bin_op_type, which preserves the math
        # form, so for this subclass the result really is a genMat or a genVec.
        return cast("Union[genMat, genVec]", genType._op(self, operator, other))

    def _iop(self, operator:str, other:Union[float, bool, int, genType])->genMat:
        if operator == "**" or (operator in ["/", "//", "%"] and isinstance(other, genType)):
            raise TypeError(f"unsupported operand type(s) for {operator}=: '{self.__class__.__name__}' and '{other.__class__.__name__}'")

        if operator == "*" and isinstance(other, genType):
            if not isinstance(other, genMat):
                raise TypeError(f"unsupported operand type(s) for {operator}=: '{self.__class__.__name__}' and '{other.__class__.__name__}'")

            if self.cols != other.rows or other.rows != other.cols:
                raise TypeError(f"unsupported operand type(s) for {operator}=: '{self.__class__.__name__}' and '{other.__class__.__name__}'")

            # self[:] = product[:] used to be a silent no-op: a slice index matched
            # neither branch of __setitem__, so `matrix *= matrix` never wrote
            # anything back. Copy the elements instead.
            product:genMat = cast(genMat, self * other)
            for i in range(self.rows):
                for j in range(self.cols):
                    self.put(i, j, product.at(i, j))

            self._update_data()
            return self

        return cast(genMat, genType._iop(self, operator, other))

    def _compare_op(self, operator:str, other:Union[float, bool, int, genType])->genType:
        # A matrix index yields a row, so walk rows and columns to stay
        # element-wise; len(self) counts elements and would overrun.
        other_is_homo:bool = self._is_homo(other)
        if not other_is_homo and not is_number(other):
            raise TypeError(f"unsupported operand type(s) for {operator}: '{self.__class__.__name__}' and '{other.__class__.__name__}'")

        result:genMat = self.gen_type(self.math_form, ctypes.c_bool, self.shape)()
        operator_func:Callable[[Any,Any], Any] = self._operator_funcs[operator]
        for i in range(self.rows):
            for j in range(self.cols):
                result[i, j] = operator_func(self[i, j], genType._at(other, (i, j)) if other_is_homo else other)

        return result

    def _compare_rop(self, operator:str, other:Union[float, bool, int, genType])->genType:
        result:genMat = self.gen_type(self.math_form, ctypes.c_bool, self.shape)()
        operator_func:Callable[[Any,Any], Any] = self._operator_funcs[operator]
        for i in range(self.rows):
            for j in range(self.cols):
                result[i, j] = operator_func(other, self[i, j])

        return result

MatType: TypeAlias = Union[genMat, Tuple[VecType, ...]]
