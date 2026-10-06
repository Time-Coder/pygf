from typing import Tuple, TypeAlias, Union

from .genMat import genMat
from .genVec4 import Vec4Type


class genMat4(genMat):

    _length_ = 16

    @property
    def shape(self)->Tuple[int, ...]:
        return (4, 4)

Mat4Type: TypeAlias = Union[genMat4, Tuple[Vec4Type, Vec4Type, Vec4Type, Vec4Type]]
