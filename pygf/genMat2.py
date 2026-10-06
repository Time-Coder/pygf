from typing import Tuple, TypeAlias, Union

from .genMat import genMat
from .genVec2 import Vec2Type


class genMat2(genMat):

    _length_ = 4

    @property
    def shape(self)->Tuple[int, ...]:
        return (2, 2)

Mat2Type: TypeAlias = Union[genMat2, Tuple[Vec2Type, Vec2Type]]
