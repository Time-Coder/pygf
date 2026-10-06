from typing import Tuple, TypeAlias, Union

from .genVec import genVec
from .helper import Number


class genVec4(genVec):

    def __len__(self)->int:
        return 4

Vec4Type: TypeAlias = Union[genVec4, Tuple[Number, Number, Number, Number]]
