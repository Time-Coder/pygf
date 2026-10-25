from typing import Tuple, Union
from typing_extensions import TypeAlias

from .genVec import genVec
from .helper import Number


class genVec4(genVec):

    def __len__(self)->int:
        return 4

Vec4Type: TypeAlias = Union[genVec4, Tuple[Number, Number, Number, Number]]
