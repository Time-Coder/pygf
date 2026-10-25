from typing import Tuple, Union
from typing_extensions import TypeAlias

from .genVec import genVec
from .helper import Number


class genVec2(genVec):

    def __len__(self)->int:
        return 2

Vec2Type: TypeAlias = Union[genVec2, Tuple[Number, Number]]
