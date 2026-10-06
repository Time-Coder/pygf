from typing import Tuple, TypeAlias, Union

from .genVec import genVec
from .helper import Number


class genVec3(genVec):

    def __len__(self)->int:
        return 3

Vec3Type: TypeAlias = Union[genVec3, Tuple[Number, Number, Number]]
