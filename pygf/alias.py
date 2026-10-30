from .double2 import double2
from .double3 import double3
from .double4 import double4
from .float2 import float2
from .float3 import float3
from .float4 import float4
from .genType import (
    COMPONENT_ROLE,
    DIRECTION_ROLE,
    DISPLACEMENT_ROLE,
    POSITION_ROLE,
    register_alias_role,
)
from .half2 import half2
from .half3 import half3
from .half4 import half4
from .matrix4d import matrix4d


class color3d(double3):
    pass

class color3f(float3):
    pass

class color3h(half3):
    pass


class color4d(double4):
    pass

class color4f(float4):
    pass

class color4h(half4):
    pass


class normal3d(double3):
    pass

class normal3f(float3):
    pass

class normal3h(half3):
    pass


class point3d(double3):
    pass

class point3f(float3):
    pass

class point3h(half3):
    pass


class vector3d(double3):
    pass

class vector3f(float3):
    pass

class vector3h(half3):
    pass


class frame4d(matrix4d):
    pass


class texCoord2d(double2):
    pass

class texCoord2f(float2):
    pass

class texCoord2h(half2):
    pass


class texCoord3d(double3):
    pass

class texCoord3f(float3):
    pass

class texCoord3h(half3):
    pass


# --- what each alias means, for the result type of arithmetic ------------------------------
# A position and a displacement occupy the same three floats but are not the same kind of
# thing: subtracting two positions has to yield a displacement, and adding two positions is
# meaningless. Registering the role here (rather than in genType, which cannot import these
# classes) is what lets the arithmetic keep the meaning instead of flattening every result to
# the base vector.
for _alias, _base in ((color3d, double3), (color3f, float3), (color3h, half3),
                      (color4d, double4), (color4f, float4), (color4h, half4)):
    register_alias_role(_alias, _base, COMPONENT_ROLE)

for _alias, _base in ((normal3d, double3), (normal3f, float3), (normal3h, half3)):
    register_alias_role(_alias, _base, DIRECTION_ROLE)

for _alias, _base in ((point3d, double3), (point3f, float3), (point3h, half3)):
    register_alias_role(_alias, _base, POSITION_ROLE)

for _alias, _base in ((vector3d, double3), (vector3f, float3), (vector3h, half3)):
    register_alias_role(_alias, _base, DISPLACEMENT_ROLE)

for _alias, _base in ((texCoord2d, double2), (texCoord2f, float2), (texCoord2h, half2),
                      (texCoord3d, double3), (texCoord3f, float3), (texCoord3h, half3)):
    register_alias_role(_alias, _base, COMPONENT_ROLE)
