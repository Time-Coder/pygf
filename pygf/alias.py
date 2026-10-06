from .double2 import double2
from .double3 import double3
from .double4 import double4
from .float2 import float2
from .float3 import float3
from .float4 import float4
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
