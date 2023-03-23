"""Global definitions of enums."""

from enum import Enum


class VelocityBorderMethod(Enum):  # pylint: disable=too-few-public-methods
    """Identifier which frames at the border to include at movement calculation."""

    EXCLUDE = 1
    SINGLE_SIDED = 2
    ADAPTIVE = 3
    MAXIMUM_RANGE = 4


VelocityBorderMethod.EXCLUDE.__doc__ = (
    "Exclude frames where frame_step can not be used on both sides"
)
VelocityBorderMethod.SINGLE_SIDED.__doc__ = (
    "Use frame_step only on one side, if not possible on the other"
)
VelocityBorderMethod.ADAPTIVE.__doc__ = (
    "Use a balanced window, with as many frames available"
)
VelocityBorderMethod.MAXIMUM_RANGE.__doc__ = "Use as many frames as possible"
