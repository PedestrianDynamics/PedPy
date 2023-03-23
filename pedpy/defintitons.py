from aenum import Enum


class VelocityBorderMethod(Enum):  # pylint: disable=too-few-public-methods
    """Identifier which frames at the border to include at movement calculation."""

    _init_ = "value __doc__"
    EXCLUDE = 1, "Exclude frames where frame_step can not be used on both sides"
    SINGLE_SIDED = (
        2,
        "Use frame_step only on one side, if not possible on the other",
    )
    ADAPTIVE = 3, ""
    MAXIMUM_RANGE = 4, "Use as many frames as possible"
