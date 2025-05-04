"""Custom exception classes for PedPy.

This module defines a hierarchy of exceptions for clear and specific
error reporting when using PedPy.
"""


class PedPyError(Exception):
    """Base class for all PedPy-specific errors."""


class GeometryError(PedPyError):
    """Class reflecting errors when creating PedPy geometry objects."""

    def __init__(self, message):
        """Create GeometryError with the given message.

        Args:
            message: Error message
        """
        super().__init__(message)


class LoadTrajectoryError(PedPyError):
    """Class reflecting errors when loading trajectories with PedPy."""

    def __init__(self, message):
        """Create LoadTrajectoryError with the given message.

        Args:
            message: Error message
        """
        super().__init__(message)


class InputError(PedPyError):
    """Class reflecting errors when incorrect input was given."""

    def __init__(self, message):
        """Create InputError with the given message.

        Args:
            message: Error message
        """
        super().__init__(message)


class SpeedError(PedPyError):
    """Class reflecting errors when computing speeds with PedPy."""

    def __init__(self, message):
        """Create SpeedError with the given message.

        Args:
            message: Error message
        """
        super().__init__(message)


class AccelerationError(PedPyError):
    """Class reflecting errors when computing accelerations with PedPy."""

    def __init__(self, message):
        """Create AccelerationError with the given message.

        Args:
            message: Error message
        """
        super().__init__(message)


class PedPyAttributeError(AttributeError):
    """Class reflecting errors when accessing or assigning attributes."""

    def __init__(self, message):
        """Create PedPyAttributeError with the given message.

        Args:
            message: Error message
        """
        super().__init__(message)


class PedPyRuntimeError(RuntimeError):
    """Class reflecting errors when a runtime error occurs."""

    def __init__(self, message):
        """Create PedPyRuntimeError with the given message.

        Args:
            message: Error message
        """
        super().__init__(message)


class PedPyValueError(ValueError):
    """Class reflecting errors when a value error occurs."""

    def __init__(self, message):
        """Create PedPyValueError with the given message.

        Args:
            message: Error message
        """
        super().__init__(message)


class PedPyTypeError(TypeError):
    """Class reflecting errors when a type error occurs."""

    def __init__(self, message):
        """Create PedPyTypeError with the given message.

        Args:
            message: Error message
        """
        super().__init__(message)
