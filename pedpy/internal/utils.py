"""Module containing internal utilities"""
import functools


def alias(aliases):
    """Decorator for specifying parameters aliases.

    Args:
        aliases: Dictionary with parameter names as keys and
            the aliases as values.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(**kwargs):
            for name, other_name in aliases.items():
                if name not in kwargs and other_name in kwargs:
                    kwargs[name] = kwargs[other_name]
            return func(**kwargs)

        return wrapper

    return decorator
