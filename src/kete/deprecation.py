import functools
import warnings


def deprecated(deprecated_version, additional_msg=None):
    """
    Decorator for marking a function as deprecated.
    """
    additional_msg = "" if additional_msg is None else " " + str(additional_msg)

    def decorate(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            warnings.warn(
                (
                    f"Function `{func.__name__}` was deprecated in kete version "
                    f"'{deprecated_version}.{additional_msg}"
                ),
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapped

    return decorate


def rename(new_func, deprecated_version, additional_msg=None, old_name=None):
    """
    Decorator for marking a function as being renamed.
    """
    additional_msg = "" if additional_msg is None else " " + str(additional_msg)

    @functools.wraps(new_func)
    def wrapped(*args, **kwargs):
        warnings.warn(
            (
                f"Function `{old_name}` was renamed in kete version "
                f"'{deprecated_version}'. Use `{new_func.__name__}` instead."
                f"{additional_msg}"
            ),
            category=DeprecationWarning,
            stacklevel=2,
        )
        return new_func(*args, **kwargs)

    return wrapped
