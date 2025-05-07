def def_symbol(lib, name, return_type, arg_types, docstring=None):
    c_function = getattr(lib, name)
    c_function.restype = return_type
    c_function.argtypes = arg_types
    if docstring is not None:
        c_function.__doc__ = docstring


def load_libps(name: str):
    """
    Try to find picoscope library on the current system.

    :param name: prefix of the library (e.g., "ps2000a", "ps6000")
    :returns: a handle (ctypes or windll) on the library.
    """
    import platform

    system = platform.system()
    if system == "Linux":
        from ctypes import cdll

        lib = cdll.LoadLibrary(f"lib{name}.so.2")
    elif system == "Darwin":
        raise NotImplementedError("MacOSX is not supported yet")
    elif system == "Windows":
        from ctypes import windll
        from ctypes.util import find_library

        lib = windll.LoadLibrary(find_library(f"{name}.dll"))
    else:
        raise NotImplementedError(
            f"Unsupported platform for loading lib{name}: {system}"
        )
    return lib