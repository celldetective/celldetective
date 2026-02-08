def color_from_status(status: int, recently_modified: bool = False) -> str:
    """
    Get color based on status.

    Parameters
    ----------
    status : int
        Status code.
    recently_modified : bool, optional
        If True, use recently modified color scheme. Default is False.

    Returns
    -------
    str
        Color string.
    """

    if not recently_modified:
        if status == 0:
            return "tab:blue"
        elif status == 1:
            return "tab:red"
        elif status == 2:
            return "yellow"
        else:
            return "k"
    else:
        if status == 0:
            return "tab:cyan"
        elif status == 1:
            return "tab:orange"
        elif status == 2:
            return "tab:olive"
        else:
            return "k"


def color_from_class(cclass: int, recently_modified: bool = False) -> str:
    """
    Get color based on class.

    Parameters
    ----------
    cclass : int
        Class ID.
    recently_modified : bool, optional
        If True, use recently modified color scheme. Default is False.

    Returns
    -------
    str
        Color string.
    """

    if not recently_modified:
        if cclass == 0:
            return "tab:red"
        elif cclass == 1:
            return "tab:blue"
        elif cclass == 2:
            return "yellow"
        else:
            return "k"
    else:
        if cclass == 0:
            return "tab:orange"
        elif cclass == 1:
            return "tab:cyan"
        elif cclass == 2:
            return "tab:olive"
        else:
            return "k"
