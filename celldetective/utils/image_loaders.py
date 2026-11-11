import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from typing import Callable, Optional, List, Union, Tuple, Dict, Any

import numpy as np
from celldetective.utils.io import save_tiff_imagej_compatible
from natsort import natsorted
from tifffile import imread, memmap, TiffFile

from celldetective.utils.experiment import get_position_movie_path
from celldetective.utils.image_cleaning import interpolate_nan_multichannel
from celldetective.utils.normalization import normalize_multichannel
from celldetective import get_logger

import logging
import threading
import warnings

# Serializes the native image-decode path of :func:`load_frames`. ``tifffile``
# decoding and ``warnings.catch_warnings`` both mutate process-global
# state and are not thread-safe; without this lock a background prefetch thread
# (e.g. the viewer's StackLoader) decoding the same stack concurrently with the
# main thread corrupts the heap and triggers a Windows access violation
# (0xC0000005). The lock is process-local, so multiprocessing workers are
# unaffected, and decoding is not a threaded hot path, so contention is negligible.
# Reentrant: _read_images takes it for the page reads of a lazy stack, and load_frames
# already holds it when it calls _read_images.
_DECODE_LOCK = threading.RLock()

# Page shape of the stacks read by memory map, by absolute path, with the (modification
# time, size) it was read at: the read by page index, bound to fail on them, is only tried
# once per file, and a file replaced since is probed again.
_MEMMAP_PAGE_SHAPES: Dict[str, tuple] = {}

# Page plan of the stacks read, by absolute path, with the (modification time, size)
# it was built at (see _get_page_plan). The oldest are dropped past _MAX_PAGE_PLANS:
# the plan of a multi-file set holds a reference per plane.
_PAGE_PLANS: Dict[str, tuple] = {}
_MAX_PAGE_PLANS = 64

# Axes stored inside a page: the image plane and the samples of an RGB page.
_PAGE_AXES = "YXS"

logger = get_logger(__name__)

# Suppress tifffile warnings about missing files in MMStack
logging.getLogger("tifffile").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*MMStack series is missing files.*")


def locate_stack(position: str, prefix: str = "Aligned") -> np.ndarray:
    """

    Locate and load a stack of images.

    Parameters
    ----------
    position : str
        The position folder within the well where the stack is located.
    prefix : str, optional
        The prefix used to identify the stack. The default is 'Aligned'.

    Returns
    -------
    stack : ndarray
        The loaded stack as a NumPy array with shape ``(T, Y, X, C)``.

    Raises
    ------
    FileNotFoundError
        If no stack with the specified prefix is found.

    Notes
    -----
    This function locates and loads a stack of images based on the specified position and prefix.
    It assumes that the stack is stored in a directory named 'movie' within the specified position.
    The function loads the stack as a NumPy array and reshapes it to ``(T, Y, X, C)`` using
    TIFF metadata (ImageJ or OME-TIFF axes) when available, falling back to shape heuristics
    otherwise. ``.tif``, ``.tiff``, ``.ome.tif`` and ``.ome.tiff`` files are found; when
    several match the prefix, the first in natural order is used, with a warning.

    A movie with both Z and time keeps its time points: the slices of each are
    reduced to their maximum intensity projection, with a warning. Without a time axis,
    the Z slices are taken as frames.

    Examples
    --------
    >>> stack = locate_stack(position, prefix='Aligned')
    # Locate and load a stack of images for further processing.

    """

    stack_path = get_position_movie_path(position, prefix)
    if stack_path is None:
        raise FileNotFoundError(f"No movie with prefix {prefix} found...")

    file_path = stack_path.replace("\\", "/")

    # --- Metadata-aware loading via TiffFile.series ---
    try:
        stack = _load_stack_from_series(file_path)
        if stack is not None:
            return stack
    except Exception as e:
        logger.debug(f"Metadata-aware loading failed, reading the pages as frames: {e}")

    # --- Fallback: the pages with the axis rule of a plain stack (see _to_tzc),
    # read without the OME metadata, which may be what failed ---
    plan = _get_page_plan(file_path)
    with TiffFile(file_path, is_ome=False) as tif:
        series = tif.series[0]
        stack = series.asarray()
        # The plan may come from the OME metadata, which this series ignores.
        axes = _series_read_axes(series, plan, file_path)
    stack = _reshape_series_to_tyxc(stack, axes)
    return stack


def _load_stack_from_series(file_path: str) -> Optional[np.ndarray]:
    """
    Load a TIFF stack using ``TiffFile.series[0]`` metadata.

    Uses the axis labels embedded in ImageJ or OME-TIFF metadata (or the
    shape tifffile gives a plain TIFF) to deterministically reshape the data
    to ``(T, Y, X, C)``. What cannot be read (other positions, missing files of
    a multi-file set, Z projected) is reported once with a warning, when the
    page plan of the file is built (see :func:`_get_page_plan`).

    Parameters
    ----------
    file_path : str
        Path to the TIFF file.

    Returns
    -------
    ndarray or None
        Stack with shape ``(T, Y, X, C)``, or ``None`` if the file
        lacks usable series metadata.
    """
    with TiffFile(file_path) as tif:
        if not tif.series:
            return None

        series = tif.series[0]
        axes = series.axes.upper()  # e.g. 'TCYX', 'YX', 'TYX'

        if "Y" not in axes or "X" not in axes:
            return None

        plan = _get_page_plan(file_path, tif=tif)
        # The cached plan may have been built without the OME metadata.
        axes = _series_read_axes(series, plan, file_path)
        stack = series.asarray()

    return _reshape_series_to_tyxc(stack, axes)


def _to_tzc(stack: Any, axes: str) -> Tuple[Any, str, List[Tuple[str, int]]]:
    """
    Reorder an array given in ``axes`` order to ``T, Z, C`` followed by the
    page axes (``Y``, ``X``, ``S``) it has, in their order.

    This is the one rule giving each axis its role, shared by the readers of
    whole stacks and by the page plan of :func:`load_frames`, so that they
    agree on what a frame and a channel are:

    - time is the ``T`` axis; without one, the Z slices are taken as frames,
      and without either, the first generic axis (``Q``, ``I``) of a plain
      stack of pages;
    - channels are the ``C`` axis; without one, the next generic axis (a
      plain ``(T, C, Y, X)`` stack). A ``C`` axis with neither time nor Z may
      have been relabelled time before (see :func:`_effective_axes`);
    - ``S`` (the samples of RGB pages) is kept, to be merged into the channels;
    - any other axis is reduced to its first index.

    Works on NumPy and dask arrays.

    Parameters
    ----------
    stack : ndarray or dask.array.Array
        The array in ``axes`` order.
    axes : str
        Its axis labels.

    Returns
    -------
    stack : ndarray or dask.array.Array
        The reordered array.
    order : str
        Its axis labels, ``'TZC'`` followed by its page axes.
    dropped : list of (str, int)
        The axes reduced to their first index, with their size, when larger than 1.
    """

    axes = axes.upper()
    if "T" not in axes and "Z" in axes:
        axes = axes.replace("Z", "T")
    # The generic axes of a plain stack (tifffile labels the leading axes of a
    # TIFF without ImageJ or OME metadata ``Q``), in order: time, then channels.
    # Generic axes of size 1 (tifffile pads a plain array with them) take no role.
    for role in "TC":
        if role not in axes:
            for i, a in enumerate(axes):
                if a in "QI" and stack.shape[i] > 1:
                    axes = axes[:i] + role + axes[i + 1 :]
                    break

    index, kept, dropped = [], "", []
    for i, a in enumerate(axes):
        if a in "TZC" + _PAGE_AXES and a not in kept:
            index.append(slice(None))
            kept += a
        else:
            index.append(0)
            if stack.shape[i] > 1:
                dropped.append((a, int(stack.shape[i])))
    stack = stack[tuple(index)]

    for a in "CZT":
        if a not in kept:
            stack = np.expand_dims(stack, 0)
            kept = a + kept
    order = "TZC" + "".join(a for a in kept if a in _PAGE_AXES)
    perm = [kept.index(a) for a in order]
    if perm != list(range(len(perm))):
        stack = np.transpose(stack, perm)
    return stack, order, dropped


def _reshape_series_to_tyxc(stack: Any, axes: str) -> Any:
    """
    Reshape an array given in ``axes`` order to ``(T, Y, X, C)``.

    Works on both NumPy and dask arrays (every operation used here dispatches
    through NumPy's protocol), so the eager and lazy loaders share identical
    reshape logic. The axes get their role from :func:`_to_tzc`; when the
    movie has both Z and time, each time point is the maximum intensity
    projection of its slices, and the samples of RGB pages are channels.

    Parameters
    ----------
    stack : ndarray or dask.array.Array
        The raw array in ``axes`` order.
    axes : str
        Axis labels, e.g. ``'TCYX'``, ``'YX'``, ``'TYX'``.

    Returns
    -------
    ndarray or dask.array.Array
        Stack reshaped to ``(T, Y, X, C)``.
    """

    stack, order, _ = _to_tzc(stack, axes)
    if stack.shape[1] > 1:
        stack = stack.max(axis=1)
    else:
        stack = stack[:, 0]
    # Now T, C and the page axes.
    page = order[3:]
    perm = [0, 2 + page.index("Y"), 2 + page.index("X"), 1]
    if "S" in page:
        perm.append(2 + page.index("S"))
    stack = np.transpose(stack, perm)
    if "S" in page:
        n_t, n_y, n_x, n_c, n_s = stack.shape
        stack = stack.reshape((n_t, n_y, n_x, n_c * n_s))
    return stack


class _PagePlan:
    """
    Where every image of a movie is stored, for :func:`load_frames`.

    The analysis loops number images as if the channels were interleaved
    within each time point, in one file: image ``t * n_channels + c`` is
    channel ``c`` of frame ``t`` (see :func:`_get_img_num_per_channel`). The
    plan maps that number to the planes holding it, in the order and the
    files the metadata gives, with the roles of :func:`_to_tzc`.

    Attributes
    ----------
    positions : ndarray
        ``(T, Z, C)`` positions of the planes in the series.
    refs : list or None
        The file and page index of each position, ``None`` for a plane of a
        file that could not be read. ``None`` when every position is the page
        of the same index in the file read.
    files : set of str
        The files holding the movie (several for a multi-file OME-TIFF set),
        normalized for comparison.
    n_samples : int
        Samples per page (3 for RGB), each a channel.
    page_axes : str
        The axes of a page, e.g. ``'YX'`` or ``'YXS'``.
    page_shape : tuple
        The shape of a page.
    dtype : numpy.dtype
        The pixel type.
    axes, series_shape : str, tuple
        Axes and shape of the series.
    contiguous : bool
        Whether the series is one uncompressed block, which can be memory-mapped.
    identity : bool
        Whether image ``k`` is page ``k`` of the file: the plan is then not
        needed and pages are read as they always were.
    """

    def __init__(
        self,
        positions,
        refs,
        files,
        n_samples,
        page_axes,
        page_shape,
        dtype,
        axes,
        series_shape,
        contiguous,
    ):
        self.positions = positions
        self.refs = refs
        self.files = files
        self.n_samples = n_samples
        self.page_axes = page_axes
        self.page_shape = page_shape
        self.dtype = dtype
        self.axes = axes
        self.series_shape = series_shape
        self.contiguous = contiguous
        # (path, version) of the config.ini whose channels chose the axes, if any.
        self.config_dep = None
        self.identity = (
            refs is None
            and n_samples == 1
            and positions.shape[1] == 1
            and np.array_equal(positions.reshape(-1), np.arange(positions.size))
        )

    @property
    def n_frames(self) -> int:
        return int(self.positions.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.positions.shape[2] * self.n_samples)

    @property
    def yx_shape(self) -> Tuple[int, int]:
        return tuple(self.page_shape[self.page_axes.index(a)] for a in "YX")


def _file_version(path: str) -> Tuple[int, int]:
    """The (modification time, size) of a file, to tell when it was replaced."""
    stat = os.stat(path)
    return stat.st_mtime_ns, stat.st_size


def _plane_yx(series) -> Optional[Tuple[int, int]]:
    """The image size of a series, or ``None`` if it has no Y and X axes."""
    axes = series.axes.upper()
    if "Y" not in axes or "X" not in axes:
        return None
    return series.shape[axes.index("Y")], series.shape[axes.index("X")]


def _optional_file_version(path: Optional[str]) -> Optional[Tuple[int, int]]:
    """The version of a file (see :func:`_file_version`), ``None`` if it does not exist."""
    if path is None:
        return None
    try:
        return _file_version(path)
    except OSError:
        return None


def _experiment_config_path(stack_path: str) -> Optional[str]:
    """
    The ``config.ini`` the experiment holding the movie at ``stack_path``
    (``<experiment>/<well>/<position>/movie/<file>``) would have, whether it
    exists or not, or ``None`` when the path is too short to be in an experiment.
    """

    from celldetective.utils.experiment import extract_experiment_from_position

    position = os.path.dirname(os.path.dirname(os.path.abspath(stack_path)))
    experiment = extract_experiment_from_position(position)
    if not experiment:
        # Not three folders deep: a bare "config.ini" would be looked up in the
        # working directory.
        return None
    return os.path.join(experiment, "config.ini")


def _experiment_channel_count(stack_path: str) -> Optional[int]:
    """
    The number of channels the experiment declares for the movie at
    ``stack_path`` (``<experiment>/<well>/<position>/movie/<file>``), counted
    as the analysis loops count them, or ``None`` when the movie is not in an
    experiment with a ``config.ini``, or its channels cannot be read.
    """

    from celldetective.utils.parsing import _extract_nbr_channels_from_config

    config = _experiment_config_path(stack_path)
    if config is None or not os.path.isfile(config):
        return None
    try:
        n_channels = _extract_nbr_channels_from_config(config)
    except Exception as e:
        logger.debug(f"Could not read the channels of {config}: {e}")
        return None
    # No channel declared: unknown, not "no channel" (every C axis would be time).
    return n_channels or None


def _channels_ambiguous(axes: str) -> bool:
    """Whether a ``C`` axis may be time (see :func:`_effective_axes`)."""
    axes = axes.upper()
    return "C" in axes and "T" not in axes and "Z" not in axes


def _series_read_axes(series, plan: Optional["_PagePlan"], stack_path: str) -> str:
    """
    The axes to read ``series`` with: those of the page plan when it was built
    from this series, else its own, with the channel rule of :func:`_effective_axes`.
    """

    if plan is not None and tuple(plan.series_shape) == tuple(series.shape):
        return plan.axes
    axes = series.axes.upper()
    n_channels = (
        _experiment_channel_count(stack_path) if _channels_ambiguous(axes) else None
    )
    return _effective_axes(axes, tuple(series.shape), n_channels)


def _effective_axes(axes: str, shape: tuple, n_channels: Optional[int]) -> str:
    """
    The axes of a series, with a ``C`` axis read as time when the movie has
    neither time nor Z.

    ``tifffile`` writes a 3D OME-TIFF without axes as ``CYX``, so many
    single-channel movies (those saved by earlier versions of celldetective
    among them) carry their frames as channels. Without a time or Z axis, the
    ``C`` axis is taken for time when its size differs from the number of
    channels the experiment declares, or, outside an experiment, when it holds
    more than 5 planes.

    Parameters
    ----------
    axes : str
        The axes of the series.
    shape : tuple
        Its shape.
    n_channels : int or None
        The number of channels of the experiment, if known.

    Returns
    -------
    str
        The axes to read the series with.
    """

    axes = axes.upper()
    if not _channels_ambiguous(axes):
        return axes
    size = shape[axes.index("C")]
    if n_channels is not None:
        is_time = size != n_channels
    else:
        is_time = size > 5
    return axes.replace("C", "T") if is_time else axes


def _path_key(path: str) -> str:
    """
    One spelling of ``path`` for comparisons: absolute, links and Windows short
    names (``RUNNER~1``) resolved, case folded where the file system ignores it.
    tifffile opens the other files of a multi-file set by their real path.
    """

    return os.path.normcase(os.path.realpath(path))


def _build_page_plan(tif: TiffFile, stack_path: str) -> Optional[_PagePlan]:
    """
    Read the page plan of a stack from its first series, or ``None`` if it has
    no usable series.

    What the loaders do not read is logged as a warning here, so once per
    version of the file: the other positions of the file (series with the same
    image size), the planes of a multi-file set whose file is missing (read as
    zeros), the Z slices projected because the movie also has time, and the
    extra axes of which only the first index is read.
    """

    if not tif.series:
        return None
    series = tif.series[0]
    axes = series.axes.upper()
    if "Y" not in axes or "X" not in axes:
        return None
    shape = tuple(series.shape)
    # The experiment's channels are read, and the plan depends on its config.ini,
    # only when a C axis may be time.
    config_dep = None
    n_channels = None
    if _channels_ambiguous(axes):
        config = _experiment_config_path(stack_path)
        if config is not None:
            config_dep = (config, _optional_file_version(config))
            n_channels = _experiment_channel_count(stack_path)
    effective = _effective_axes(axes, shape, n_channels)
    if effective != axes:
        logger.info(
            f"{os.path.basename(stack_path)}: the {shape[axes.index('C')]} channels "
            f"of the file, with no time axis, are read as frames."
        )
        axes = effective
    plane_axes = "".join(a for a in axes if a not in _PAGE_AXES)
    plane_shape = tuple(n for a, n in zip(axes, shape) if a not in _PAGE_AXES)
    page_axes = "".join(a for a in axes if a in _PAGE_AXES)
    page_shape = tuple(n for a, n in zip(axes, shape) if a in _PAGE_AXES)
    n_planes = int(math.prod(plane_shape))

    this = _path_key(stack_path)
    files = {this}
    refs = None
    missing = 0
    contiguous = getattr(series, "dataoffset", None) is not None
    if tif.is_ome or not contiguous:
        # The series may take its planes from other files (multi-file OME and
        # NDTiff sets), in another order than the file's (NDTiff), or skip some
        # (missing files, an aborted acquisition, tiles that were not imaged).
        pages = list(series.pages)
        if len(pages) == n_planes:
            refs = [
                None
                if p is None
                else (_path_key(p.parent.filehandle.path), int(p.index))
                for p in pages
            ]
            missing = sum(r is None for r in refs)
            files.update(r[0] for r in refs if r is not None)
            if not missing and all(r == (this, i) for i, r in enumerate(refs)):
                refs = None

    positions = np.arange(n_planes).reshape(plane_shape)
    positions, _, dropped = _to_tzc(positions, plane_axes)
    plan = _PagePlan(
        np.asarray(positions),
        refs,
        files,
        shape[axes.index("S")] if "S" in axes else 1,
        page_axes,
        page_shape,
        series.dtype,
        axes,
        shape,
        contiguous,
    )
    plan.config_dep = config_dep

    name = os.path.basename(stack_path)
    n_positions = 1 + sum(_plane_yx(s) == _plane_yx(series) for s in tif.series[1:])
    if n_positions > 1:
        logger.warning(
            f"{name}: the file holds {n_positions} positions (series); only the "
            f"first is read. Export one file per position."
        )
    if missing:
        where = (
            "in files that could not be read"
            if tif.is_ome
            else "not in the files (an acquisition stopped early, or tiles not imaged)"
        )
        logger.warning(
            f"{name}: {missing} of the {n_planes} planes of the movie are {where}, "
            f"and are read as zeros. Check that every file of a multi-file set is "
            f"in the movie folder."
        )
    n_t, n_z, _ = plan.positions.shape
    if n_t > 1 and n_z > 1:
        logger.warning(
            f"{name}: the movie has {n_t} time points of {n_z} Z slices; each time "
            f"point is read as the maximum intensity projection of its slices."
        )
    for axis, size in dropped:
        logger.warning(
            f"{name}: axis {axis} of size {size} is not supported; only its first "
            f"index is read."
        )
    return plan


def _get_page_plan(
    stack_path: str, tif: Optional[TiffFile] = None
) -> Optional[_PagePlan]:
    """
    The page plan of a stack, built once per version of the file.

    Parameters
    ----------
    stack_path : str
        Path to the stack.
    tif : TiffFile, optional
        The stack, already open, to build the plan from without opening it again.
    """

    path = os.path.abspath(stack_path)
    version = _file_version(path)
    cached = _PAGE_PLANS.get(path)
    if cached is not None and cached[0] == version:
        dep = getattr(cached[1], "config_dep", None)
        # A plan whose axes follow the channels of the experiment is rebuilt
        # when its config.ini changes.
        if dep is None or _optional_file_version(dep[0]) == dep[1]:
            return cached[1]
    try:
        if tif is None:
            with TiffFile(stack_path) as opened:
                plan = _build_page_plan(opened, stack_path)
        else:
            plan = _build_page_plan(tif, stack_path)
    except Exception as e:
        plan = None
        try:
            # OME metadata tifffile cannot parse (e.g. the companion file of a
            # multi-file set without plane counts): the pages, in order.
            with TiffFile(stack_path, is_ome=False) as opened:
                plan = _build_page_plan(opened, stack_path)
            logger.warning(
                f"{os.path.basename(stack_path)}: the metadata of the file cannot be "
                f"read ({e}); its pages are read in order, as frames."
            )
        except Exception as e2:
            logger.debug(f"No page plan for {stack_path}, pages read by index: {e2}")
    _PAGE_PLANS.pop(path, None)
    while len(_PAGE_PLANS) >= _MAX_PAGE_PLANS:
        try:
            _PAGE_PLANS.pop(next(iter(_PAGE_PLANS)), None)
        except (StopIteration, RuntimeError):
            break
    _PAGE_PLANS[path] = (version, plan)
    return plan


def _read_pages(stack_path: str, img_nums: Union[int, List[int]]) -> np.ndarray:
    """
    Read pages by index from a TIFF file.

    Reads with ``tifffile`` by ``key``, which indexes the pages of the file
    itself (``imageio`` indexes the series of an OME-TIFF instead, which for a
    file of a multi-file set starts with the planes of the other files, and
    fails on OME metadata tifffile cannot parse). For a Fiji hyperstack over
    4 GB, stored as one IFD followed by the pixels, the read by page index
    fails past the first page and the pages are read from a memory map instead
    (the page shape of such a file is cached, see ``_MEMMAP_PAGE_SHAPES``).
    """

    path = os.path.abspath(stack_path)
    cached = _MEMMAP_PAGE_SHAPES.get(path)
    if cached is not None:
        # Only a file read by memory map before is stat-ed, to see it was not
        # replaced since.
        if cached[0] != _file_version(path):
            del _MEMMAP_PAGE_SHAPES[path]
            cached = None
    if cached is not None:
        return _read_pages_from_memmap(stack_path, img_nums, cached[1])
    try:
        return imread(stack_path, key=img_nums)
    except IndexError:
        with TiffFile(stack_path) as tif:
            page_shape = tif.pages.first.shape
        frames = _read_pages_from_memmap(stack_path, img_nums, page_shape)
        _MEMMAP_PAGE_SHAPES[path] = (_file_version(path), page_shape)
        return frames


def _read_images(
    stack_path: str, plan: _PagePlan, img_nums: Union[int, List[int]]
) -> np.ndarray:
    """
    Read images by their number in the loops (see :class:`_PagePlan`).

    Parameters
    ----------
    stack_path : str
        The file the plan was built from.
    plan : _PagePlan
        Its page plan.
    img_nums : int or list of int
        Image numbers; ``-1`` gives a black image (a channel the movie lacks).

    Returns
    -------
    ndarray
        ``(Y, X)`` for a single number, ``(N, Y, X)`` for a list.

    Raises
    ------
    IndexError
        If a number is outside the movie.
    """

    single = not isinstance(img_nums, (list, tuple))
    nums = [img_nums] if single else list(img_nums)

    # (positions, sample) of the planes of each image: the Z slices of a time point.
    wanted = []
    for k in nums:
        k = int(k)
        if k == -1:
            wanted.append(None)
            continue
        if k < 0 or k >= plan.n_frames * plan.n_channels:
            raise IndexError(
                f"image {k} is outside the movie ({plan.n_frames} frames of "
                f"{plan.n_channels} channels)"
            )
        t, ch = divmod(k, plan.n_channels)
        c, s = divmod(ch, plan.n_samples)
        wanted.append(([int(p) for p in plan.positions[t, :, c]], s))

    # Read every plane needed once, grouped by file. A plane of a missing file
    # (reported when the plan was built) is left out, and read as zeros.
    needed = sorted({p for w in wanted if w is not None for p in w[0]})
    by_file: Dict[str, List[Tuple[int, int]]] = {}
    for p in needed:
        ref = (stack_path, p) if plan.refs is None else plan.refs[p]
        if ref is not None:
            by_file.setdefault(ref[0], []).append((p, ref[1]))
    planes = {}
    with _DECODE_LOCK:
        for file, items in by_file.items():
            data = np.asarray(_read_pages(file, [ifd for _, ifd in items]))
            if data.ndim == len(plan.page_shape):
                data = data[np.newaxis]
            for (p, _), plane in zip(items, data):
                planes[p] = plane

    has_samples = "S" in plan.page_axes
    to_yxs = [plan.page_axes.index(a) for a in ("YXS" if has_samples else "YX")]
    black = np.zeros(plan.yx_shape, dtype=plan.dtype)
    images = []
    for w in wanted:
        if w is None:
            images.append(black)
            continue
        slices = []
        for p in w[0]:
            if p not in planes:
                slices.append(black)
                continue
            plane = np.transpose(planes[p], to_yxs)
            slices.append(plane[..., w[1]] if has_samples else plane)
        images.append(np.maximum.reduce(slices))

    return images[0] if single else np.stack(images)


def locate_stack_lazy(position: str, prefix: str = "Aligned") -> Optional[Any]:
    """
    Lazily load the movie as a dask array shaped ``(T, Y, X, C)``.

    Only the frames actually displayed are read from disk — dramatically
    reducing the time and memory needed to open the viewer on large movies.
    An uncompressed, contiguously stored TIFF is memory-mapped; any other
    (compressed, written by Bio-Formats with an IFD between planes, or a
    multi-file OME-TIFF set) is read frame by frame through the page plan of
    :func:`load_frames`. The axis handling is identical to the eager loader.

    Parameters
    ----------
    position : str
        The position folder containing the ``movie`` subdirectory.
    prefix : str, optional
        The movie filename prefix. Default is ``'Aligned'``.

    Returns
    -------
    dask.array.Array or None
        The lazy stack, or ``None`` if lazy loading is unavailable or fails
        (the caller should fall back to :func:`locate_stack`).
    """

    stack_path = get_position_movie_path(position, prefix)
    if stack_path is None:
        return None
    file_path = stack_path.replace("\\", "/")

    try:
        import dask.array as da

        plan = _get_page_plan(file_path)
        if plan is None:
            return None

        if plan.contiguous:
            try:
                mm = memmap(file_path)
                if tuple(mm.shape) == plan.series_shape:
                    # One chunk per leading-axis slice keeps per-frame reads lazy.
                    chunks = (1,) + tuple(mm.shape[1:]) if mm.ndim > 1 else mm.shape
                    stack = _reshape_series_to_tyxc(
                        da.from_array(mm, chunks=chunks), plan.axes
                    )
                    if stack.ndim == 4:
                        return stack
            except Exception as e:
                logger.debug(f"No memory map of {file_path}, reading by frame: {e}")

        n_channels = plan.n_channels

        def read_frame(block_info=None):
            t = block_info[None]["chunk-location"][0]
            images = _read_images(
                file_path, plan, list(range(t * n_channels, (t + 1) * n_channels))
            )
            return np.moveaxis(images, 0, -1)[np.newaxis]

        y, x = plan.yx_shape
        return da.map_blocks(
            read_frame,
            chunks=((1,) * plan.n_frames, (y,), (x,), (n_channels,)),
            dtype=plan.dtype,
            meta=np.empty((0, 0, 0, 0), dtype=plan.dtype),
        )
    except Exception as e:
        logger.debug(f"Lazy stack loading unavailable, falling back to eager: {e}")
        return None


def locate_labels(
    position: str,
    population: str = "target",
    frames: Optional[Union[int, List[int], np.ndarray]] = None,
    threads: int = 4,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> Union[np.ndarray, List[Optional[np.ndarray]], None]:
    """
    Locate and load label images for a given position and population in an experiment.

    This function retrieves and optionally loads labeled images (e.g., targets or effectors)
    for a specified position in an experiment. It supports loading all frames, a specific
    frame, or a list of frames.

    Parameters
    ----------
    position : str
            Path to the position directory containing label images.
    population : str, optional
            The population to load labels for. Options are `'target'` (or `'targets'`) and
            `'effector'` (or `'effectors'`). Default is `'target'`.
    frames : int, list of int, numpy.ndarray, or None, optional
            Specifies which frames to load:
            - `None`: Load all frames (default).
            - `int`: Load a single frame, identified by its index.
            - `list` or `numpy.ndarray`: Load multiple specific frames.
    threads : int, optional
            Number of worker threads used to read all frames in parallel when
            ``frames is None``. Values ``> 1`` enable concurrent I/O. The default
            is `4`. Output ordering is preserved regardless of thread count.
    progress_callback : callable, optional
            Called with an int percentage (0-100) after each frame is read when
            ``frames is None``. The default is None.

    Returns
    -------
    numpy.ndarray or list of numpy.ndarray or None
            If `frames` is `None` or a single integer, returns a NumPy array of the corresponding
            labels. If `frames` is a list or array, returns a list of NumPy arrays for each frame.
            If a frame is not found, `None` is returned for that frame.

    Notes
    -----
    - The function assumes label images are stored in subdirectories named `"labels_targets"`
      or `"labels_effectors"`, with filenames formatted as `####.tif` (e.g., `0001.tif`).
    - Frame indices are zero-padded to four digits for matching.
    - If `frames` is invalid or a frame is not found, `None` is returned for that frame.

    Examples
    --------
    Load all label images for a position:

    >>> labels = locate_labels("/path/to/position", population="target")

    Load a single frame (frame index 3):

    >>> label = locate_labels("/path/to/position", population="effector", frames=3)

    Load multiple specific frames:

    >>> labels = locate_labels("/path/to/position", population="target", frames=[0, 1, 2])

    """

    if not position.endswith(os.sep):
        position += os.sep

    if population.lower() == "target" or population.lower() == "targets":
        label_path = natsorted(
            glob(position + os.sep.join(["labels_targets", "*.tif"]))
        )
    elif population.lower() == "effector" or population.lower() == "effectors":
        label_path = natsorted(
            glob(position + os.sep.join(["labels_effectors", "*.tif"]))
        )
    else:
        label_path = natsorted(
            glob(position + os.sep.join([f"labels_{population}", "*.tif"]))
        )

    label_names = [os.path.split(lbl)[-1] for lbl in label_path]
    name_to_idx = {name: i for i, name in enumerate(label_names)}

    if frames is None:

        def _read_label(path: str) -> np.ndarray:
            return imread(path.replace("\\", "/"))

        n = len(label_path)
        if n == 0:
            labels = np.array([])
        elif threads and threads > 1 and n > 1:
            # Parallel I/O: tifffile decompression releases the GIL, so reading
            # the per-frame masks concurrently is much faster than serially.
            results: List[Optional[np.ndarray]] = []
            with ThreadPoolExecutor(max_workers=threads) as ex:
                for idx, arr in enumerate(ex.map(_read_label, label_path)):
                    results.append(arr)
                    # A False return signals cancellation: stop consuming results.
                    if progress_callback and progress_callback(int((idx + 1) / n * 100)) is False:
                        return None
            labels = np.array(results)
        else:
            results = []
            for idx, path in enumerate(label_path):
                results.append(_read_label(path))
                if progress_callback and progress_callback(int((idx + 1) / n * 100)) is False:
                    return None
            labels = np.array(results)

    elif isinstance(frames, (int, float, np.int_)):

        tzfill = str(int(frames)).zfill(4)
        idx = name_to_idx.get(f"{tzfill}.tif", -1)

        if idx == -1:
            labels = None
        else:
            labels = np.array(imread(label_path[idx].replace("\\", "/")))

    elif isinstance(frames, (list, np.ndarray)):
        labels = []
        for f in frames:
            tzfill = str(int(f)).zfill(4)
            idx = name_to_idx.get(f"{tzfill}.tif", -1)

            if idx == -1:
                labels.append(None)
            else:
                labels.append(np.array(imread(label_path[idx].replace("\\", "/"))))
    else:
        logger.warning("Frames argument must be None, int or list.")
        labels = None

    return labels


def locate_stack_and_labels(
    position: str, prefix: str = "Aligned", population: str = "target"
) -> Tuple[np.ndarray, Union[np.ndarray, List[Optional[np.ndarray]], None]]:
    """

    Locate and load the stack and corresponding segmentation labels.

    Parameters
    ----------
    position : str
            The position or directory path where the stack and labels are located.
    prefix : str, optional
            The prefix used to identify the stack. The default is 'Aligned'.
    population : str, optional
            The population for which the segmentation must be located. The default is 'target'.

    Returns
    -------
    stack : ndarray
            The loaded stack as a NumPy array.
    labels : ndarray
            The loaded segmentation labels as a NumPy array.

    Raises
    ------
    AssertionError
            If no stack with the specified prefix is found or if the shape of the stack and labels do not match.

    Notes
    -----
    This function locates the stack and corresponding segmentation labels based on the specified position and population.
    It assumes that the stack and labels are stored in separate directories: 'movie' for the stack and 'labels' or 'labels_effectors' for the labels.
    The function loads the stack and labels as NumPy arrays and performs shape validation.

    Examples
    --------
    >>> stack, labels = locate_stack_and_labels(position, prefix='Aligned', population="target")
    # Locate and load the stack and segmentation labels for further processing.

    """

    position = position.replace("\\", "/")
    labels = locate_labels(position, population=population)
    stack = locate_stack(position, prefix=prefix)
    if len(labels) < len(stack):
        fix_missing_labels(position, population=population, prefix=prefix)
        labels = locate_labels(position, population=population)
    if len(stack) != len(labels):
        raise ValueError(f"The shape of the stack {stack.shape} does not match with the shape of the labels {labels.shape}")

    return stack, labels


def auto_load_number_of_frames(stack_path: str) -> Optional[int]:
    """
    Automatically determine the number of frames in a TIFF image stack.

    The frames are counted with the rule every loader uses (see :func:`_to_tzc`),
    from the axes tifffile reads in the ImageJ or OME-TIFF metadata, or the shape of a
    plain stack of pages.

    Parameters
    ----------
    stack_path : str
            Path to the TIFF image stack file.

    Returns
    -------
    int or None
            The number of frames in the image stack. Returns `None` if the path is `None`
            or the frame count cannot be determined.

    Notes
    -----
    - The frame count is the ``T`` axis of the first series, else its Z slices, else
      the pages of a plain stack. Channels are never taken for time.
    - When tifffile cannot read a series, ``None`` is returned and the callers use the
      frame count of the experiment.

    Examples
    --------
    Automatically detect the number of frames in a TIFF stack:

    >>> frames = auto_load_number_of_frames("experiment_stack.tif")
    Automatically detected stack length: 120...

    Handle a single-frame TIFF:

    >>> frames = auto_load_number_of_frames("single_frame_stack.tif")
    Automatically detected stack length: 1...

    Handle invalid or missing paths gracefully:

    >>> frames = auto_load_number_of_frames("stack.tif")
    >>> print(frames)
    None

    """

    if stack_path is None:
        return None

    stack_path = stack_path.replace("\\", "/")
    plan = _get_page_plan(stack_path)
    if plan is None:
        logger.info("The stack length could not be detected.")
        return None

    logger.info(f"Automatically detected stack length: {plan.n_frames}...")
    return plan.n_frames


def _require_frames(
    frames: Optional[np.ndarray], indices: Union[List[int], np.ndarray], file: str
) -> np.ndarray:
    """
    Return ``frames``, or raise a clear error if :func:`load_frames` could not
    read them (it logs the cause and returns ``None``).

    Parameters
    ----------
    frames : ndarray or None
        What :func:`load_frames` returned.
    indices : list or ndarray
        The image numbers asked for.
    file : str
        The stack they were read from.

    Returns
    -------
    ndarray
        ``frames``.

    Raises
    ------
    ValueError
        If ``frames`` is ``None``.
    """

    if frames is None:
        raise ValueError(
            f"Could not read the images {np.asarray(indices).tolist()} of {file}. "
            f"Check that the channels and the number of frames of the experiment "
            f"match the movie."
        )
    return frames


def _load_raw_frames(file: str, indices: Union[List[int], np.ndarray]) -> np.ndarray:
    """
    The frames ``indices`` of ``file`` as stored (no scaling, no normalization),
    shape ``(Y, X, len(indices))``.

    Raises
    ------
    ValueError
        If the images cannot be read.
    """

    frames = load_frames(indices, file, scale=None, normalize_input=False)
    return _require_frames(frames, indices, file)


def _load_frames_to_segment(
    file: str,
    indices: Union[List[int], np.ndarray],
    scale_model: Optional[float] = None,
    normalize_kwargs: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Load frames for segmentation.

    Parameters
    ----------
    file : str
        Path to the image stack.
    indices : list or ndarray
        Indices of frames to load.
    scale_model : float, optional
        Scaling factor for the model. Default is None.
    normalize_kwargs : dict, optional
        Normalization keyword arguments. Default is None.

    Returns
    -------
    ndarray
        Loaded and preprocessed frames.

    Raises
    ------
    ValueError
        If the images cannot be read.
    """

    frames = load_frames(
        indices,
        file,
        scale=scale_model,
        normalize_input=True,
        normalize_kwargs=normalize_kwargs,
    )
    frames = _require_frames(frames, indices, file)
    frames = interpolate_nan_multichannel(frames)

    if np.any(indices == -1):
        frames[:, :, np.where(indices == -1)[0]] = 0.0

    return frames


def _load_frames_to_measure(
    file: str, indices: Union[List[int], np.ndarray]
) -> Optional[np.ndarray]:
    """
    Load frames for measurement.

    Parameters
    ----------
    file : str
        Path to the image stack.
    indices : list or ndarray
        Indices of frames to load.

    Returns
    -------
    ndarray
        Loaded frames.
    """
    return load_frames(indices, file, scale=None, normalize_input=False)


def _read_pages_from_memmap(
    stack_path: str, img_nums: Union[int, List[int]], page_shape: tuple
) -> np.ndarray:
    """
    Read images by page index from a memory map of the first series.

    Fiji saves a hyperstack larger than 4 GB with a single IFD and the pixels
    stored contiguously after it, so the file has one page where the movie has
    hundreds of images, and reading by page index fails past the first one.
    tifffile rebuilds the series from the ImageJ metadata; flattening every axis
    before the image plane gives back the page order (channels interleaved).

    Parameters
    ----------
    stack_path : str
        Path to an uncompressed, contiguously stored TIFF.
    img_nums : int or list of int
        Page index or indices, as for :func:`load_frames`.
    page_shape : tuple
        Shape of the first page.

    Returns
    -------
    ndarray
        ``(Y, X)`` for a single index, ``(N, Y, X)`` for a list, as
        ``tifffile.imread(stack_path, key=img_nums)`` would return them.
    """

    pages = memmap(stack_path).reshape((-1, *page_shape))
    return np.array(pages[img_nums])


def load_frames(
    img_nums: Union[int, List[int], np.ndarray],
    stack_path: str,
    scale: Optional[float] = None,
    normalize_input: bool = True,
    dtype: type = np.float64,
    normalize_kwargs: Dict[str, Any] = {"percentiles": (0.0, 99.99)},
) -> Optional[np.ndarray]:
    """
    Loads and optionally normalizes and rescales specified frames from a stack located at a given path.

    This function reads specified frames from a stack file, applying systematic adjustments to ensure
    the channel axis is last. It supports optional normalization of the input frames and rescaling. An
    artificial pixel modification is applied to frames with uniform values to prevent errors during
    normalization.

    Parameters
    ----------
    img_nums : int or list of int
            The number (or numbers) of the image(s) to load, as given by
            :func:`_get_img_num_per_channel`: image ``t * n_channels + c`` is channel
            ``c`` of frame ``t``. The images are read where the metadata stores them,
            whatever the order of the pages, in the other files of a multi-file
            OME-TIFF set, or as a sample of an RGB page; a frame of a movie with Z and
            time is the maximum intensity projection of its slices, as in
            :func:`locate_stack`.
    stack_path : str
            The file path to the stack from which frames are to be loaded.
    scale : float, optional
            The scaling factor to apply to the frames. If None, no scaling is applied (default is None).
    normalize_input : bool, optional
            Whether to normalize the loaded frames. If True, normalization is applied according to
            `normalize_kwargs` (default is True).
    dtype : data-type, optional
            The desired data-type for the output frames (default is float).
    normalize_kwargs : dict, optional
            Keyword arguments to pass to the normalization function (default is {"percentiles": (0., 99.99)}).

    Returns
    -------
    ndarray or None
            The loaded, and possibly normalized and rescaled, frames as a NumPy array. Returns None if there
            is an error in loading the frames.

    Raises
    ------
    Exception
            Prints an error message if the specified frames cannot be loaded or if there is a mismatch between
            the provided experiment channel information and the stack format.

    Notes
    -----
    - The function uses scikit-image for reading frames and supports multi-frame TIFF stacks.
    - Normalization and scaling are optional and can be customized through function parameters.
    - Uniform frames are returned unchanged. Callers that need contrast (e.g. deep-learning
      segmentation) should apply `_fix_no_contrast` themselves.

    Examples
    --------
    >>> frames = load_frames([0, 1, 2], '/path/to/stack.tif', scale=0.5, normalize_input=True, dtype=np.uint8)
    # Loads the first three frames from '/path/to/stack.tif', normalizes them, rescales by a factor of 0.5,
    # and converts them to uint8 data type.

    """

    try:
        # Serialize the global-state-mutating, non-thread-safe decode path so a
        # background prefetch thread and the main thread never decode concurrently.
        with _DECODE_LOCK:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message=".*MMStack series is missing files.*"
                )
                if isinstance(img_nums, np.ndarray):
                    img_nums = img_nums.tolist()
                plan = _get_page_plan(stack_path)
                if plan is None or plan.identity:
                    # Image k is page k of the file.
                    frames = _read_pages(stack_path, img_nums)
                else:
                    frames = _read_images(stack_path, plan, img_nums)
    except Exception as e:
        logger.error(
            f"Error in loading the frame {img_nums}: {e}. Please check that the experiment channel information is consistent with the movie being read."
        )
        return None
    try:
        if np.any(np.isinf(frames)):
            frames = frames.astype(float)
            frames[np.isinf(frames)] = np.nan
    except Exception as e:
        logger.warning(f"inf check failed: {e}")

    frames = _rearrange_multichannel_frame(frames)

    # Order matters and must match training: load_image_dataset rescales (zoom) to
    # the model's spatial calibration FIRST, then normalize_multichannel is applied.
    # Doing it in this order keeps the intensity statistics the model was trained on
    # consistent at inference whenever a calibration mismatch triggers rescaling.
    if scale is not None:
        frames = zoom_multiframes(frames.astype(float), scale)

    if normalize_input:
        frames = normalize_multichannel(frames.astype(float), **normalize_kwargs)

    # Return the pixels as they are on disk. Consumers that cannot handle a uniform
    # frame (segmentation models, contrast sliders) guard for it themselves; faking
    # contrast here would leak into exported movies, measurements and blank-frame checks.
    return frames  # .astype(dtype)


def _rearrange_multichannel_frame(
    frame: np.ndarray, n_channels: Optional[int] = None
) -> np.ndarray:
    """
    Rearranges the axes of a multi-channel frame to ensure the channel axis is at the end.

    This function standardizes the input frame to ensure that the channel axis (if present)
    is moved to the last position. For 2D frames, it adds a singleton channel axis at the end.

    Parameters
    ----------
    frame : ndarray
            The input frame to be rearranged. Can be 2D or 3D.
            - If 3D, the function identifies the channel axis (assumed to be the axis with the smallest size)
              and moves it to the last position.
            - If 2D, the function adds a singleton channel axis to make it compatible with 3D processing.
    n_channels : int, optional
            Number of channels in the image. If provided, helps identify the channel axis. Default is None.

    Returns
    -------
    ndarray
            The rearranged frame with the channel axis at the end.
            - For 3D frames, the output shape will have the channel axis as the last dimension.
            - For 2D frames, the output will have shape `(H, W, 1)` where `H` and `W` are the height and width of the frame.

    Notes
    -----
    - This function assumes that in a 3D input, the channel axis is the one with the smallest size.
    - For 2D frames, this function ensures compatibility with multi-channel processing pipelines by
      adding a singleton dimension for the channel axis.

    Examples
    --------
    Rearranging a 3D multi-channel frame:
    >>> frame = np.zeros((10, 10, 3))  # Already channel-last
    >>> _rearrange_multichannel_frame(frame).shape
    (10, 10, 3)

    Rearranging a 3D frame with channel axis not at the end:
    >>> frame = np.zeros((3, 10, 10))  # Channel-first
    >>> _rearrange_multichannel_frame(frame).shape
    (10, 10, 3)

    Converting a 2D frame to have a channel axis:
    >>> frame = np.zeros((10, 10))  # Grayscale image
    >>> _rearrange_multichannel_frame(frame).shape
    (10, 10, 1)
    """

    if frame.ndim == 3:
        # Systematically move channel axis to the end
        if n_channels is not None and n_channels in list(frame.shape):
            channel_axis = list(frame.shape).index(n_channels)
        else:
            channel_axis = np.argmin(frame.shape)
        frame = np.moveaxis(frame, channel_axis, -1)

    if frame.ndim == 2:
        frame = frame[:, :, np.newaxis]

    return frame


def zoom_multiframes(frames: np.ndarray, zoom_factor: float) -> np.ndarray:
    """
    Applies zooming to each frame (channel) in a multi-frame image.

    This function resizes each channel of a multi-frame image independently using a specified zoom factor.
    The zoom is applied using spline interpolation of the specified order, and the channels are combined
    back into the original format.

    Parameters
    ----------
    frames : ndarray
            A multi-frame image with dimensions `(height, width, channels)`. The last axis represents different
            channels.
    zoom_factor : float
            The zoom factor to apply to each channel. Values greater than 1 increase the size, and values
            between 0 and 1 decrease the size.

    Returns
    -------
    ndarray
            A new multi-frame image with the same number of channels as the input, but with the height and width
            scaled by the zoom factor.

    Notes
    -----
    - The function uses spline interpolation (order 3) for resizing, which provides smooth results.
    - `prefilter=False` is used to prevent additional filtering during the zoom operation.
    - The function assumes that the input is in `height x width x channels` format, with channels along the
      last axis.
    """

    from scipy.ndimage import zoom

    frames = [
        zoom(
            frames[:, :, c].copy(), [zoom_factor, zoom_factor], order=3, prefilter=False
        )
        for c in range(frames.shape[-1])
    ]
    frames = np.moveaxis(frames, 0, -1)
    return frames


def fix_missing_labels(
    position: str, population: str = "target", prefix: str = "Aligned"
) -> None:
    """
    Fix missing label files by creating empty label images for frames that do not have corresponding label files.

    This function locates missing label files in a sequence of frames and creates empty labels (filled with zeros)
    for the frames that are missing. The function works for two types of populations: 'target' or 'effector'.

    Parameters
    ----------
    position : str
        The file path to the folder containing the images/label files. This is the root directory where
        the label files are expected to be found.
    population : str, optional
        Specifies whether to look for 'target' or 'effector' labels. Accepts 'target' or 'effector'
        as valid values. Default is 'target'.
    prefix : str, optional
        The prefix used to locate the image stack (default is 'Aligned').

    Returns
    -------
    None
        The function creates new label files in the corresponding folder for any frames missing label files.

    """

    if not position.endswith(os.sep):
        position += os.sep

    stack = locate_stack(position, prefix=prefix)
    template = np.zeros((stack[0].shape[0], stack[0].shape[1]), dtype=int)
    all_frames = np.arange(len(stack))

    if population.lower() == "target" or population.lower() == "targets":
        label_path = natsorted(
            glob(position + os.sep.join(["labels_targets", "*.tif"]))
        )
        path = position + os.sep + "labels_targets"
    elif population.lower() == "effector" or population.lower() == "effectors":
        label_path = natsorted(
            glob(position + os.sep.join(["labels_effectors", "*.tif"]))
        )
        path = position + os.sep + "labels_effectors"
    else:
        label_path = natsorted(
            glob(position + os.sep.join([f"labels_{population}", "*.tif"]))
        )
        path = position + os.sep + f"labels_{population}"

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    if label_path:
        # path = os.path.split(label_path[0])[0]
        int_valid = [int(lbl.split(os.sep)[-1].split(".")[0]) for lbl in label_path]
        to_create = [x for x in all_frames if x not in int_valid]
    else:
        to_create = all_frames
    to_create = [str(x).zfill(4) + ".tif" for x in to_create]
    for file in to_create:
        save_tiff_imagej_compatible(
            os.sep.join([path, file]), template.astype(np.int16), axes="YX"
        )
    # imwrite(os.sep.join([path, file]), template.astype(int))


def _get_img_num_per_channel(
    channels_indices: Union[int, List[Optional[int]], np.ndarray],
    len_movie: int,
    nbr_channels: int,
) -> np.ndarray:
    """
    Calculates the image frame numbers for each specified channel in a multi-channel movie.

    Given the indices of channels of interest, the total length of the movie, and the number of channels,
    this function computes the frame numbers corresponding to each channel throughout the movie. If a
    channel index is specified as None, it assigns a placeholder value to indicate no frames for that channel.

    Parameters
    ----------
    channels_indices : list of int or None
            A list containing the indices of channels for which to calculate frame numbers. If an index is None,
            it is interpreted as a channel with no frames to be processed.
    len_movie : int
            The total number of frames in the movie across all channels.
    nbr_channels : int
            The total number of channels in the movie.

    Returns
    -------
    ndarray
            A 2D numpy array where each row corresponds to a channel specified in `channels_indices` and contains
            the frame numbers for that channel throughout the movie. If a channel index is None, the corresponding
            row contains placeholder values (-1).

    Notes
    -----
    - The function assumes that frames in the movie are interleaved by channel, with frames for each channel
      appearing in a regular sequence throughout the movie.
    - This utility is particularly useful for multi-channel time-lapse movies where analysis or processing
      needs to be performed on a per-channel basis.

    Examples
    --------
    >>> channels_indices = [0]  # Indices for channels 1, 3, and a non-existing channel
    >>> len_movie = 10  # Total frames for each channel
    >>> nbr_channels = 3  # Total channels in the movie
    >>> img_num_per_channel = _get_img_num_per_channel(channels_indices, len_movie, nbr_channels)
    >>> print(img_num_per_channel)
    # array([[ 0,  3,  6,  9, 12, 15, 18, 21, 24, 27]])

    >>> channels_indices = [1,2]  # Indices for channels 1, 3, and a non-existing channel
    >>> len_movie = 10  # Total frames for each channel
    >>> nbr_channels = 3  # Total channels in the movie
    >>> img_num_per_channel = _get_img_num_per_channel(channels_indices, len_movie, nbr_channels)
    >>> print(img_num_per_channel)
    # array([[ 1,  4,  7, 10, 13, 16, 19, 22, 25, 28],
    #   [ 2,  5,  8, 11, 14, 17, 20, 23, 26, 29]])

    """

    if isinstance(channels_indices, (int, np.int_)):
        channels_indices = [channels_indices]

    len_movie = int(len_movie)
    nbr_channels = int(nbr_channels)

    img_num_all_channels = []
    for c in channels_indices:
        if c is not None:
            indices = np.arange(len_movie * nbr_channels)[c::nbr_channels]
        else:
            indices = [-1] * len_movie
        img_num_all_channels.append(indices)
    img_num_all_channels = np.array(img_num_all_channels, dtype=int)

    return img_num_all_channels


def _extract_channel_indices(
    channels: Optional[List[str]], required_channels: List[str]
) -> List[Optional[int]]:
    """
    Extracts the indices of required channels from a list of available channels.

    This function is designed to match the channels required by a model or analysis process with the channels
    present in the dataset. It returns the indices of the required channels within the list of available channels.
    If the required channels are not found among the available channels, the function prints an error message and
    returns None.

    Parameters
    ----------
    channels : list of str or None
            A list containing the names of the channels available in the dataset. If None, it is assumed that the
            dataset channels are in the same order as the required channels.
    required_channels : list of str
            A list containing the names of the channels required by the model or analysis process.

    Returns
    -------
    list of int or None
            A list of indices indicating the positions of the required channels within the list of available
            channels. Returns None if there is a mismatch between required and available channels.

    Notes
    -----
    - The function is useful for preprocessing steps where specific channels of multi-channel data are needed
      for further analysis or model input.
    - In cases where `channels` is None, indicating that the dataset does not specify channel names, the function
      assumes that the dataset's channel order matches the order of `required_channels` and returns an array of
      indices based on this assumption.

    Examples
    --------
    >>> available_channels = ['DAPI', 'GFP', 'RFP']
    >>> required_channels = ['GFP', 'RFP']
    >>> indices = _extract_channel_indices(available_channels, required_channels)
    >>> print(indices)
    # [1, 2]

    >>> indices = _extract_channel_indices(None, required_channels)
    >>> print(indices)
    # [0, 1]
    """

    channel_indices = []
    channels_lower = [ch.lower() for ch in channels] if channels is not None else None
    for c in required_channels:
        if c != "None" and c is not None:
            try:
                c_lower = c.lower()
                ch_idx = channels_lower.index(c_lower)
                channel_indices.append(ch_idx)
            except Exception as e:
                channel_indices.append(None)
        else:
            channel_indices.append(None)

    return channel_indices


def load_image_dataset(
    datasets: List[str],
    channels: Union[str, List[str]],
    train_spatial_calibration: Optional[float] = None,
    mask_suffix: str = "labelled",
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """
    Loads image and corresponding mask datasets, optionally applying spatial calibration adjustments.

    This function iterates over specified datasets, loading image and mask pairs based on provided channels
    and adjusting images according to a specified spatial calibration factor. It supports loading images with
    multiple channels and applies necessary transformations to match the training spatial calibration.

    Parameters
    ----------
    datasets : list of str
            A list of paths to the datasets containing the images and masks.
    channels : str or list of str
            The channel(s) to be loaded from the images. If a string is provided, it is converted into a list.
    train_spatial_calibration : float, optional
            The spatial calibration (e.g., micrometers per pixel) used during model training. If provided, images
            will be rescaled to match this calibration. Default is None, indicating no rescaling is applied.
    mask_suffix : str, optional
            The suffix used to identify mask files corresponding to the images. Default is 'labelled'.

    Returns
    -------
    tuple of lists
            A tuple containing three lists: `X` for images, `Y` for corresponding masks, and `files` for the
            original file paths. All lists contain elements corresponding to the loaded samples.

    Raises
    ------
    AssertionError
            If the provided `channels` argument is not a list or if the number of loaded images does not match
            the number of loaded masks.

    Notes
    -----
    - The function assumes that mask filenames are derived from image filenames by appending a `mask_suffix`
      before the file extension.
    - Spatial calibration adjustment involves rescaling the images and masks to match the `train_spatial_calibration`.
    - Only images with a corresponding mask and a valid configuration file specifying channel indices and
      spatial calibration are loaded.
    - The image samples must have at least one channel in common with the required channels to be accepted. The missing
      channels are passed as black frames.

    Examples
    --------
    >>> datasets = ['/path/to/dataset1', '/path/to/dataset2']
    >>> channels = ['DAPI', 'GFP']
    >>> X, Y, files = load_image_dataset(datasets, channels, train_spatial_calibration=0.65)
    # Loads DAPI and GFP channels from specified datasets, rescaling images to match a spatial calibration of 0.65.
    """

    from scipy.ndimage import zoom

    if isinstance(channels, str):
        channels = [channels]

    if not isinstance(channels, list):
        raise TypeError("Please provide a list of channels. Abort.")

    X = []
    Y = []
    files = []

    for ds in datasets:
        logger.info(f"Loading data from dataset {ds}...")
        if not ds.endswith(os.sep):
            ds += os.sep
        img_paths = list(
            set(glob(ds + "*.tif")) - set(glob(ds + f"*_{mask_suffix}.tif"))
        )
        for im in img_paths:
            logger.debug(f"Processing image: {im}")
            mask_path = os.sep.join(
                [
                    os.path.split(im)[0],
                    os.path.split(im)[-1].replace(".tif", f"_{mask_suffix}.tif"),
                ]
            )
            if os.path.exists(mask_path):
                # load image and mask
                image = imread(im)
                if image.ndim == 2:
                    image = image[np.newaxis]
                if image.ndim > 3:
                    logger.warning("Invalid image shape, skipping")
                    continue
                mask = imread(mask_path)
                config_path = im.replace(".tif", ".json")
                if os.path.exists(config_path):
                    # Load config
                    with open(config_path, "r") as f:
                        config = json.load(f)

                    existing_channels = config["channels"]
                    channels_lower = [ch.lower() for ch in channels]
                    existing_channels_lower = [ch.lower() for ch in existing_channels]
                    intersection = list(
                        set(channels_lower) & set(existing_channels_lower)
                    )
                    logger.debug(f"existing_channels={existing_channels}, intersection={intersection}")
                    if len(intersection) == 0:
                        logger.warning(
                            "Channels could not be found in the config... Skipping image."
                        )
                        continue
                    else:
                        ch_idx = []
                        for c in channels:
                            c_lower = c.lower()
                            if c_lower in existing_channels_lower:
                                idx = existing_channels_lower.index(c_lower)
                                ch_idx.append(idx)
                            else:
                                # For None or missing channel pass black frame
                                ch_idx.append(np.nan)
                        im_calib = config["spatial_calibration"]
                else:
                    logger.warning(f"No config file found for {im}, skipping.")
                    continue

                ch_idx = np.array(ch_idx)
                ch_idx_safe = np.copy(ch_idx)
                ch_idx_safe[ch_idx_safe != ch_idx_safe] = 0
                ch_idx_safe = ch_idx_safe.astype(int)

                image = image[ch_idx_safe]
                image[np.where(ch_idx != ch_idx)[0], :, :] = 0

                image = np.moveaxis(image, 0, -1)
                if image.ndim != 3:
                    raise ValueError("The image has a wrong number of dimensions. Abort.")

                if im_calib != train_spatial_calibration:
                    factor = im_calib / train_spatial_calibration
                    image = np.moveaxis(
                        [
                            zoom(
                                image[:, :, c].astype(float).copy(),
                                [factor, factor],
                                order=3,
                                prefilter=False,
                            )
                            for c in range(image.shape[-1])
                        ],
                        0,
                        -1,
                    )  # zoom(image, [factor,factor,1], order=3)
                    mask = zoom(mask, [factor, factor], order=0)

            X.append(image)
            Y.append(mask)

            # fig,ax = plt.subplots(1,image.shape[-1]+1)
            # for k in range(image.shape[-1]):
            # 	ax[k].imshow(image[:,:,k],cmap='gray')
            # ax[image.shape[-1]].imshow(mask)
            # plt.pause(1)
            # plt.close()

            files.append(im)

    if len(X) != len(Y):
        raise ValueError("The number of images does not match with the number of masks... Abort.")
    return X, Y, files
