import threading
import logging
from pathlib import Path
from typing import Union, Optional, Any, List, Dict, Tuple, Callable

import numpy as np

logger = logging.getLogger("celldetective")
import pandas as pd
from skimage.measure import regionprops_table, label
from skimage.transform import resize
from tqdm import tqdm

from celldetective.utils.image_loaders import load_frames
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import find_objects
import concurrent.futures


def fill_label_holes(lbl_img: np.ndarray, **kwargs: Any) -> np.ndarray:
    """
    Fill small holes in label image.
    from https://github.com/stardist/stardist/blob/main/stardist/utils.py

    Parameters
    ----------
    lbl_img : ndarray
        Label image.
    **kwargs : dict
        Additional arguments for `scipy.ndimage.binary_fill_holes`.

    Returns
    -------
    ndarray
        Label image with filled holes.
    """

    def grow(
        sl: Tuple[slice, ...], interior: List[Tuple[bool, bool]]
    ) -> Tuple[slice, ...]:
        """
        Grow slice.

        Parameters
        ----------
        sl : tuple
            Slice tuple.
        interior : list
            List of interior flags.
        """
        return tuple(
            slice(s.start - int(w[0]), s.stop + int(w[1])) for s, w in zip(sl, interior)
        )

    def shrink(interior: List[Tuple[bool, bool]]) -> Tuple[slice, ...]:
        """
        Shrink slice.

        Parameters
        ----------
        interior : list
            List of interior flags.
        """
        return tuple(slice(int(w[0]), (-1 if w[1] else None)) for w in interior)

    objects = find_objects(lbl_img)
    lbl_img_filled = np.zeros_like(lbl_img)
    for i, sl in enumerate(objects, 1):
        if sl is None:
            continue
        interior = [(s.start > 0, s.stop < sz) for s, sz in zip(sl, lbl_img.shape)]
        shrink_slice = shrink(interior)
        grown_mask = lbl_img[grow(sl, interior)] == i
        mask_filled = binary_fill_holes(grown_mask, **kwargs)[shrink_slice]
        lbl_img_filled[sl][mask_filled] = i
    if lbl_img.min() < 0:
        # preserve (and fill holes in) negative labels ('find_objects' ignores these)
        lbl_neg_filled = -fill_label_holes(-np.minimum(lbl_img, 0))
        mask = lbl_neg_filled < 0
        lbl_img_filled[mask] = lbl_neg_filled[mask]
    return lbl_img_filled


def _check_label_dims(
    lbl: np.ndarray,
    file: Optional[Union[str, Path]] = None,
    template: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Check and resize label image to match template dimensions.

    Parameters
    ----------
    lbl : ndarray
        Label image.
    file : str, optional
        Path to the file to load as template. Default is None.
    template : ndarray, optional
        Template image. Default is None.

    Returns
    -------
    ndarray
        Resized label image.
    """

    if file is not None:
        template = load_frames(0, file, scale=1, normalize_input=False)
    elif template is not None:
        template = template
    else:
        return lbl

    if lbl.shape != template.shape[:2]:
        lbl = resize(lbl, template.shape[:2], order=0)
    return lbl


def auto_correct_masks(
    masks: np.ndarray,
    bbox_factor: float = 1.75,
    min_area: int = 9,
    fill_labels: bool = False,
    correct_anomalies: bool = True,
) -> np.ndarray:
    """
    Correct segmentation masks to ensure consistency and remove anomalies.

    This function processes a labeled mask image to correct anomalies and reassign labels.
    It performs the following operations:

    1. Corrects negative mask values by taking their absolute values.
    2. Identifies and corrects segmented objects with a bounding box area that is disproportionately
       larger than the actual object area. This indicates potential segmentation errors where separate objects
       share the same label.
    3. Removes small objects that are considered noise (default threshold is an area of less than 9 pixels).
    4. Reorders the labels so they are consecutive from 1 up to the number of remaining objects (to avoid encoding errors).

    Parameters
    ----------
    masks : np.ndarray
            A 2D array representing the segmented mask image with labeled regions. Each unique value
            in the array represents a different object or cell.
    bbox_factor : float, optional
            A factor on cell area that is compared directly to the bounding box area of the cell, to detect remote cells
            sharing a same label value. The default is `1.75`.
    min_area : int, optional
            Discard cells that have an area smaller than this minimum area (px²). The default is `9` (3x3 pixels).
    fill_labels : bool, optional
            Fill holes within cell masks automatically. The default is `False`.
    correct_anomalies : bool, optional
            Split objects sharing a single label when their bounding box area is
            disproportionately larger than the object area (see ``bbox_factor``).
            The default is `True`.

    Returns
    -------
    clean_labels : np.ndarray
            A corrected version of the input mask, with anomalies corrected, small objects removed,
            and labels reordered to be consecutive integers.

    Notes
    -----
    - This function is useful for post-processing segmentation outputs to ensure high-quality
      object detection, particularly in applications such as cell segmentation in microscopy images.
    - The function assumes that the input masks contain integer labels and that the background
      is represented by 0.

    Examples
    --------
    Surviving labels are renumbered consecutively -- 3 and 7 below become 1
    and 2, which is what keeps downstream label encodings dense:

    >>> masks = np.array([[0, 0, 3, 3],
    ...                   [0, 7, 7, 3],
    ...                   [0, 7, 0, 0]])
    >>> auto_correct_masks(masks, min_area=3)
    array([[0, 0, 1, 1],
           [0, 2, 2, 1],
           [0, 2, 0, 0]])

    Note the explicit ``min_area``: both objects are 3 px, so the default of 9
    treats them as noise and discards them.

    >>> auto_correct_masks(masks)
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]])
    """

    if masks.ndim != 2:
        raise ValueError("`masks` should be a 2D numpy array...")

    # Avoid negative mask values. np.abs returns a fresh array, so the caller's
    # input is never mutated in place.
    masks = np.abs(masks)

    corrected_lbl = masks.copy()  # .astype(int)

    if correct_anomalies:
        props = pd.DataFrame(
            regionprops_table(masks, properties=("label", "area", "area_bbox"))
        )
        max_lbl = int(props["label"].max()) if len(props) else 0

        for row in props.itertuples(index=False):

            if row.area_bbox > bbox_factor * row.area:  # condition for anomaly

                lbl = masks == row.label
                relabelled = label(lbl, connectivity=2)
                relabelled[lbl] += max_lbl
                corrected_lbl[lbl] = relabelled[lbl]
                max_lbl = int(corrected_lbl.max())

    # Second routine: drop objects that are too small and renumber the
    # survivors to consecutive labels (1..N) in a single vectorized pass.
    counts = np.bincount(corrected_lbl.ravel())
    # `counts > 0` as well as the area test: `np.bincount` reports a zero count
    # for every integer below the maximum label, so a `min_area` of 0 or less
    # would otherwise "keep" label values that are absent from the image, make
    # the lookup table an identity, and skip the renumbering entirely. Callers
    # do pass `min_area=0` -- that is what unticking "Remove small objects" sends.
    keep = (counts >= min_area) & (counts > 0)
    keep[0] = False  # background
    kept_labels = np.nonzero(keep)[0]
    lut = np.zeros(counts.size, dtype=int)
    lut[kept_labels] = np.arange(1, kept_labels.size + 1)
    clean_labels = lut[corrected_lbl].astype(int)

    if fill_labels:
        clean_labels = fill_label_holes(clean_labels)

    return clean_labels


def relabel_segmentation(
    labels: np.ndarray,
    df: pd.DataFrame,
    exclude_nans: bool = True,
    column_labels: Dict[str, str] = {
        "track": "TRACK_ID",
        "frame": "FRAME",
        "y": "POSITION_Y",
        "x": "POSITION_X",
        "label": "class_id",
    },
    threads: int = 1,
    progress_callback: Optional[Callable[[float], bool]] = None,
) -> Optional[np.ndarray]:
    """
    Relabel the segmentation labels with the tracking IDs from the tracks.

    The function reassigns the mask value for each cell with the associated `TRACK_ID`, if it exists
    in the trajectory table (`df`). If no track uses the cell mask, a new track with a single point
    is generated on the fly (max of `TRACK_ID` values + i, for i=0 to N such cells). It supports
    multithreaded processing for faster execution on large datasets.

    Parameters
    ----------
    labels : np.ndarray
            A (TYX) array where each frame contains a 2D segmentation mask. Each unique
            non-zero integer represents a labeled object.
    df : pandas.DataFrame
            A DataFrame containing tracking information with columns
            specified in `column_labels`. Must include at least frame, track ID, and object ID.
    exclude_nans : bool, optional
            Whether to exclude rows in `df` with NaN values in the column specified by
            `column_labels['label']`. Default is `True`.
    column_labels : dict, optional
            A dictionary specifying the column names in `df`. Default is:
            - `'track'`: Track ID column name (`"TRACK_ID"`)
            - `'frame'`: Frame column name (`"FRAME"`)
            - `'y'`: Y-coordinate column name (`"POSITION_Y"`)
            - `'x'`: X-coordinate column name (`"POSITION_X"`)
            - `'label'`: Object ID column name (`"class_id"`)
    threads : int, optional
            Number of threads to use for multithreaded processing. Default is `1`.
    progress_callback : callable, optional
            A function to report progress. Should accept a single float argument (0-100) and return True to continue or False to cancel. Default is None.

    Returns
    -------
    np.ndarray
            A new (TYX) array with the same shape as `labels`, where objects are relabeled
            according to their tracking identity in `df`.

    Notes
    -----
    - For frames where labeled objects in `labels` do not match any entries in the `df`,
      new track IDs are generated for the unmatched labels.
    - The relabeling process maintains synchronization across threads using a shared
      counter for generating unique track IDs.

    Examples
    --------
    Relabel segmentation using tracking data:

    >>> labels = np.random.randint(0, 5, (10, 100, 100))
    >>> df = pd.DataFrame({
    ...     "TRACK_ID": [1, 2, 1, 2],
    ...     "FRAME": [0, 0, 1, 1],
    ...     "class_id": [1, 2, 1, 2],
    ... })
    >>> new_labels = relabel_segmentation(labels, df, threads=2)
    Done.

    Use custom column labels and exclude rows with NaNs:

    >>> column_labels = {
    ...     'track': "track_id",
    ...     'frame': "time",
    ...     'label': "object_id"
    ... }
    >>> new_labels = relabel_segmentation(labels, df, column_labels=column_labels, exclude_nans=True)
    Done.

    """

    n_threads = threads
    df = df.sort_values(by=[column_labels["track"], column_labels["frame"]])
    if exclude_nans:
        df = df.dropna(subset=[column_labels["label"]])

    new_labels = np.zeros_like(labels)
    shared_data = {"s": 0}
    label_lock = threading.Lock()  # shared lock guarding `shared_data["s"]`

    # Pre-group the trajectory table by frame once, so the worker loop does a
    # dict lookup per frame instead of a full-DataFrame scan.
    frame_col = column_labels["frame"]
    cells_by_frame = {
        int(f): g[[column_labels["track"], column_labels["label"]]].to_numpy()
        for f, g in df.dropna(subset=[frame_col]).groupby(frame_col)
    }

    # Progress tracking
    shared_progress = {"val": 0, "lock": threading.Lock()}
    total_frames = len(cells_by_frame)

    def rewrite_labels(indices: List[int]) -> None:
        """
        Rewrite labels for a batch of frames.

        Parameters
        ----------
        indices : list
            List of frame indices to process.
        """

        all_track_ids = df[column_labels["track"]].dropna().unique()
        max_track_id = max(all_track_ids) if len(all_track_ids) else 0

        # Check for cancellation
        if progress_callback:
            with shared_progress["lock"]:
                if shared_progress.get("cancelled", False):
                    return

        disable_tqdm = progress_callback is not None

        for t in tqdm(indices, disable=disable_tqdm):

            # Cancellation check inside loop
            if progress_callback:
                with shared_progress["lock"]:
                    if shared_progress.get("cancelled", False):
                        return

                    shared_progress["val"] += 1
                    p = int((shared_progress["val"] / total_frames) * 100)

                if not progress_callback(p):
                    with shared_progress["lock"]:
                        shared_progress["cancelled"] = True
                    return

            f = int(t)
            cells = cells_by_frame.get(f, np.empty((0, 2)))
            tracks_at_t = list(cells[:, 0])
            identities = list(cells[:, 1])

            labels_at_t = list(np.unique(labels[f]))
            if 0 in labels_at_t:
                labels_at_t.remove(0)
            labels_not_in_df = [lbl for lbl in labels_at_t if lbl not in identities]
            for lbl in labels_not_in_df:
                with label_lock:  # Synchronize access to `shared_data["s"]`
                    track_id = max_track_id + shared_data["s"]
                    shared_data["s"] += 1
                tracks_at_t.append(track_id)
                identities.append(lbl)

            # Drop NaN identities/tracks, then rewrite the whole frame in a
            # single lookup-table pass (label value -> track id).
            tracks_at_t = np.array(tracks_at_t)
            identities = np.array(identities)
            valid = (identities == identities) & (tracks_at_t == tracks_at_t)
            identities = identities[valid]
            tracks_at_t = tracks_at_t[valid]

            if identities.size:
                frame = labels[f]
                max_lbl = int(frame.max())
                ids_int = identities.astype(np.int64)
                in_range = (ids_int >= 0) & (ids_int <= max_lbl)
                lut = np.zeros(max_lbl + 1, dtype=new_labels.dtype)
                lut[ids_int[in_range]] = np.round(tracks_at_t[in_range]).astype(
                    new_labels.dtype
                )
                new_labels[f] = lut[frame]

    # Multithreading
    indices = list(df[column_labels["frame"]].dropna().unique())
    chunks = np.array_split(indices, n_threads)

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:

        results = executor.map(
            rewrite_labels, chunks
        )  # list(map(lambda x: executor.submit(self.parallel_job, x), chunks))
        try:
            for i, return_value in enumerate(results):
                # print(f"Thread {i} output check: ", return_value)
                pass
        except Exception as e:
            logger.error(f"Thread exception in relabeling: {e}")

    if shared_progress.get("cancelled", False):
        logger.info("Relabeling cancelled.")
        return None

    logger.info("Relabeling done.")

    return new_labels
