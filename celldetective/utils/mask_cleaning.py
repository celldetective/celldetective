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
    >>> masks = np.array([[0, 0, 1, 1], [0, 2, 2, 1], [0, 2, 0, 0]])
    >>> corrected_masks = auto_correct_masks(masks)
    >>> corrected_masks
    array([[0, 0, 1, 1],
               [0, 2, 2, 1],
               [0, 2, 0, 0]])
    """

    if masks.ndim != 2:
        raise ValueError("`masks` should be a 2D numpy array...")

    # Work on a copy so we never mutate the caller's array; np.abs also folds in
    # the previous negative-value correction.
    masks = np.abs(masks)

    props = pd.DataFrame(
        regionprops_table(masks, properties=("label", "area", "area_bbox"))
    )
    max_lbl = props["label"].max()
    corrected_lbl = masks.copy()  # .astype(int)

    for cell in props["label"].unique():

        bbox_area = props.loc[props["label"] == cell, "area_bbox"].values
        area = props.loc[props["label"] == cell, "area"].values

        if bbox_area > bbox_factor * area:  # condition for anomaly

            lbl = masks == cell
            lbl = lbl.astype(int)

            relabelled = label(lbl, connectivity=2)
            relabelled += max_lbl
            relabelled[np.where(lbl == 0)] = 0

            corrected_lbl[np.where(relabelled != 0)] = relabelled[
                np.where(relabelled != 0)
            ]

        max_lbl = np.amax(corrected_lbl)

    # Second routine to eliminate objects too small (vectorized: collect every
    # under-sized label and zero them in one np.isin pass).
    props2 = pd.DataFrame(
        regionprops_table(corrected_lbl, properties=("label", "area"))
    )
    small_labels = props2.loc[props2["area"] < min_area, "label"].to_numpy()
    if small_labels.size:
        corrected_lbl[np.isin(corrected_lbl, small_labels)] = 0

    # Reorder labels from 1..N via a lookup table instead of one masked
    # assignment per label.
    label_ids = np.unique(corrected_lbl)
    label_ids = label_ids[label_ids != 0]
    lut = np.zeros(int(corrected_lbl.max()) + 1, dtype=int)
    lut[label_ids] = np.arange(1, label_ids.size + 1)
    clean_labels = lut[corrected_lbl]

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

    # int32 output: track IDs routinely exceed the int16 range of the on-disk
    # masks, which would otherwise silently overflow.
    new_labels = np.zeros(labels.shape, dtype=np.int32)
    shared_data = {"s": 0}
    counter_lock = threading.Lock()  # protects shared_data["s"] across threads

    # Progress tracking
    shared_progress = {"val": 0, "lock": threading.Lock()}
    total_frames = len(df[column_labels["frame"]].dropna().unique())

    # Base value for fresh IDs given to masks that are not in the table.
    all_track_ids = df[column_labels["track"]].dropna().to_numpy()
    base_track_id = int(np.max(all_track_ids)) if all_track_ids.size else 0

    def rewrite_labels(indices: List[int]) -> None:
        """
        Rewrite labels for a batch of frames.

        Parameters
        ----------
        indices : list
            List of frame indices to process.
        """

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
            frame_lbl = labels[f]
            max_lbl = int(frame_lbl.max())
            if max_lbl <= 0:
                continue

            # (track_id, class_id) pairs for this frame; cast to float so NaNs
            # survive the comparison regardless of the source dtype.
            cells = df.loc[
                df[column_labels["frame"]] == f,
                [column_labels["track"], column_labels["label"]],
            ].to_numpy(dtype=float)

            if cells.size:
                valid = ~(np.isnan(cells[:, 0]) | np.isnan(cells[:, 1]))
                tracked_class = cells[valid, 1].astype(np.int64)
                tracked_track = np.rint(cells[valid, 0]).astype(np.int64)
            else:
                tracked_class = np.empty(0, dtype=np.int64)
                tracked_track = np.empty(0, dtype=np.int64)

            # Build a lookup table mapping each mask value (class_id) to its
            # track ID, then apply it to the whole frame in one vectorized pass.
            lut = np.zeros(max_lbl + 1, dtype=np.int32)
            in_range = (tracked_class > 0) & (tracked_class <= max_lbl)
            lut[tracked_class[in_range]] = tracked_track[in_range].astype(np.int32)

            # Masks present in the frame but absent from the table get fresh IDs.
            present = np.unique(frame_lbl)
            present = present[present != 0]
            tracked_set = set(tracked_class.tolist())
            untracked = [int(lbl) for lbl in present.tolist() if int(lbl) not in tracked_set]
            if untracked:
                with counter_lock:
                    for lbl in untracked:
                        shared_data["s"] += 1
                        if lbl <= max_lbl:
                            lut[lbl] = base_track_id + shared_data["s"]

            new_labels[f] = lut[frame_lbl]

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
