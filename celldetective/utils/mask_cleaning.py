import concurrent.futures
import threading

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table, label
from skimage.transform import resize
from tqdm import tqdm

from celldetective.utils.image_loaders import load_frames
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import find_objects


def fill_label_holes(lbl_img, **kwargs):
    """Fill small holes in label image.
    from https://github.com/stardist/stardist/blob/main/stardist/utils.py
    """

    # TODO: refactor 'fill_label_holes' and 'edt_prob' to share code
    def grow(sl, interior):
        return tuple(
            slice(s.start - int(w[0]), s.stop + int(w[1])) for s, w in zip(sl, interior)
        )

    def shrink(interior):
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


def _check_label_dims(lbl, file=None, template=None):

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
    masks, bbox_factor: float = 1.75, min_area: int = 9, fill_labels: bool = False
):
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

    assert masks.ndim == 2, "`masks` should be a 2D numpy array..."

    # Avoid negative mask values
    masks[masks < 0] = np.abs(masks[masks < 0])

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

    # Second routine to eliminate objects too small
    props2 = pd.DataFrame(
        regionprops_table(corrected_lbl, properties=("label", "area", "area_bbox"))
    )
    for cell in props2["label"].unique():
        area = props2.loc[props2["label"] == cell, "area"].values
        lbl = corrected_lbl == cell
        if area < min_area:
            corrected_lbl[lbl] = 0

    # Additionnal routine to reorder labels from 1 to number of cells
    label_ids = np.unique(corrected_lbl)[1:]
    clean_labels = corrected_lbl.copy()

    for k, lbl in enumerate(label_ids):
        clean_labels[corrected_lbl == lbl] = k + 1

    clean_labels = clean_labels.astype(int)

    if fill_labels:
        clean_labels = fill_label_holes(clean_labels)

    return clean_labels


def relabel_segmentation(
    labels,
    df,
    exclude_nans=True,
    column_labels={
        "track": "TRACK_ID",
        "frame": "FRAME",
        "y": "POSITION_Y",
        "x": "POSITION_X",
        "label": "class_id",
    },
    threads=1,
    progress_callback=None,
):
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
        df = df.dropna(subset=column_labels["label"])

    new_labels = np.zeros_like(labels)
    shared_data = {"s": 0}

    # Progress tracking
    shared_progress = {"val": 0, "lock": threading.Lock()}
    total_frames = len(df[column_labels["frame"]].dropna().unique())

    def rewrite_labels(indices):

        all_track_ids = df[column_labels["track"]].dropna().unique()

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
            cells = df.loc[
                df[column_labels["frame"]] == f,
                [column_labels["track"], column_labels["label"]],
            ].to_numpy()
            tracks_at_t = list(cells[:, 0])
            identities = list(cells[:, 1])

            labels_at_t = list(np.unique(labels[f]))
            if 0 in labels_at_t:
                labels_at_t.remove(0)
            labels_not_in_df = [lbl for lbl in labels_at_t if lbl not in identities]
            for lbl in labels_not_in_df:
                with threading.Lock():  # Synchronize access to `shared_data["s"]`
                    track_id = max(all_track_ids) + shared_data["s"]
                    shared_data["s"] += 1
                tracks_at_t.append(track_id)
                identities.append(lbl)

            # exclude NaN
            tracks_at_t = np.array(tracks_at_t)
            identities = np.array(identities)

            tracks_at_t = tracks_at_t[identities == identities]
            identities = identities[identities == identities]

            for k in range(len(identities)):

                # need routine to check values from labels not in class_id of this frame and add new track id

                loc_i, loc_j = np.where(labels[f] == identities[k])
                track_id = tracks_at_t[k]

                if track_id == track_id:
                    new_labels[f, loc_i, loc_j] = round(track_id)

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
            print("Exception: ", e)

    if shared_progress.get("cancelled", False):
        print("Relabeling cancelled.")
        return None

    print("\nDone.")

    return new_labels
