import numpy as np
from celldetective import get_logger

logger = get_logger()


def split_by_ratio(arr, *ratios):
    """

    Split an array into multiple chunks based on given ratios.

    Parameters
    ----------
    arr : array-like
            The input array to be split.
    *ratios : float
            Ratios specifying the proportions of each chunk. The sum of ratios should be less than or equal to 1.

    Returns
    -------
    list
            A list of arrays containing the splits/chunks of the input array.

    Notes
    -----
    This function randomly permutes the input array (`arr`) and then splits it into multiple chunks based on the provided ratios.
    The ratios determine the relative sizes of the resulting chunks. The sum of the ratios should be less than or equal to 1.
    The function uses the accumulated ratios to determine the split indices.

    The function returns a list of arrays representing the splits of the input array. The number of splits is equal to the number
    of provided ratios. If there are more ratios than splits, the extra ratios are ignored.

    Examples
    --------
    >>> arr = np.arange(10)
    >>> splits = split_by_ratio(arr, 0.6, 0.2, 0.2)
    >>> print(len(splits))
    3
    # Split the array into 3 chunks with ratios 0.6, 0.2, and 0.2.

    >>> arr = np.arange(100)
    >>> splits = split_by_ratio(arr, 0.5, 0.25)
    >>> print([len(split) for split in splits])
    [50, 25]
    # Split the array into 2 chunks with ratios 0.5 and 0.25.

    """

    arr = np.random.permutation(arr)
    ind = np.add.accumulate(np.array(ratios) * len(arr)).astype(int)
    return [x.tolist() for x in np.split(arr, ind)][: len(ratios)]


def compute_weights(y):
    """

    Compute class weights based on the input labels.

    Parameters
    ----------
    y : array-like
            Array of labels.

    Returns
    -------
    dict
            A dictionary containing the computed class weights.

    Notes
    -----
    This function calculates the class weights based on the input labels (`y`) using the "balanced" method.
    The class weights are computed to address the class imbalance problem, where the weights are inversely
    proportional to the class frequencies.

    The function returns a dictionary (`class_weights`) where the keys represent the unique classes in `y`
    and the values represent the computed weights for each class.

    Examples
    --------
    >>> labels = np.array([0, 1, 0, 1, 1])
    >>> weights = compute_weights(labels)
    >>> print(weights)
    {0: 1.5, 1: 0.75}
    # Compute class weights for the binary labels.

    >>> labels = np.array([0, 1, 2, 0, 1, 2, 2])
    >>> weights = compute_weights(labels)
    >>> print(weights)
    {0: 1.1666666666666667, 1: 1.1666666666666667, 2: 0.5833333333333334}
    # Compute class weights for the multi-class labels.

    """
    from sklearn.utils import compute_class_weight

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y),
        y=y,
    )
    class_weights = dict(zip(np.unique(y), class_weights))

    return class_weights


def train_test_split(
    data_x, data_y1, data_class=None, validation_size=0.25, test_size=0, n_iterations=10
):
    """

    Split the dataset into training, validation, and test sets.

    Parameters
    ----------
    data_x : array-like
            Input features or independent variables.
    data_y1 : array-like
            Target variable 1.
    data_y2 : array-like
            Target variable 2.
    validation_size : float, optional
            Proportion of the dataset to include in the validation set. Default is 0.25.
    test_size : float, optional
            Proportion of the dataset to include in the test set. Default is 0.

    Returns
    -------
    dict
            A dictionary containing the split datasets.
            Keys: "x_train", "x_val", "y1_train", "y1_val", "y2_train", "y2_val".
            If test_size > 0, additional keys: "x_test", "y1_test", "y2_test".

    Notes
    -----
    This function divides the dataset into training, validation, and test sets based on the specified proportions.
    It shuffles the data and splits it according to the proportions defined by `validation_size` and `test_size`.

    The input features (`data_x`) and target variables (`data_y1`, `data_y2`) should be arrays or array-like objects
    with compatible dimensions.

    The function returns a dictionary containing the split datasets. The training set is assigned to "x_train",
    "y1_train", and "y2_train". The validation set is assigned to "x_val", "y1_val", and "y2_val". If `test_size` is
    greater than 0, the test set is assigned to "x_test", "y1_test", and "y2_test".

    """

    if data_class is not None:
        logger.info(
            f"Unique classes: {np.sort(np.argmax(np.unique(data_class,axis=0),axis=1))}"
        )

    for i in range(n_iterations):

        n_values = len(data_x)
        randomize = np.arange(n_values)
        np.random.shuffle(randomize)

        train_percentage = 1 - validation_size - test_size

        chunks = split_by_ratio(randomize, train_percentage, validation_size, test_size)

        x_train = data_x[chunks[0]]
        y1_train = data_y1[chunks[0]]
        if data_class is not None:
            y2_train = data_class[chunks[0]]

        x_val = data_x[chunks[1]]
        y1_val = data_y1[chunks[1]]
        if data_class is not None:
            y2_val = data_class[chunks[1]]

        if data_class is not None:
            print(
                f"classes in train set: {np.sort(np.argmax(np.unique(y2_train,axis=0),axis=1))}; classes in validation set: {np.sort(np.argmax(np.unique(y2_val,axis=0),axis=1))}"
            )
            same_class_test = np.array_equal(
                np.sort(np.argmax(np.unique(y2_train, axis=0), axis=1)),
                np.sort(np.argmax(np.unique(y2_val, axis=0), axis=1)),
            )
            print(f"Check that classes are found in all sets: {same_class_test}...")
        else:
            same_class_test = True

        if same_class_test:

            ds = {
                "x_train": x_train,
                "x_val": x_val,
                "y1_train": y1_train,
                "y1_val": y1_val,
            }
            if data_class is not None:
                ds.update({"y2_train": y2_train, "y2_val": y2_val})

            if test_size > 0:
                x_test = data_x[chunks[2]]
                y1_test = data_y1[chunks[2]]
                ds.update({"x_test": x_test, "y1_test": y1_test})
                if data_class is not None:
                    y2_test = data_class[chunks[2]]
                    ds.update({"y2_test": y2_test})
            return ds
        else:
            continue

    raise Exception(
        "Some classes are missing from the train or validation set... Abort."
    )
