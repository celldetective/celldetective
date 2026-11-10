# Event detection measurements

The scripts behind the "Measured accuracy" section of the
[Event detection wiki page](https://github.com/celldetective/celldetective/wiki/Feature-Event-detection).
They are not part of the package and are not run by the tests.

## Data

Run the scripts from a working folder of your own, or point
`CELLDETECTIVE_EVAL_DIR` to it. It holds the data and receives the results:

```
zenodo_data/db-si-NucPI/.../*.npy
zenodo_data/db-si-NucCondensation/.../*.npy
zenodo_models/lysis_H_PI/lysis_H_PI/
```

The links to the annotated sets and to the models are in
`celldetective/links/zenodo.json`. The sets are pickled `.npy` files.
`safe_npy.py` reads them with an unpickler limited to numpy array types, and
`read_scores.py` reads the `scores.npy` stored with a model the same way. Use them
rather than `np.load(allow_pickle=True)`, which can run code from the file.
`inspect_h5.py` lists the layers of a Keras model without building it, to check
that it holds no `Lambda` layer before loading it.

`lysis_PI_area` and `NucCond` are read from `celldetective/models/signal_detection`,
so they must be installed there. `REPOSITORY` below is the path of this repository.

## Scripts

| Script | Measures |
|---|---|
| `eval_models.py REPOSITORY MODELS_FOLDER` | Shipped models through the detection route, on every annotated track |
| `eval_conditions.py REPOSITORY` | Times from conditions, on real traces and in a noise sweep |
| `measure_padding.py REPOSITORY MODELS_FOLDER` | Padding after a track end, on simulated early endings |
| `eval_h_pi_channels.py REPOSITORY`, `diagnose_h_pi.py REPOSITORY` | Why `lysis_H_PI` misses events |
| `lomo_train.py REPOSITORY [MODEL]` | Leave one movie out: retrain, then predict the movie left out. Resumes where it stopped. |
| `lomo_summary.py` | Retrained models against the shipped ones, on the same movies |

`eval_common.py` holds the scores and the table building.

Retraining needs TensorFlow on a GPU: a fold took 27 to 60 minutes on an MX150. It
uses a learning rate of 0.001, because a fold at the default of 0.01 gave a time
regressor stuck on one value.
