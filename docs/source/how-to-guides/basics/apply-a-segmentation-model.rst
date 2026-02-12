How to apply a segmentation model
----------------------------------

This guide shows you how to import and run a Deep Learning segmentation model (:term:`StarDist` or :term:`Cellpose`) on your data.

Reference keys: :term:`instance segmentation`, :term:`cell population`


Import a model
~~~~~~~~~~~~~~

#. In the **Segmentation** section of the Control Panel, click **UPLOAD**.

#. Select the model type (:term:`StarDist`, :term:`Cellpose`, or **Threshold**).

#. Click **Choose File** to select your model folder (:term:`StarDist`) or file (:term:`Cellpose`/JSON).

#. Configure the import settings (:term:`Input spatial calibration`, :term:`Channel Mapping`, :term:`Normalization`). For a detailed list of all parameters, see the :ref:`Segmentation Data Import Reference <ref_segmentation_settings>`.

#. Click **Upload** to save the model and its configuration to the project's model zoo.


Run the model
~~~~~~~~~~~~~

#. Tick the **SEGMENT** option in the Control Panel.

#. Select your model from the dropdown list.

#. Click **Submit** to start processing.


Generalist model configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you selected a **generalist** model (e.g., ``SD_versatile_fluo``, ``CP_cyto2``), a configuration window appears after clicking **Submit**. You must map your experiment's channels to the model's expected inputs.

For a detailed list of runtime parameters, see the :ref:`Segmentation Runtime Settings Reference <ref_runtime_segmentation_settings>`.

:term:`StarDist` generalist models

*   Select the channel containing the nuclei (e.g., DAPI or Hoechst).

:term:`Cellpose` generalist models

*   **Channel Mapping**: Select the "Cytoplasm" (channel 1) and "Nuclei" (channel 2, optional) channels from your experiment.

*   **Diameter [px]**: The expected cell diameter in pixels.

    *   *Interactive Tool*: Click the **eye icon** next to the diameter field to open a specific viewer. Adjust the diameter slider until the red circle matches your cells' size. This ensures the model receives images scaled correctly for its training parameters.

*   **Thresholds**:

    *   **Flow threshold**: Controls shape consistency. Maximum error allowed for the flows. Increase (e.g., > 0.4) if cells are missing; decrease to strictly enforce shape constraints.
    *   **Cellprob threshold**: Controls detection sensitivity. Decrease (e.g., < 0.0) to detect fainter or less confident objects.

Image rescaling and normalization are handled automatically based on the internal model configuration.
