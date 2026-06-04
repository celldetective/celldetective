Preprocessing
=============

.. _preprocessing:


Overview
--------

Preprocessing is an essential step to prepare your microscopy data for analysis in Celldetective. It includes both off-software and in-software methods to align, correct, and optimize your stacks for segmentation and downstream measurements.


Off-software preprocessing
--------------------------

Registration
~~~~~~~~~~~~

We highly recommend aligning your movies before using Celldetective to correct for stage drift. This is typically done using external tools like Fiji (ImageJ).

.. seealso::
    :doc:`how-to-guides/basics/register-stacks-with-fiji` for a step-by-step guide on using the customized Fiji macro for batch registration.



In-software preprocessing
--------------------------

The **Preprocessing** module lets you batch-correct stacks directly within Celldetective. Corrected stacks are saved with the prefix ``Corrected_`` (or ``Aligned_`` for registration).

.. list-table::
   :widths: 30 50 20
   :header-rows: 1

   * - Task
     - Description
     - Guide
   * - Background correction (Model Fit)
     - Fits a 2D surface to the background, excluding cells. Best for fluorescence images.
     - :doc:`how-to <how-to-guides/basics/correct-a-fluo-background>`
   * - Background correction (Model Free)
     - Computes a median background from multiple positions or timeframes. Best for brightfield/reflection microscopy.
     - :doc:`how-to <how-to-guides/basics/perform-model-free-background-correction>`
   * - Channel offset correction
     - Aligns channels that have pixel shifts between modalities.
     - :doc:`how-to <how-to-guides/basics/align-channels>`
   * - Image Registration & Drift Correction
     - Native multi-channel subpixel translation registration using Fourier, SIFT, or Hybrid correlation.
     - See the section below.

Native Image Registration
~~~~~~~~~~~~~~~~~~~~~~~~~~

Celldetective includes a native, premium-tier Image Registration and Drift Correction engine designed for multi-modal microscopy.

* **Multiple Registration Methods**:
  - **Fourier (Phase Cross-Correlation)**: Optimal for structured, dense, or continuous intensity fields.
  - **SIFT (Scale-Invariant Feature Transform)**: Robust for sparse or dot-like feature fields. Finds robust matching keypoints using RANSAC estimation.
  - **Hybrid**: First attempts SIFT matching, dynamically falling back to Fourier registration if keypoints are sparse or fail RANSAC.
  - **SPT (Single-Particle Tracking)**: Detects local maxima peaks (landmarks/particles/beads) and links their trajectories across frames using `trackpy`. Best for tracking discrete fluorescent puncta, beads, or isolated cells to estimate stage drift without relying on full-image texture or keypoints.
* **Interactive Spot Detection Preview**:
  Launches a live, PyQt-based interactive preview visualization. Users can adjust parameters (e.g. minimum distance, relative detection threshold, smoothing sigma, linking search range, and memory) with real-time feedback. Matplotlib scatter rings overlay detected spots dynamically as the user changes parameters or drags the frame slider.
* **Multi-Channel Joint Consensus**:
  Simultaneously estimates a unified consensus translation shift across multiple selected target channels using weighted consensus:
  
  .. math::
     \mathbf{v}_{\text{consensus}} = \frac{\sum_{c \in \mathcal{C}} w_c \mathbf{v}_c}{\sum_{c \in \mathcal{C}} w_c}

  where weights are dynamically computed via RANSAC inlier counts (SIFT/Hybrid) or correlation confidence (Fourier).
* **Fault-Tolerant Clamping & Fallbacks**: Corrects interleaved stacks safely by automatically clamping physical stack length pages, and gracefully falls back to the last valid shift if all channels fail to register confidently on any frame.
* **Interactive Drift Trajectory Plotter**: Launches an interactive PyQt5/Matplotlib canvas visualizing raw versus filtered (median-smoothed) translation curves, outliers, fallback lines, and RANSAC keypoint quality trends.
* **Advanced Subpixel Shift Methods**:
  - **Spatial Spline Interpolation**: Translates images using spatial spline interpolation (e.g. bilinear/bicubic). Suitable for most datasets but can introduce minor high-frequency interpolation blurring or edge artifacts.
  - **Fourier Domain Phase Multiplier (Subpixel Precision)**: Performs sub-pixel translation directly in the frequency domain using analytical phase multiplication in Fourier space. Perfect for preserving fine texture details, sharp edges, and completely avoiding spline-related spatial interpolation artifacts.
* **Out-of-Bounds & Periodic Artifact Prevention**:
  Fourier-domain shifting inherently wraps image boundaries periodically due to the discrete Fourier transform. Celldetective implements dynamic, analytical boundary coordinate range calculation to mask and zero/NaN out-of-bounds coordinates, maintaining correct physical camera bounds.
* **NaN-Resistant FFT Shifting**:
  Standard FFT operations yield empty results if any input pixel is NaN. Celldetective implements a robust, three-stage NaN-resistant pipeline: (1) dynamically interpolates NaNs in the input array before FFT shift; (2) applies the Fourier-domain subpixel shift on the interpolated array; (3) translates the original NaN boolean mask using nearest spatial translation (nearest-neighbor, ``order=0``) and reapplies it to perfectly mask the translated NaN coordinates.


For a full list of parameters (thresholds, tile options, etc.), see the :ref:`Preprocessing Protocols Reference <ref_preprocessing_settings>`.


Bibliography
------------

 Schindelin, J., Arganda-Carreras, I., Frise, E. et al. Fiji: an open-source platform for biological-image analysis. Nat Methods 9, 676–682 (2012). https://doi.org/10.1038/nmeth.2019

