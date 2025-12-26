import sys
import argparse
import os
import json
import gc
from PyQt5.QtWidgets import (
    QApplication,
    QProgressDialog,
    QWidget,
    QVBoxLayout,
    QMessageBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from celldetective.utils.experiments import (
    locate_stack_and_labels,
    get_experiment_wells,
    get_experiment_labels,
    get_experiment_metadata,
)
from celldetective.utils.common import (
    config_section_to_dict,
    extract_experiment_channels,
)
from celldetective.log_manager import get_logger

logger = get_logger(__name__)

napari = None
magicgui = None
tqdm = None
view_tracks_in_napari = None
_get_contrast_limits = None
auto_correct_masks = None
save_tiff_imagej_compatible = None
Styles = None
np = None
Path = None
PurePath = None


class DataLoaderThread(QThread):
    finished = pyqtSignal(object, object)  # stack, labels
    error = pyqtSignal(str)

    def __init__(self, position, prefix, population):
        super().__init__()
        self.position = position
        self.prefix = prefix
        self.population = population

    def run(self):
        try:
            stack, labels = locate_stack_and_labels(
                self.position, prefix=self.prefix, population=self.population
            )
            self.finished.emit(stack, labels)
        except Exception as e:
            self.error.emit(str(e))


def custom_control_segmentation_napari(args, stack, labels, dialog=None):
    if dialog:
        dialog.setLabelText("Preparing viewer...")
        QApplication.processEvents()

    position = args.position
    prefix = args.prefix
    population = args.population

    output_folder = position + f"labels_{population}{os.sep}"
    logger.info(f"Shape of the loaded image stack: {stack.shape}...")

    viewer = napari.Viewer()  # Heavy call

    if dialog:
        dialog.setLabelText("Calculating contrast limits...")
        QApplication.processEvents()

    contrast_limits = _get_contrast_limits(stack)
    viewer.add_image(
        stack,
        channel_axis=-1,
        colormap=["gray"] * stack.shape[-1],
        contrast_limits=contrast_limits,
    )

    if dialog:
        dialog.setLabelText("Adding labels...")
        QApplication.processEvents()

    viewer.add_labels(labels.astype(int), name="segmentation", opacity=0.4)

    def export_labels():
        labels_layer = viewer.layers["segmentation"].data
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        for t, im in enumerate(tqdm(labels_layer)):
            try:
                im = auto_correct_masks(im)
            except Exception as e:
                logger.error(e)

            save_tiff_imagej_compatible(
                output_folder + f"{str(t).zfill(4)}.tif", im.astype(np.int16), axes="YX"
            )
        logger.info("The labels have been successfully rewritten.")

    def export_annotation():
        parent1 = Path(position).parent
        expfolder = parent1.parent
        config = PurePath(expfolder, Path("config.ini"))
        expfolder = str(expfolder)
        exp_name = os.path.split(expfolder)[-1]

        wells = get_experiment_wells(expfolder)
        well_idx = list(wells).index(str(parent1) + os.sep)

        label_info = get_experiment_labels(expfolder)
        metadata_info = get_experiment_metadata(expfolder)

        info = {}
        for k in list(label_info.keys()):
            values = label_info[k]
            try:
                info.update({k: values[well_idx]})
            except Exception as e:
                logger.error(f"{e=}")

        if metadata_info is not None:
            keys = list(metadata_info.keys())
            for k in keys:
                info.update({k: metadata_info[k]})

        spatial_calibration = 1.0
        try:
            spatial_calibration = float(
                config_section_to_dict(config, "MovieSettings")["pxtoum"]
            )
        except:
            pass

        channel_names, channel_indices = extract_experiment_channels(expfolder)

        annotation_folder = expfolder + os.sep + f"annotations_{population}" + os.sep
        if not os.path.exists(annotation_folder):
            os.mkdir(annotation_folder)

        logger.info("Exporting!")
        t = viewer.dims.current_step[0]
        labels_layer = viewer.layers["segmentation"].data[t]

        try:
            labels_layer = auto_correct_masks(labels_layer)
        except Exception as e:
            logger.error(e)

        fov_export = True

        if "Shapes" in viewer.layers:
            squares = viewer.layers["Shapes"].data
            test_in_frame = np.array(
                [
                    squares[i][0, 0] == t and len(squares[i]) == 4
                    for i in range(len(squares))
                ]
            )
            squares = np.array(squares)
            squares = squares[test_in_frame]
            nbr_squares = len(squares)
            logger.info(f"Found {nbr_squares} ROIs...")
            if nbr_squares > 0:
                fov_export = False

            for k, sq in enumerate(squares):
                xmin = int(sq[0, 1])
                xmax = int(sq[2, 1])
                if xmax < xmin:
                    xmax, xmin = xmin, xmax
                ymin = int(sq[0, 2])
                ymax = int(sq[1, 2])
                if ymax < ymin:
                    ymax, ymin = ymin, ymax

                frame = viewer.layers["Image"].data[t][xmin:xmax, ymin:ymax]

                pad_to_256 = False
                if frame.shape[1] < 256 or frame.shape[0] < 256:
                    pad_to_256 = True

                multichannel = [frame]
                for i in range(len(channel_indices) - 1):
                    try:
                        frame = viewer.layers[f"Image [{i + 1}]"].data[t][
                            xmin:xmax, ymin:ymax
                        ]
                        multichannel.append(frame)
                    except:
                        pass
                multichannel = np.array(multichannel)
                lab = labels_layer[xmin:xmax, ymin:ymax].astype(np.int16)

                if pad_to_256:
                    pad_length_x = max([0, 256 - multichannel.shape[1]])
                    if pad_length_x > 0 and pad_length_x % 2 == 1:
                        pad_length_x += 1
                    pad_length_y = max([0, 256 - multichannel.shape[2]])
                    if pad_length_y > 0 and pad_length_y % 2 == 1:
                        pad_length_y += 1

                    padded_image = np.array(
                        [
                            np.pad(
                                im,
                                (
                                    (pad_length_x // 2, pad_length_x // 2),
                                    (pad_length_y // 2, pad_length_y // 2),
                                ),
                                mode="constant",
                            )
                            for im in multichannel
                        ]
                    )
                    padded_label = np.pad(
                        lab,
                        (
                            (pad_length_x // 2, pad_length_x // 2),
                            (pad_length_y // 2, pad_length_y // 2),
                        ),
                        mode="constant",
                    )
                    lab = padded_label
                    multichannel = padded_image

                save_tiff_imagej_compatible(
                    annotation_folder
                    + f"{exp_name}_{position.split(os.sep)[-2]}_{str(t).zfill(4)}_roi_{xmin}_{xmax}_{ymin}_{ymax}_labelled.tif",
                    lab,
                    axes="YX",
                )
                save_tiff_imagej_compatible(
                    annotation_folder
                    + f"{exp_name}_{position.split(os.sep)[-2]}_{str(t).zfill(4)}_roi_{xmin}_{xmax}_{ymin}_{ymax}.tif",
                    multichannel,
                    axes="CYX",
                )
                info.update(
                    {
                        "spatial_calibration": spatial_calibration,
                        "channels": list(channel_names),
                        "frame": t,
                    }
                )
                with open(
                    annotation_folder
                    + f"{exp_name}_{position.split(os.sep)[-2]}_{str(t).zfill(4)}_roi_{xmin}_{xmax}_{ymin}_{ymax}.json",
                    "w",
                ) as f:
                    json.dump(info, f, indent=4)

        if fov_export:
            frame = viewer.layers["Image"].data[t]
            multichannel = [frame]
            for i in range(len(channel_indices) - 1):
                try:
                    frame = viewer.layers[f"Image [{i + 1}]"].data[t]
                    multichannel.append(frame)
                except:
                    pass
            multichannel = np.array(multichannel)
            save_tiff_imagej_compatible(
                annotation_folder
                + f"{exp_name}_{position.split(os.sep)[-2]}_{str(t).zfill(4)}_labelled.tif",
                labels_layer,
                axes="YX",
            )
            save_tiff_imagej_compatible(
                annotation_folder
                + f"{exp_name}_{position.split(os.sep)[-2]}_{str(t).zfill(4)}.tif",
                multichannel,
                axes="CYX",
            )
            info.update(
                {
                    "spatial_calibration": spatial_calibration,
                    "channels": list(channel_names),
                    "frame": t,
                }
            )
            with open(
                annotation_folder
                + f"{exp_name}_{position.split(os.sep)[-2]}_{str(t).zfill(4)}.json",
                "w",
            ) as f:
                json.dump(info, f, indent=4)

        logger.info("Done.")

    @magicgui(call_button="Save the modified labels")
    def save_widget():
        return export_labels()

    @magicgui(call_button="Export the annotation\nof the current frame")
    def export_widget():
        return export_annotation()

    button_container = QWidget()
    layout = QVBoxLayout(button_container)
    layout.setSpacing(10)
    layout.addWidget(save_widget.native)
    layout.addWidget(export_widget.native)
    viewer.window.add_dock_widget(button_container, area="right")

    save_widget.native.setStyleSheet(Styles().button_style_sheet)
    export_widget.native.setStyleSheet(Styles().button_style_sheet)

    def lock_controls(layer, widgets=(), locked=True):
        qctrl = viewer.window.qt_viewer.controls.widgets[layer]
        for wdg in widgets:
            try:
                getattr(qctrl, wdg).setEnabled(not locked)
            except:
                pass

    try:
        label_widget_list = ["polygon_button", "transform_button"]
        lock_controls(viewer.layers["segmentation"], label_widget_list)
    except Exception as e:
        logger.warning(f"Could not lock controls: {e}")

    # Close dialog just before showing relevant window
    if dialog:
        dialog.close()

    viewer.show(block=True)

    for i in range(10000):
        try:
            viewer.layers.pop()
        except:
            pass
    del viewer
    del stack
    del labels
    gc.collect()


def main():
    parser = argparse.ArgumentParser(
        description="Launch Napari Viewer with Progress Bar"
    )
    parser.add_argument("--position", type=str, required=True, help="Position path")
    parser.add_argument("--prefix", type=str, default="Aligned", help="Image prefix")
    parser.add_argument(
        "--population", type=str, default="target", help="Cell population"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["segmentation", "tracks"],
        help="Viewer mode",
    )
    parser.add_argument(
        "--threads", type=int, default=1, help="Threads for tracking processing"
    )

    args = parser.parse_args()

    # Create App
    app = QApplication.instance() or QApplication(sys.argv)

    # Progress Dialog - Init immediately
    dialog = QProgressDialog("Initializing Viewer...", None, 0, 0)
    dialog.setWindowTitle("Please wait")
    dialog.setWindowModality(Qt.WindowModal)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)
    dialog.setMinimumDuration(0)
    dialog.setCancelButton(None)
    dialog.show()
    app.processEvents()

    # 1. Start loading data immediately (in thread)
    dialog.setLabelText("Loading data...")
    app.processEvents()

    results = {"stack": None, "labels": None, "error": None}

    def on_finished(stack, labels):
        results["stack"] = stack
        results["labels"] = labels
        # Do NOT close dialog here
        app.quit()  # Exit loading loop

    def on_error(msg):
        results["error"] = msg
        dialog.close()
        app.quit()  # Exit loading loop

    thread = DataLoaderThread(args.position, args.prefix, args.population)
    thread.finished.connect(on_finished)
    thread.error.connect(on_error)
    thread.start()

    logger.info("Starting loading loop...")
    app.exec_()
    logger.info("Loading loop finished.")

    if results["error"]:
        logger.error(f"Failed to load data: {results['error']}")
        QMessageBox.critical(None, "Error", f"Failed to load data:\n{results['error']}")
        return

    if results["stack"] is None:
        logger.error("Loading incomplete - no data returned.")
        return

    # 2. Load heavy libraries while dialog is still up
    dialog.setLabelText("Loading napari libraries...")
    app.processEvents()

    try:
        import numpy as np
        from pathlib import Path, PurePath
        import napari
        from magicgui import magicgui
        from tqdm import tqdm
        from celldetective.gui import Styles
        from celldetective.utils.experiments import (
            _get_contrast_limits,
            auto_correct_masks,
            view_tracks_in_napari,
            save_tiff_imagej_compatible,
        )

        # Assign to globals
        globals()["napari"] = napari
        globals()["magicgui"] = magicgui
        globals()["tqdm"] = tqdm
        globals()["np"] = np
        globals()["Path"] = Path
        globals()["PurePath"] = PurePath
        globals()["Styles"] = Styles
        globals()["_get_contrast_limits"] = _get_contrast_limits
        globals()["auto_correct_masks"] = auto_correct_masks
        globals()["view_tracks_in_napari"] = view_tracks_in_napari
        globals()["save_tiff_imagej_compatible"] = save_tiff_imagej_compatible

    except Exception as e:
        dialog.close()
        logger.error(f"Import failed: {e}")
        QMessageBox.critical(
            None, "Start Error", f"Failed to initialize libraries:\n{e}"
        )
        return

    # 3. Launch Viewer
    try:
        if args.mode == "segmentation":
            custom_control_segmentation_napari(
                args, results["stack"], results["labels"], dialog=dialog
            )
        elif args.mode == "tracks":
            if dialog:
                dialog.setLabelText("Preparing tracks...")
                QApplication.processEvents()
                # view_tracks_in_napari handles its own UI or might be fast?
                # It opens a viewer too.
                # If it doesn't accept dialog, we close it before.

            res = view_tracks_in_napari(
                args.position,
                population=args.population,
                stack=results["stack"],
                labels=results["labels"],
                threads=args.threads,
                dialog=dialog,
            )
            if res is None:
                msg = "No tracking data found. Please compute trajectories first."
                logger.error(msg)
                QMessageBox.warning(None, "Warning", msg)

    except Exception as e:
        logger.error(f"Error during visualization: {e}")
        QMessageBox.critical(None, "Error", f"Error during visualization:\n{e}")


if __name__ == "__main__":
    main()
