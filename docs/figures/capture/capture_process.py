"""
The processing windows of a population block: the block itself, the model upload
window, the threshold configuration wizard (replaying the demo's own
threshold_config_targets.json), the tracking settings and the two training windows.

Runs on demo_adcc (CELLDETECTIVE_DOCS_ADCC), target population.
"""

from unittest import mock

from harness import *

init, cp = open_experiment(require_adcc())
init.hide()
block = cp.ProcessPopulations[1]
block.collapse_btn.click()
pump(1)
move(cp, 20, 40, 460, 700)
grab(cp, "process_targets", marks={
    "segment": block.segment_action, "seg_models": block.seg_model_list, "upload": block.upload_model_btn,
    "seg_config": block.segmentation_config_btn, "train_seg": block.train_btn,
    "track": block.track_action, "track_config": block.track_config_btn, "track_view": block.check_tracking_result_btn,
    "measure": block.measure_action, "classify": block.classify_btn, "measure_config": block.measurements_config_btn,
    "events": block.signal_analysis_action, "event_models": block.signal_models_list,
    "event_config": block.config_signal_annotator_btn, "event_view": block.check_signals_btn,
    "explore": block.view_tab_btn, "submit": block.submit_btn,
})

# -- Upload a segmentation model ------------------------------------------------
block.upload_segmentation_model()
pump(2)
loader = block.seg_model_loader
move(loader, 500, 40)
grab(loader, "upload_model_stardist")
loader.cellpose_button.click()
pump(0.8)
loader.adjustSize()
pump(0.5)
grab(loader, "upload_model_cellpose", marks={
    "cellpose": loader.cellpose_button, "stardist": loader.stardist_button, "threshold": loader.threshold_button,
    "calibration": loader.spatial_calib_le, "diameter": loader.cp_diameter_le,
    "cellprob": loader.cp_cellprob_le, "flow": loader.cp_flow_le,
    "channel0": loader.channel_layout.channel_cbs[0], "channel1": loader.channel_layout.channel_cbs[1],
    "choose": loader.open_dialog_button, "upload": loader.upload_button,
})
loader.threshold_button.click()
pump(0.8)
loader.adjustSize()
pump(0.5)
grab(loader, "upload_model_threshold", marks={
    "threshold": loader.threshold_button, "wizard": loader.threshold_config_button,
    "choose": loader.open_dialog_button, "upload": loader.upload_button,
})

# -- Threshold configuration wizard ----------------------------------------------
loader.open_threshold_config_wizard()
pump(6)
wiz = loader.thresh_wizard
move(wiz, 20, 20, 1400, 820)
pump(2)
config = os.path.join(ADCC, "configs", "threshold_config_targets.json")
with mock.patch(
    "celldetective.gui.thresholds_gui.QFileDialog.getOpenFileName",
    return_value=(config, "JSON (*.json)"),
):
    wiz.load_previous_config()
pump(8)
wiz.scroll_area.verticalScrollBar().setValue(0)
pump(1)
grab(wiz, "threshold_wizard_1", marks={
    "preprocessing": wiz.preprocessing.list, "apply": wiz.preprocessing.apply_btn,
    "hist": wiz.canvas_hist, "slider": wiz.threshold_slider, "fill": wiz.fill_holes_btn,
    "log": wiz.ylog_check, "equalize": wiz.equalize_option_btn, "markers": wiz.marker_option,
    "footprint": wiz.footprint_slider, "min_dist": wiz.min_dist_slider, "run": wiz.markers_btn,
    "watershed": wiz.watershed_btn, "props": wiz.propscanvas, "feature0": wiz.features_cb[0],
    "feature1": wiz.features_cb[1], "query": wiz.property_query_le, "submit": wiz.submit_query_btn,
    "save": wiz.save_btn, "viewer": wiz.viewer.canvas, "channel": wiz.viewer.channel_cb,
})
wiz.scroll_area.verticalScrollBar().setValue(wiz.scroll_area.verticalScrollBar().maximum())
pump(1)
grab(wiz, "threshold_wizard_2", marks={
    "preprocessing": wiz.preprocessing.list, "apply": wiz.preprocessing.apply_btn,
    "hist": wiz.canvas_hist, "slider": wiz.threshold_slider, "fill": wiz.fill_holes_btn,
    "log": wiz.ylog_check, "equalize": wiz.equalize_option_btn, "markers": wiz.marker_option,
    "footprint": wiz.footprint_slider, "min_dist": wiz.min_dist_slider, "run": wiz.markers_btn,
    "watershed": wiz.watershed_btn, "props": wiz.propscanvas, "feature0": wiz.features_cb[0],
    "feature1": wiz.features_cb[1], "query": wiz.property_query_le, "submit": wiz.submit_query_btn,
    "save": wiz.save_btn, "viewer": wiz.viewer.canvas, "channel": wiz.viewer.channel_cb,
})
wiz.close()
pump(0.5)
loader.close()
pump(0.5)

# -- Tracking settings ------------------------------------------------------------
block.open_tracking_configuration_ui()
pump(2)
tr = block.settings_tracking
tr.btrack_option.click()
pump(1)
move(tr, 500, 40)
grab(tr, "tracking_settings_btrack", marks={
    "btrack": tr.btrack_option, "trackpy": tr.trackpy_option, "config": tr.config_frame,
    "features": tr.features_frame, "post": tr.post_proc_frame, "tracklength": tr.min_tracklength_slider,
    "save": tr.submit_btn,
})
tr.close()
pump(0.5)

# -- Training windows ---------------------------------------------------------------
block.open_segmentation_model_config_ui()
pump(3)
st = block.settings_segmentation_training
move(st, 500, 40, 560, 900)
st.stardist_model.click()
pump(0.5)
st.ch_norm.channel_cbs[0].setCurrentText("live_nuclei_channel")
pump(0.5)
st._scroll_area.verticalScrollBar().setValue(0)
grab(st, "train_segmentation_1", marks={
    "model": st.model_frame, "data": st.data_frame, "hyper": st.hyper_frame, "train": st.submit_btn,
    "name": st.modelname_le, "pretrained": st.browse_pretrained_btn, "data_folder": st.select_data_folder_btn,
    "dataset": st.dataset_cb, "augmentation": st.augmentation_slider, "validation": st.validation_slider,
    "channel0": st.ch_norm.channel_cbs[0], "add_channel": st.ch_norm.add_col_btn,
    "calibration": getattr(st, "spatial_calib_le", None), "length": getattr(st, "model_length_slider", None),
    "event_name": getattr(st, "class_name_le", None), "epochs": st.epochs_slider,
})
st._scroll_area.verticalScrollBar().setValue(st._scroll_area.verticalScrollBar().maximum())
pump(0.5)
grab(st, "train_segmentation_2", marks={
    "model": st.model_frame, "data": st.data_frame, "hyper": st.hyper_frame, "train": st.submit_btn,
    "name": st.modelname_le, "pretrained": st.browse_pretrained_btn, "data_folder": st.select_data_folder_btn,
    "dataset": st.dataset_cb, "augmentation": st.augmentation_slider, "validation": st.validation_slider,
    "channel0": st.ch_norm.channel_cbs[0], "add_channel": st.ch_norm.add_col_btn,
    "calibration": getattr(st, "spatial_calib_le", None), "length": getattr(st, "model_length_slider", None),
    "event_name": getattr(st, "class_name_le", None), "epochs": st.epochs_slider,
})
st.close()
pump(0.5)

block.open_signal_model_config_ui()
pump(3)
se = block.settings_event_detection_training
move(se, 500, 40, 560, 900)
pump(0.5)
se._scroll_area.verticalScrollBar().setValue(0)
grab(se, "train_event_1", marks={
    "model": se.model_frame, "data": se.data_frame, "hyper": se.hyper_frame, "train": se.submit_btn,
    "name": se.modelname_le, "pretrained": se.browse_pretrained_btn, "data_folder": se.select_data_folder_btn,
    "dataset": se.dataset_cb, "augmentation": se.augmentation_slider, "validation": se.validation_slider,
    "channel0": se.ch_norm.channel_cbs[0], "add_channel": se.ch_norm.add_col_btn,
    "calibration": getattr(se, "spatial_calib_le", None), "length": getattr(se, "model_length_slider", None),
    "event_name": getattr(se, "class_name_le", None), "epochs": se.epochs_slider,
})
se._scroll_area.verticalScrollBar().setValue(se._scroll_area.verticalScrollBar().maximum())
pump(0.5)
grab(se, "train_event_2", marks={
    "model": se.model_frame, "data": se.data_frame, "hyper": se.hyper_frame, "train": se.submit_btn,
    "name": se.modelname_le, "pretrained": se.browse_pretrained_btn, "data_folder": se.select_data_folder_btn,
    "dataset": se.dataset_cb, "augmentation": se.augmentation_slider, "validation": se.validation_slider,
    "channel0": se.ch_norm.channel_cbs[0], "add_channel": se.ch_norm.add_col_btn,
    "calibration": getattr(se, "spatial_calib_le", None), "length": getattr(se, "model_length_slider", None),
    "event_name": getattr(se, "class_name_le", None), "epochs": se.epochs_slider,
})
se.close()
pump(0.5)
App.quit()
