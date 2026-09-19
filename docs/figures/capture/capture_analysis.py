"""
Classification, event annotation and the Analyze tab: the classifier with a
previewed query, the event annotator settings and viewer, and the survival and
synchronized signal plots with their option windows.

Runs on demo_ricm (CELLDETECTIVE_DOCS_EXP), effector population: cells spread
(t_spread) after they are first detected (t_firstdetection).
"""

from PyQt5.QtWidgets import QTabWidget

from harness import *

init, cp = open_experiment()
init.hide()
move(cp, 20, 40)
block = cp.ProcessPopulations[0]
block.collapse_btn.click()
pump(1)

# -- Classifier ----------------------------------------------------------------
block.open_classifier_ui()
pump(3)
cl = block.classifier_widget
move(cl, 500, 40)
cl.name_le.setText("spread")
cl.features_cb[0].setCurrentText("area")
cl.features_cb[1].setCurrentText("adhesion_channel_mean")
pump(1)
cl.frame_slider.setValue(30)
pump(1.5)
cl.property_query_le.setText("area > 150 and adhesion_channel_mean < 0.95")
cl.submit_query_btn.click()
pump(1.5)
grab(cl, "classifier_preview", marks={
    "name": cl.name_le, "project": cl.project_times_btn, "frame": cl.frame_slider, "alpha": cl.alpha_slider,
    "feature0": cl.features_cb[0], "feature1": cl.features_cb[1], "query": cl.property_query_le,
    "preview": cl.submit_query_btn, "time_corr": cl.time_corr, "unique": cl.unique_state_btn,
    "irreversible": cl.irreversible_event_btn, "transient": cl.transient_event_btn, "r2": cl.r2_slider,
    "prereq": cl.prereq_event_check, "apply": cl.submit_btn,
})
cl.time_corr.setChecked(True)
pump(0.8)
grab(cl, "classifier_time_correlated", marks={
    "name": cl.name_le, "project": cl.project_times_btn, "frame": cl.frame_slider, "alpha": cl.alpha_slider,
    "feature0": cl.features_cb[0], "feature1": cl.features_cb[1], "query": cl.property_query_le,
    "preview": cl.submit_query_btn, "time_corr": cl.time_corr, "unique": cl.unique_state_btn,
    "irreversible": cl.irreversible_event_btn, "transient": cl.transient_event_btn, "r2": cl.r2_slider,
    "prereq": cl.prereq_event_check, "apply": cl.submit_btn,
})
cl.close()
pump(0.5)

# -- Event annotator settings and viewer --------------------------------------------
block.open_signal_annotator_configuration_ui()
pump(2)
sa = block.settings_signal_annotator
move(sa, 500, 40)
pump(0.5)
grab(sa, "signal_annotator_settings", marks={
    "grayscale": sa.gs_btn, "rgb": sa.rgb_btn, "channel": sa.channel_cbs[0], "fraction": sa.fraction_slider,
    "save": sa.submit_btn, "log": sa.log_btn, "percentile": sa.percentile_btn,
})
sa.close()
pump(0.5)

# -- Analyze tab ----------------------------------------------------------------------
tabs = cp.findChildren(QTabWidget)[0]
tabs.setCurrentIndex(1)
pump(1)
move(cp, 20, 40, 460, 480)
grab(cp, "analyze_tab", marks={
    "survival": cp.SurvivalBlock.survival_btn, "signals": cp.SurvivalBlock.plot_signal_btn,
})

panel = cp.SurvivalBlock
panel.configure_survival()
pump(2)
sv = panel.config_survival
move(sv, 480, 40)
sv.cbs[0].setCurrentText("effectors")
pump(1)
sv.cbs[1].setCurrentText("t_firstdetection")
sv.cbs[2].setCurrentText("t_spread")
pump(0.5)
grab(sv, "survival_options", marks={
    "population": sv.cbs[0], "reference": sv.cbs[1], "interest": sv.cbs[2], "cmap": sv.cbs[3],
    "query": sv.query_le, "calibration": sv.time_calibration_le, "submit": sv.submit_btn,
})
sv.submit_btn.click()
pump(4)
move(sv.plot_window, 900, 40)
pump(1)
grab(sv.plot_window, "survival_plot")
sv.plot_window.close()
sv.close()
pump(0.5)

panel.configure_plot_signals()
pump(2)
sg = panel.config_signal_plot
move(sg, 480, 40)
sg.cbs[0].setCurrentText("effectors")
pump(1)
sg.cbs[1].setCurrentText("class_spread")
sg.cbs[2].setCurrentText("t_spread")
pump(0.5)
grab(sg, "signals_options", marks={
    "population": sg.cbs[0], "class": sg.cbs[1], "interest": sg.cbs[2], "cmap": sg.cbs[3],
    "absolute": sg.abs_time_checkbox, "query": sg.query_le, "calibration": sg.time_calibration_le,
    "pool": sg.pool_option_cb, "submit": sg.submit_btn,
})
sg.submit_btn.click()
pump(2)
fw = sg.feature_choice_widget
fw.move(900, 40)
pump(0.5)
sg.feature_cb.setCurrentText("adhesion_channel_mean")
grab(fw, "signals_feature", marks={"feature": sg.feature_cb, "set": sg.set_feature_btn})
sg.set_feature_btn.click()
pump(4)
move(sg.plot_window, 900, 40)
pump(1)
grab(sg.plot_window, "signals_plot")
sg.plot_window.close()
sg.close()
pump(0.5)
App.quit()
