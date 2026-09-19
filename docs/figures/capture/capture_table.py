from harness import *
from PyQt5.QtCore import QPoint

init, cp = open_experiment()
init.hide()
move(cp, 1150, 10)
block = cp.ProcessPopulations[0]
block.view_tab_btn.click()
pump(3)
tab = block.tab_ui
move(tab, 20, 20, 1050, 560)

# Select two columns to show the selection caption and the enabled tools.
tv = tab.table_view
cols = list(tab.data.columns)
from PyQt5.QtCore import QItemSelectionModel
sm = tv.selectionModel()
for name in ["POSITION_X", "POSITION_Y"]:
    j = cols.index(name)
    tv.selectColumn(j) if name == "POSITION_X" else sm.select(
        tv.model().index(0, j), QItemSelectionModel.Select | QItemSelectionModel.Columns
    )
pump(0.6)
tv.horizontalScrollBar().setValue(0)
tv.verticalScrollBar().setValue(0)
pump(0.4)
grab(tab, "table_ui")

# Distributions and statistics dialog.
tab.plot_inst_action.trigger()
pump(1.5)
dlg = tab.plot1Dparams
for name in ["boxplot", "strip"]:
    dlg_card = tab.plot_selector.cards[name]
    dlg_card.setChecked(True)
for name in list(tab.stats_selector.cards):
    tab.stats_selector.cards[name].setChecked(True)
tab.x_cb.setCurrentText("well_name") if tab.x_cb.findText("well_name") >= 0 else None
tab.y_cb.setCurrentText("adhesion_channel_mean")
tab.hue_cb.setCurrentText("class_firstdetection")
pump(0.5)
move(dlg, 60, 40)
grab(dlg, "plot_1d_dialog")
tab.plot1d_btn.click()
pump(3)

for attr in ["pval_table", "effect_size_table"]:
    w = getattr(tab, attr, None)
    if w is not None:
        grab(w, attr)
