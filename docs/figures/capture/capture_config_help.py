from harness import *

init, cp = open_experiment()
init.hide()
move(cp, 40, 10)
block = cp.ProcessPopulations[0]
block.collapse_btn.click()
pump(1)
grab(cp, "control_panel_effectors")

# Help: the menu of helpers, then a helper half-way and at its suggestion.
block.help_seg_btn.click()
pump(1)
menu = next((w for w in App.topLevelWidgets() if w.isVisible() and type(w).__name__ == "HelpMenu"), None)
if menu is not None:
    move(menu, 520, 60)
    grab(menu, "help_menu")
    menu.close()

block.help_tracking()
panel = block._help_panel
move(panel, 520, 60)
grab(panel, "help_question")
panel.answer("yes")
pump(0.5)
panel.answer("yes")
pump(0.5)
grab(panel, "help_suggestion")
panel.close()

# Configuration editor, one capture per tab.
cp.open_config_editor()
pump(1)
ed = cp.cfg_editor
ed.path_label.full_text = r"C:\Users\me\Experiments\demo_ricm\config.ini"
move(ed, 440, 40, 640, 500)
from PyQt5.QtWidgets import QTableWidgetItem
ed.add_label("replicate")
ed.labels_table.setItem(0, ed.labels_table.columnCount() - 1, QTableWidgetItem("R1"))
ed.labels_table.setCurrentCell(0, ed.labels_table.columnCount() - 1)
row = ed.metadata_table.rowCount()
ed.metadata_table.insertRow(row)
ed.metadata_table.setItem(row, 0, QTableWidgetItem("date"))
ed.metadata_table.setItem(row, 1, QTableWidgetItem("2024-03-27"))
ed.metadata_table.setCurrentCell(row, 1)
pump(0.3)
ed.labels_table.horizontalScrollBar().setValue(0)
for i in (1, 2):  # the well labels and the metadata tabs
    ed.tabs.setCurrentIndex(i)
    pump(0.4)
    ed.labels_table.resizeColumnsToContents()
    ed.labels_table.horizontalScrollBar().setValue(0)
    pump(0.3)
    grab(ed, f"config_editor_{i}_" + ed.tabs.tabText(i).replace(" ", "_").lower())
ed.close()  # never saved: the demo copy stays as it is
