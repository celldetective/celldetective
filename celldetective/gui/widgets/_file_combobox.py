import sys
import os
import shutil
from typing import List

from PyQt5.QtWidgets import QComboBox, QMenu, QMessageBox, QListView
from PyQt5.QtCore import Qt, QPoint
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6

def truncate_text(text: str, max_len: int = 30) -> str:
	"""Truncate text if longer than max_len, adding '...'."""
	return text if len(text) <= max_len else text[: max_len - 3] + "..."

from PyQt5.QtWidgets import QStyledItemDelegate

class PaddedItemDelegate(QStyledItemDelegate):
	def sizeHint(self, option, index):
		size = super().sizeHint(option, index)
		size.setHeight(size.height() + 4)  # add a few extra pixels
		return size

class FileComboBox(QComboBox):
	def __init__(self, parent=None):
		super().__init__(parent)
		self.setView(QListView())
		self.view().setItemDelegate(PaddedItemDelegate(self))
		
		# Right-click menus for collapsed and expanded combo
		self.setContextMenuPolicy(Qt.CustomContextMenu)
		self.customContextMenuRequested.connect(self.on_combo_context_menu)
		self.view().setContextMenuPolicy(Qt.CustomContextMenu)
		self.view().customContextMenuRequested.connect(self.on_view_context_menu)

		
		self.delete_icon = icon(MDI6.delete, color="black")

	# -------------------------------
	# Add items with truncated display and tooltip
	# -------------------------------
	def addPath(self, path: str):
		"""Add a path, showing only filename or last folder (truncated)."""
		full_path = os.path.normpath(path)
		display_name0 = os.path.basename(full_path) or full_path
		display_name = truncate_text(display_name0, 30)

		self.addItem(display_name)
		idx = self.count() - 1
		self.setItemData(idx, full_path, Qt.UserRole)
		self.setItemData(idx, display_name0, Qt.ToolTipRole)
	
	def addPaths(self, paths: List[str]):
		for path in paths:
			full_path = os.path.normpath(path)
			display_name0 = os.path.basename(full_path) or full_path
			display_name = truncate_text(display_name0, 30)
			
			self.addItem(display_name)
			idx = self.count() - 1
			self.setItemData(idx, full_path, Qt.UserRole)
			self.setItemData(idx, display_name0, Qt.ToolTipRole)
	
	# -------------------------------
	# Public utility methods
	# -------------------------------
	def get_item_paths(self, index: int | None = None):
		"""Return full paths for all items, or for a single index if provided."""
		if index is None:
			return [self.itemData(i, Qt.UserRole) for i in range(self.count())]
		if 0 <= index < self.count():
			return self.itemData(index, Qt.UserRole)
		return None

	def get_item_names(self, index: int | None = None):
		"""Return basenames (filenames or folder names) for all items or one index."""
		if index is None:
			return [os.path.basename(self.itemData(i, Qt.UserRole)) for i in range(self.count())]
		if 0 <= index < self.count():
			return os.path.basename(self.itemData(index, Qt.UserRole))
		return None

	# -------------------------------
	# Collapsed combo right-click
	# -------------------------------
	def on_combo_context_menu(self, pos: QPoint):
		index = self.currentIndex()
		if index < 0:
			return
		full_path = self.itemData(index, Qt.UserRole)
		self.show_context_menu(full_path, index, self.mapToGlobal(pos))

	# -------------------------------
	# Expanded popup right-click
	# -------------------------------
	def on_view_context_menu(self, pos: QPoint):
		view = self.view()
		index = view.indexAt(pos)
		if not index.isValid():
			return
		full_path = self.itemData(index.row(), Qt.UserRole)
		global_pos = view.viewport().mapToGlobal(pos)
		self.show_context_menu(full_path, index.row(), global_pos)

	# -------------------------------
	# Shared context menu
	# -------------------------------
	def show_context_menu(self, full_path, index, global_pos):
		if full_path is None:
			return
		base_name = os.path.basename(full_path)
		menu = QMenu()
		delete_action = menu.addAction(self.delete_icon, f'Delete "{truncate_text(base_name, 30)}"')
		action = menu.exec_(global_pos)
		if action == delete_action:
			self.delete_item(index, full_path)

	# -------------------------------
	# Delete handler (files & directories)
	# -------------------------------
	def delete_item(self, index, full_path):
		reply = QMessageBox.question(
			self,
			"Confirm Deletion",
			f"Delete '{full_path}' from list (and disk)?",
			QMessageBox.Yes | QMessageBox.No,
		)
		if reply != QMessageBox.Yes:
			return

		self.removeItem(index)

		if os.path.exists(full_path):
			try:
				if os.path.isfile(full_path) or os.path.islink(full_path):
					os.remove(full_path)
				elif os.path.isdir(full_path):
					shutil.rmtree(full_path)
				QMessageBox.information(self, "Deleted", f"Removed {full_path} from disk.")
			except Exception as e:
				QMessageBox.warning(self, "Error", f"Failed to delete '{full_path}':\n{e}")
		else:
			QMessageBox.information(self, "Not found", f"'{full_path}' not found on disk.")
