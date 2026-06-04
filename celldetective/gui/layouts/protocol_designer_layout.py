from typing import Optional

from PyQt5.QtCore import QSize, Qt, QMimeData, QPoint
from PyQt5.QtGui import QDrag, QPixmap
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QLabel,
    QTabWidget,
    QSizePolicy,
    QListWidget,
    QPushButton,
    QHBoxLayout,
    QMainWindow,
    QScrollArea,
    QFrame,
    QWidget,
    QApplication,
)
from fonticon_mdi6 import MDI6
from superqt.fonticon import icon

from celldetective.gui.base.components import CelldetectiveWidget
from celldetective.gui.base.styles import Styles


class PipelineCard(QFrame):
    """Represent an individual card inside the horizontal pipeline chain"""

    def __init__(self, text: str, index: int, parent_pipeline: "VisualPipelineWidget") -> None:
        super().__init__()
        self.index = index
        self.parent_pipeline = parent_pipeline
        self.drag_start_position = QPoint()

        self.setObjectName("PipelineCard")

        # Decide icon based on text parameters
        card_icon = MDI6.cog
        color = "#1565c0"

        text_lower = text.lower()
        if "model-free" in text_lower or "fit" in text_lower or "rolling" in text_lower or "background" in text_lower:
            card_icon = MDI6.image_filter_black_white
            color = "#2e7d32"  # Emerald Green
        elif "offset" in text_lower or "target_channel" in text_lower:
            card_icon = MDI6.vector_difference_ba
            color = "#e65100"  # Orange Accent
        elif "registration" in text_lower or "method" in text_lower:
            card_icon = MDI6.movie_search
            color = "#1565c0"  # Cobalt Blue

        # Clean text layout
        display_text = text
        if "correction_type" in text_lower:
            parts = [p.strip() for p in text.split(",")]
            type_part = ""
            details = []
            for p in parts:
                if ":" in p:
                    k, v = p.split(":", 1)
                    k, v = k.strip(), v.strip()
                    if k == "correction_type":
                        if v == "model-free":
                            type_part = "Model-Free Background"
                        elif v == "fit":
                            type_part = "Background Fit"
                        elif v == "rolling-ball" or v == "rolling_ball":
                            type_part = "Rolling Ball"
                        else:
                            type_part = v.capitalize()
                    elif k in ["method", "target_channel", "correction_horizontal", "correction_vertical", "radius", "sigma"]:
                        details.append(f"{k}={v}")
            display_text = f"<b>{type_part}</b>"
            if details:
                display_text += f"<br/><font color='#64748b' size='1'>{' '.join(details)}</font>"

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(8, 6, 8, 6)
        self.layout.setSpacing(8)

        self.icon_lbl = QLabel()
        self.icon_lbl.setPixmap(icon(card_icon, color=color).pixmap(18, 18))
        self.layout.addWidget(self.icon_lbl)

        self.text_lbl = QLabel(display_text)
        self.text_lbl.setTextFormat(Qt.RichText)
        self.text_lbl.setStyleSheet("font-size: 11px; font-weight: normal; color: #1e293b;")
        self.layout.addWidget(self.text_lbl)

        self.del_btn = QPushButton()
        self.del_btn.setIcon(icon(MDI6.close, color="#ef4444"))
        self.del_btn.setIconSize(QSize(12, 12))
        self.del_btn.setFixedSize(18, 18)
        self.del_btn.setToolTip("Remove step")
        self.del_btn.setStyleSheet(
            """
            QPushButton {
                background-color: transparent;
                border: none;
                border-radius: 9px;
            }
            QPushButton:hover {
                background-color: #fee2e2;
            }
            """
        )
        self.del_btn.clicked.connect(self.remove_self)
        self.layout.addWidget(self.del_btn)

        self.setStyleSheet(
            f"""
            QFrame#PipelineCard {{
                background-color: #ffffff;
                border: 1px solid {color}80;
                border-radius: 8px;
                padding: 2px;
            }}
            QFrame#PipelineCard:hover {{
                border: 1.5px solid {color};
                background-color: #f8fafc;
                cursor: grab;
            }}
            """
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_start_position = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.LeftButton):
            return
        if (event.pos() - self.drag_start_position).manhattanLength() < QApplication.startDragDistance():
            return

        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setData("application/x-pipelinecard-index", str(self.index).encode('utf-8'))
        drag.setMimeData(mime_data)

        pixmap = self.grab()
        drag.setPixmap(pixmap)
        drag.setHotSpot(event.pos())

        drag.exec_(Qt.MoveAction)

    def remove_self(self):
        self.parent_pipeline.remove_protocol_at_index(self.index)


class PipelineContainer(QWidget):
    """Container widget inside the scroll area that holds the pipeline cards and arrows"""

    def __init__(self, parent_pipeline: "VisualPipelineWidget") -> None:
        super().__init__()
        self.parent_pipeline = parent_pipeline
        self.setObjectName("PipelineContainer")
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasFormat("application/x-pipelinecard-index"):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasFormat("application/x-pipelinecard-index"):
            event.acceptProposedAction()
            insert_idx = self.calculate_insert_index(event.pos())
            self.parent_pipeline.show_drop_indicator(insert_idx)
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        if event.mimeData().hasFormat("application/x-pipelinecard-index"):
            try:
                source_idx = int(event.mimeData().data("application/x-pipelinecard-index").data().decode('utf-8'))
                target_idx = self.calculate_insert_index(event.pos())
                event.acceptProposedAction()
                self.parent_pipeline.move_protocol(source_idx, target_idx)
            except Exception:
                pass
            self.parent_pipeline.hide_drop_indicator()
        else:
            super().dropEvent(event)

    def leaveEvent(self, event) -> None:
        self.parent_pipeline.hide_drop_indicator()
        super().leaveEvent(event)

    def calculate_insert_index(self, pos: QPoint) -> int:
        cards = self.parent_pipeline.cards
        if not cards:
            return 0

        drop_x = pos.x()

        # Determine target index based on horizontal position relative to card centers
        for i, card in enumerate(cards):
            card_rect = card.geometry()
            card_center_x = card_rect.x() + card_rect.width() / 2
            if drop_x < card_center_x:
                return i
        return len(cards)


class VisualPipelineWidget(QScrollArea):
    """Horizontal flowchart displaying operations as chained cards"""

    def __init__(self, parent_designer: "ProtocolDesignerLayout") -> None:
        super().__init__()
        self.parent_designer = parent_designer

        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameStyle(QFrame.NoFrame)
        self.setFixedHeight(85)
        self.setStyleSheet("background-color: transparent;")

        self.container = PipelineContainer(self)
        self.container.setStyleSheet(
            """
            QWidget#PipelineContainer {
                background-color: #f8fafc;
                border: 1px dashed #cbd5e1;
                border-radius: 8px;
            }
            """
        )

        self.container_layout = QHBoxLayout(self.container)
        self.container_layout.setContentsMargins(8, 8, 8, 8)
        self.container_layout.setSpacing(8)
        self.container_layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.setWidget(self.container)

        self.cards = []
        self.raw_texts = []
        self.drop_indicator = None
        self.update_empty_state()

    def addItem(self, text: str) -> None:
        self.raw_texts.append(text)
        self.rebuild_pipeline()

    def remove_protocol_at_index(self, index: int) -> None:
        if 0 <= index < len(self.raw_texts):
            if index < len(self.parent_designer.protocols):
                del self.parent_designer.protocols[index]
            self.raw_texts.pop(index)
            self.rebuild_pipeline()

    def takeItem(self, index: int) -> None:
        if 0 <= index < len(self.raw_texts):
            self.raw_texts.pop(index)
            self.rebuild_pipeline()

    def clear(self) -> None:
        self.raw_texts.clear()
        self.rebuild_pipeline()

    def currentRow(self) -> int:
        if len(self.raw_texts) > 0:
            return len(self.raw_texts) - 1
        return -1

    def rebuild_pipeline(self) -> None:
        # Clear container layout
        while self.container_layout.count() > 0:
            item = self.container_layout.takeAt(0)
            w = item.widget()
            if w and w is not self.drop_indicator:
                w.deleteLater()

        self.cards.clear()

        # Re-create/setup drop indicator
        if not hasattr(self, "drop_indicator") or self.drop_indicator is None:
            self.drop_indicator = QFrame()
            self.drop_indicator.setFrameShape(QFrame.VLine)
            self.drop_indicator.setLineWidth(2)
            self.drop_indicator.setStyleSheet("color: #1565c0; background-color: #1565c0;")
            self.drop_indicator.setFixedWidth(2)
            self.drop_indicator.setFixedHeight(40)
        self.drop_indicator.setVisible(False)

        # Rebuild empty state or add cards
        if len(self.raw_texts) == 0:
            self.update_empty_state()
        else:
            for idx, text in enumerate(self.raw_texts):
                if idx > 0:
                    arrow = QLabel()
                    arrow.setPixmap(icon(MDI6.chevron_right, color="#94a3b8").pixmap(14, 14))
                    self.container_layout.addWidget(arrow)

                card = PipelineCard(text, idx, self)
                self.container_layout.addWidget(card)
                self.cards.append(card)

        # Trigger summary update callback on the parent panel if it exists
        if hasattr(self.parent_designer.parent_window, "update_pipeline_summary"):
            self.parent_designer.parent_window.update_pipeline_summary()

    def update_empty_state(self) -> None:
        empty_lbl = QLabel("No corrections queued yet. Add some operations above!")
        empty_lbl.setStyleSheet("color: #64748b; font-style: italic; font-size: 11px; padding-left: 8px;")
        self.container_layout.addWidget(empty_lbl)

    def show_drop_indicator(self, insert_idx: int) -> None:
        if not hasattr(self, "drop_indicator") or self.drop_indicator is None:
            return

        self.drop_indicator.setVisible(False)
        self.container_layout.removeWidget(self.drop_indicator)

        layout_idx = insert_idx * 2
        layout_idx = min(layout_idx, self.container_layout.count())

        self.container_layout.insertWidget(layout_idx, self.drop_indicator)
        self.drop_indicator.setVisible(True)

    def hide_drop_indicator(self) -> None:
        if hasattr(self, "drop_indicator") and self.drop_indicator is not None:
            self.drop_indicator.setVisible(False)

    def move_protocol(self, source_idx: int, target_idx: int) -> None:
        if source_idx == target_idx or source_idx == target_idx - 1:
            return

        adjusted_target = target_idx
        if target_idx > source_idx:
            adjusted_target -= 1

        if 0 <= source_idx < len(self.raw_texts) and 0 <= adjusted_target < len(self.raw_texts):
            # Move in protocols list
            if source_idx < len(self.parent_designer.protocols) and adjusted_target < len(self.parent_designer.protocols):
                item_proto = self.parent_designer.protocols.pop(source_idx)
                self.parent_designer.protocols.insert(adjusted_target, item_proto)

            # Move in raw_texts
            item_text = self.raw_texts.pop(source_idx)
            self.raw_texts.insert(adjusted_target, item_text)

            self.rebuild_pipeline()


class ProtocolDesignerLayout(QVBoxLayout, Styles):
    """Multi tabs and list widget configuration for background correction
    in preprocessing and measurements
    """

    def __init__(
        self,
        parent_window: Optional[QMainWindow] = None,
        tab_layouts: Optional[list] = None,
        tab_names: Optional[list] = None,
        title: Optional[str] = "Protocol Designer",
        list_title: Optional[str] = "Order",
        use_visual_pipeline: Optional[bool] = True,
        *args,
    ) -> None:
        super().__init__(*args)

        self.title = title
        self.parent_window = parent_window
        self.channel_names = self.parent_window.channel_names
        self.tab_layouts = tab_layouts
        self.tab_names = tab_names
        self.list_title = list_title
        self.use_visual_pipeline = use_visual_pipeline
        self.protocols = []
        if len(self.tab_layouts) != len(self.tab_names):
            raise ValueError("tab_layouts and tab_names must have the same length.")

        self.generate_widgets()
        self.generate_layout()

    def generate_widgets(self):
        """Generate the widgets."""

        self.title_lbl = QLabel(self.title)
        self.title_lbl.setStyleSheet(
            """
			font-weight: bold;
			padding: 0px;
			"""
        )

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(self.qtab_style)
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        for k in range(len(self.tab_layouts)):
            wg = CelldetectiveWidget()
            self.tab_layouts[k].parent_window = self
            wg.setLayout(self.tab_layouts[k])
            self.tabs.addTab(wg, self.tab_names[k])

        self.protocol_list_lbl = QLabel(self.list_title)
        
        if self.use_visual_pipeline:
            self.protocol_list = VisualPipelineWidget(self)
        else:
            self.protocol_list = QListWidget()
            self.protocol_list.setStyleSheet(
                """
                QListWidget {
                    border: 1px solid #B8B8B8;
                    border-radius: 5px;
                    background-color: #fcfcfc;
                    padding: 4px;
                    font-family: Consolas, Monaco, monospace;
                    font-size: 10px;
                }
                QListWidget::item {
                    padding: 4px;
                    border-bottom: 1px solid #eeeeee;
                }
                QListWidget::item:selected {
                    background-color: #1565c0;
                    color: white;
                }
                """
            )

        self.delete_protocol_btn = QPushButton("")
        self.delete_protocol_btn.setStyleSheet(self.button_select_all)
        self.delete_protocol_btn.setIcon(icon(MDI6.trash_can, color="black"))
        self.delete_protocol_btn.setToolTip("Remove.")
        self.delete_protocol_btn.setIconSize(QSize(20, 20))
        self.delete_protocol_btn.clicked.connect(self.remove_protocol_from_list)

    def generate_layout(self):
        """Generate the layout."""

        self.correction_layout = QVBoxLayout()

        self.background_correction_layout = QVBoxLayout()
        self.background_correction_layout.setContentsMargins(0, 0, 0, 0)
        self.title_layout = QHBoxLayout()
        self.title_layout.addWidget(self.title_lbl, 100, alignment=Qt.AlignCenter)
        self.background_correction_layout.addLayout(self.title_layout)
        self.background_correction_layout.addWidget(self.tabs)
        self.correction_layout.addLayout(self.background_correction_layout)

        self.addLayout(self.correction_layout)

        self.list_layout = QVBoxLayout()
        list_header_layout = QHBoxLayout()
        list_header_layout.addWidget(self.protocol_list_lbl)
        list_header_layout.addWidget(self.delete_protocol_btn, alignment=Qt.AlignRight)
        self.list_layout.addLayout(list_header_layout)
        self.list_layout.addWidget(self.protocol_list)

        self.addLayout(self.list_layout)

    def remove_protocol_from_list(self):
        """Remove the selected protocol from the list."""

        current_item = self.protocol_list.currentRow()
        if current_item > -1:
            del self.protocols[current_item]
            self.protocol_list.takeItem(current_item)
