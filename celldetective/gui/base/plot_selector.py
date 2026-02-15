from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QScrollArea,
    QFrame,
    QLayout,
    QSizePolicy,
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QRect, QPoint
from PyQt5.QtGui import QPixmap
import os


class FlowLayout(QLayout):
    """
    Standard FlowLayout implementation.
    """

    def __init__(self, parent=None, margin=0, spacing=-1):
        super(FlowLayout, self).__init__(parent)
        if parent is not None:
            self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)
        self.itemList = []

    def __del__(self):
        item = self.takeAt(0)
        while item:
            item = self.takeAt(0)

    def addItem(self, item):
        self.itemList.append(item)

    def count(self):
        return len(self.itemList)

    def itemAt(self, index):
        if index >= 0 and index < len(self.itemList):
            return self.itemList[index]
        return None

    def takeAt(self, index):
        if index >= 0 and index < len(self.itemList):
            return self.itemList.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientations(Qt.Orientation(0))

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        height = self.doLayout(QRect(0, 0, width, 0), True)
        return height

    def setGeometry(self, rect):
        super(FlowLayout, self).setGeometry(rect)
        self.doLayout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self.itemList:
            size = size.expandedTo(item.minimumSize())
        size += QSize(
            2 * self.contentsMargins().top(), 2 * self.contentsMargins().top()
        )
        return size

    def doLayout(self, rect, testOnly):
        x = rect.x()
        y = rect.y()
        lineHeight = 0

        for item in self.itemList:
            wid = item.widget()
            spaceX = self.spacing() + wid.style().layoutSpacing(
                QSizePolicy.PushButton, QSizePolicy.PushButton, Qt.Horizontal
            )
            spaceY = self.spacing() + wid.style().layoutSpacing(
                QSizePolicy.PushButton, QSizePolicy.PushButton, Qt.Vertical
            )
            nextX = x + item.sizeHint().width() + spaceX
            if nextX - spaceX > rect.right() and lineHeight > 0:
                x = rect.x()
                y = y + lineHeight + spaceY
                nextX = x + item.sizeHint().width() + spaceX
                lineHeight = 0

            if not testOnly:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))

            x = nextX
            lineHeight = max(lineHeight, item.sizeHint().height())

        return y + lineHeight - rect.y()


class SelectableCard(QFrame):
    toggled = pyqtSignal(bool, str)

    def __init__(self, name, icon_path, parent=None):
        super().__init__(parent)
        self.name = name
        self.is_selected = False

        self.setFixedSize(120, 100)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)

        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)

        # Image
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        if os.path.exists(icon_path):
            pixmap = QPixmap(icon_path)
            self.image_label.setPixmap(
                pixmap.scaled(100, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
            # Ensure path is forward slash for HTML
            html_path = icon_path.replace("\\", "/")
            self.setToolTip(f"<b>{name}</b><br><img src='{html_path}' width='400'>")
        else:
            self.image_label.setText("No Image")
            self.setToolTip(name)

        # Text
        self.text_label = QLabel(name)
        self.text_label.setAlignment(Qt.AlignCenter)
        self.text_label.setStyleSheet("font-weight: bold; font-size: 10px;")

        layout.addWidget(self.image_label)
        layout.addWidget(self.text_label)
        self.setLayout(layout)

        self.update_style()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_selected = not self.is_selected
            self.update_style()
            self.toggled.emit(self.is_selected, self.name)

    def update_style(self):
        if self.is_selected:
            self.setStyleSheet(
                """
                SelectableCard {
                    background-color: #e0f7fa;
                    border: 2px solid #00acc1;
                    border-radius: 5px;
                }
            """
            )
        else:
            self.setStyleSheet(
                """
                SelectableCard {
                    background-color: white;
                    border: 1px solid #cfd8dc;
                    border-radius: 5px;
                }
                SelectableCard:hover {
                    background-color: #f5f5f5;
                }
            """
            )

    def setChecked(self, checked):
        if self.is_selected != checked:
            self.is_selected = checked
            self.update_style()
            self.toggled.emit(self.is_selected, self.name)

    def isChecked(self):
        return self.is_selected


class VisualSelectorWidget(QWidget):
    selectionChanged = pyqtSignal(list)

    def __init__(self, items, parent=None):
        """
        items: list of tuples (name, icon_path)
        """
        super().__init__(parent)

        self.selected_items = []

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Scroll Area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        self.flow_layout = FlowLayout(container)

        self.cards = {}

        for name, icon_path in items:
            card = SelectableCard(name, icon_path)
            card.toggled.connect(self._on_card_toggled)
            self.flow_layout.addWidget(card)
            self.cards[name] = card

        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    def _on_card_toggled(self, checked, name):
        if checked:
            if name not in self.selected_items:
                self.selected_items.append(name)
        else:
            if name in self.selected_items:
                self.selected_items.remove(name)
        self.selectionChanged.emit(self.selected_items)

    def get_selection(self):
        return self.selected_items


# Backward compatibility / Specialized classes
class PlotSelectorWidget(VisualSelectorWidget):
    def __init__(self, parent=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        icons_dir = os.path.join(base_dir, "..", "..", "gui", "icons", "plots")
        icons_dir = os.path.normpath(icons_dir)

        plot_types = [
            ("histogram", os.path.join(icons_dir, "histogram.png")),
            ("KDE plot", os.path.join(icons_dir, "KDE plot.png")),
            ("countplot", os.path.join(icons_dir, "countplot.png")),
            ("ECDF plot", os.path.join(icons_dir, "ECDF plot.png")),
            ("line plot", os.path.join(icons_dir, "line plot.png")),
            ("scatter plot", os.path.join(icons_dir, "scatter plot.png")),
            ("swarm", os.path.join(icons_dir, "swarm.png")),
            ("violin", os.path.join(icons_dir, "violin.png")),
            ("strip", os.path.join(icons_dir, "strip.png")),
            ("boxplot", os.path.join(icons_dir, "boxplot.png")),
            ("boxenplot", os.path.join(icons_dir, "boxenplot.png")),
        ]
        super().__init__(plot_types, parent)


class StatsSelectorWidget(VisualSelectorWidget):
    def __init__(self, parent=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        icons_dir = os.path.join(base_dir, "..", "..", "gui", "icons", "stats")
        icons_dir = os.path.normpath(icons_dir)

        stats_types = [
            ("Compute KS test\np-value?", os.path.join(icons_dir, "ks_test.png")),
            (
                "Compute effect size?\n(Cliff's Delta)",
                os.path.join(icons_dir, "cliffs_delta.png"),
            ),
        ]
        super().__init__(stats_types, parent)
