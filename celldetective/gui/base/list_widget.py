from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QListWidget, QVBoxLayout

from celldetective.gui.base.components import CelldetectiveWidget
from celldetective.gui.base.utils import center_window


class ListWidget(CelldetectiveWidget):
    """
    A customizable widget for displaying and managing a list of items, with the
    ability to add and remove items interactively.

    This widget is built around a `QListWidget` and allows for initialization with
    a set of features. It also provides options to retrieve the items, add new items
    using a custom widget, and remove selected items. The items can be parsed and
    returned as a list, with support for various data types and formatted input (e.g.,
    ranges specified with a dash).

    Parameters
    ----------
    choiceWidget : QWidget
            A custom widget that is used to add new items to the list.
    initial_features : list
            A list of initial items to populate the list widget.
    dtype : type, optional
            The data type to cast the list items to. Default is `str`.

    Attributes
    ----------
    initial_features : list
            The initial set of features or items displayed in the list.
    choiceWidget : QWidget
            The widget used to prompt the user to add new items.
    dtype : type
            The data type to convert items into when retrieved from the list.
    items : list
            A list to store the current items in the list widget.
    list_widget : QListWidget
            The core Qt widget that displays the list of items.

    Methods
    -------
    addItem()
            Opens a new window to add an item to the list using the custom `choiceWidget`.
    getItems()
            Retrieves the items from the list widget, parsing ranges (e.g., 'min-max')
            into two values, and converts them to the specified `dtype`.
    removeSel()
            Removes the currently selected item(s) from the list widget and updates the
            internal `items` list accordingly.
    """

    def __init__(self, choiceWidget, initial_features, dtype=str, *args, **kwargs):
        """
        Initialize the ListWidget.

        Parameters
        ----------
        choiceWidget : class
            The widget class used to add items.
        initial_features : list
            List of initial features.
        dtype : type, optional
            The data type of the items.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """

        super().__init__()
        self.initial_features = initial_features
        self.choiceWidget = choiceWidget
        self.dtype = dtype
        self.items = []

        self.setFixedHeight(80)

        # Initialize list widget
        self.list_widget = QListWidget()
        self.list_widget.addItems(initial_features)

        # Set up layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.list_widget)
        self.setLayout(main_layout)
        center_window(self)

    def addItem(self):
        """
        Opens the custom choiceWidget to add a new item to the list.
        """

        self.addItemWindow = self.choiceWidget(self)
        self.addItemWindow.show()
        try:
            QTimer.singleShot(10, lambda: center_window(self.addItemWindow))
        except Exception as e:
            pass

    def addItemToList(self, item):
        """
        Add an item to the list widget.

        Parameters
        ----------
        item : str
            The item to add.
        """
        self.list_widget.addItems([item])

    def getItems(self):
        """
        Retrieves and returns the items from the list widget.

        This method parses any items that contain a range (formatted as '(min,max)')
        into a list of two values, and casts all items to the specified `dtype`.

        Returns
        -------
        list
                A list of the items in the list widget, with ranges split into two values.
        """
        import re

        # Pattern for tuple format: "(min,max)" with optional negative numbers and decimals
        tuple_pattern = re.compile(r"^\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)$")

        items = []
        for x in range(self.list_widget.count()):
            text = self.list_widget.item(x).text().strip()

            # Try tuple format first: (min,max)
            tuple_match = tuple_pattern.match(text)
            if tuple_match:
                try:
                    minn = self.dtype(float(tuple_match.group(1)))
                    maxx = self.dtype(float(tuple_match.group(2)))
                    items.append([minn, maxx])
                    continue
                except ValueError:
                    pass

            # Single value
            try:
                items.append(self.dtype(text))
            except ValueError:
                print(
                    f"Warning: Could not convert '{text}' to {self.dtype.__name__}, skipping..."
                )
        return items

    def clear(self):
        """Clear the list widget and the internal items list."""
        self.items = []
        self.list_widget.clear()

    def removeSel(self):
        """
        Removes the selected item(s) from the list widget.

        If there are any selected items, they are removed both from the visual list
        and the internal `items` list that tracks the current state of the widget.
        """

        listItems = self.list_widget.selectedItems()
        if not listItems:
            return
        for item in listItems:
            idx = self.list_widget.row(item)
            if self.items:
                del self.items[idx]
            self.list_widget.takeItem(idx)
