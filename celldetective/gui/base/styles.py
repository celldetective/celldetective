from celldetective import get_software_location
from PyQt5.QtGui import QIcon
import os

CELLDETECTIVE_BLUE = "#1565c0"

# Tooltips, styled application wide (see __main__): a light card marked with a
# celldetective blue edge, in the spirit of the outlined buttons.
#
# The padding is kept at nearly zero on purpose. Qt derives the margin of the
# tooltip label from the *left* padding and border of this rule and applies it
# to the four edges, so horizontal padding is paid for in height as well: with
# `padding: 2px 6px` the card was 41px tall for a 13px line of text, against
# 27px here, the room around the text coming from the derived margin alone.
#
# The font is not set here either: a font set through a style sheet is not seen
# by the label when it computes its size. It is set with ``QToolTip.setFont``
# instead (see ``TOOLTIP_FONT_SIZE`` and __main__), so that the card is laid out
# with the font it is painted with.
TOOLTIP_STYLE = f"""
    QToolTip {{
        background-color: #FBFCFE;
        color: #263238;
        border: 1px solid #D6E2F2;
        border-left: 3px solid {CELLDETECTIVE_BLUE};
        border-radius: 4px;
        padding: 0px 1px;
    }}
"""

# Point size of the tooltip text, applied with QToolTip.setFont in __main__.
TOOLTIP_FONT_SIZE = 8

# Scroll bars, styled application wide (see __main__): a thin handle on an
# empty track, with no arrow buttons at either end. The width is the one thing
# the panels depend on, `keep_scrollbar_space` reserving it from the start so
# that a panel does not shift sideways when a bar appears; it reads it off the
# size hint of the bar, so the value below is the only place to change it.
SCROLLBAR_WIDTH = 10
SCROLLBAR_STYLE = f"""
    QScrollBar:vertical, QScrollBar:horizontal {{
        background: transparent;
        border: none;
        margin: 0px;
    }}
    QScrollBar:vertical {{ width: {SCROLLBAR_WIDTH}px; }}
    QScrollBar:horizontal {{ height: {SCROLLBAR_WIDTH}px; }}

    QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
        background: #C5CDD4;
        border-radius: {SCROLLBAR_WIDTH // 2 - 1}px;
        margin: 2px;
    }}
    QScrollBar::handle:vertical {{ min-height: 28px; }}
    QScrollBar::handle:horizontal {{ min-width: 28px; }}
    QScrollBar::handle:hover {{ background: #AAB4BD; }}
    QScrollBar::handle:pressed {{ background: #8F9BA5; }}

    /* The arrow buttons and the track on either side of the handle: given no
       size, they take none, which is what leaves the bar a plain track. */
    QScrollBar::add-line, QScrollBar::sub-line {{
        width: 0px;
        height: 0px;
    }}
    QScrollBar::add-page, QScrollBar::sub-page {{
        background: transparent;
    }}
"""


# Colors of the buttons. Every role below is built from these, so that a state
# looks the same wherever it appears: a pressed button darkens the accent, a
# disabled one goes grey, and the tint under a hovered borderless button is the
# one the item delegates and the collapsible headers already use.
ACCENT_HOVER = "#1B74D8"
ACCENT_PRESSED = "#0D47A1"
ACCENT_SOFT = "#EAF2FC"
ACCENT_SOFT_STRONG = "#D6E6F8"
DANGER_COLOR = "#C62828"
DANGER_SOFT = "#FBEBEA"
DANGER_SOFT_STRONG = "#F6DAD8"
INK_COLOR = "#1C2B36"
SURFACE_COLOR = "#F1F4F7"
SURFACE_BORDER = "#E0E5EA"
SURFACE_HOVER = "#ECEFF1"
SURFACE_PRESSED = "#DDE3E8"
# Borderless icon buttons keep the frank grey they have always had under the
# mouse: light tints read as nothing at all behind a small icon.
GHOST_HOVER = "#BDBDBD"
GHOST_PRESSED = "#A8A8A8"
DISABLED_BG = "#a8b3bd"
# White is only for text on DISABLED_BG, i.e. a filled button.
DISABLED_FG = "#ffffff"
# A disabled glyph or label drawn straight on a light surface: an icon, an
# outlined button, a missing value. White there is invisible.
DISABLED_INK = "#98A3AD"

# The icon buttons closing a row: the cogs, the eyes, the helpers. They rest at
# full weight, in the ink of the software rather than raw black.
#
# Muting them was tried and undone. These are thin outline glyphs: they carry
# far less weight on the page than their contrast against it suggests, and a
# greyed one stops looking like something to press -- it reads as a control
# that is unavailable, which is what DISABLED_INK is for. What keeps the tools
# from crowding a row is where they sit, not how pale they are: the decorative
# icons are gone from the labels, and what is left is lined up in the fixed
# columns of `tool_strip`. The accent they take under the mouse marks them as
# tools without costing them the look of being clickable.
TOOL_IDLE_COLOR = INK_COLOR
TOOL_BUTTON_SIZE = 28
TOOL_ICON_SIZE = 20

# The card a menu, a popup or a collapsible block is drawn on. Kept here rather
# than in the modules that paint them, so that the frames of the software all
# come from the same two values.
CARD_COLOR = "#FFFFFF"
CARD_BORDER_COLOR = "#E3E7EB"

# Progress bars, painted application wide by CelldetectiveStyle (see
# ``_draw_progress_bar``): the track is the surface grey of the chips and the
# chunk the accent, in one continuous bar. They are deliberately *not* styled
# with a style sheet. A style sheet cannot write the label in two colors (white
# over the chunk, ink over the track), and a chunk given a border-radius is not
# drawn at all until it is wider than its two rounded ends, so the bar sat
# empty for the first few percent and then jumped.
PROGRESSBAR_HEIGHT = 18
PROGRESSBAR_RADIUS = 6
PROGRESSBAR_FONT_SIZE = 10
PROGRESSBAR_DISABLED_BORDER = "#EDF0F3"
PROGRESSBAR_DISABLED_CHUNK = "#D3D9DF"

# Secondary text: counts, hints, the index of a table. Darker than the grey of
# a disabled control, so that it reads as information rather than as something
# unavailable.
MUTED_INK = "#66737D"

# Data tables (see ``DataTableView``). The grid is barely there and the header
# is the surface grey of the chips: the numbers are what the eye should land
# on. A selection takes the soft accent tint rather than the solid accent, which
# turned a selected column into a block of white on blue that could not be read.
TABLE_GRID_COLOR = "#EEF1F4"
TABLE_STYLE = f"""
    QTableView {{
        background-color: {CARD_COLOR};
        color: {INK_COLOR};
        border: 1px solid {CARD_BORDER_COLOR};
        gridline-color: {TABLE_GRID_COLOR};
        selection-background-color: {ACCENT_SOFT_STRONG};
        selection-color: {INK_COLOR};
        outline: 0;
    }}
    QHeaderView {{
        background-color: {SURFACE_COLOR};
        border: none;
    }}
    QHeaderView::section {{
        background-color: {SURFACE_COLOR};
        color: {INK_COLOR};
        border: none;
        border-right: 1px solid {SURFACE_BORDER};
        border-bottom: 1px solid {SURFACE_BORDER};
        padding: 5px 8px;
        font-weight: bold;
    }}
    QHeaderView::section:vertical {{
        color: {MUTED_INK};
        font-weight: normal;
        padding: 0px 8px;
    }}
    QHeaderView::section:checked {{
        background-color: {ACCENT_SOFT_STRONG};
        color: {ACCENT_PRESSED};
    }}
    QTableCornerButton::section {{
        background-color: {SURFACE_COLOR};
        border: none;
        border-right: 1px solid {SURFACE_BORDER};
        border-bottom: 1px solid {SURFACE_BORDER};
    }}
    QTableView QAbstractScrollArea::corner, QAbstractScrollArea::corner {{
        background: transparent;
        border: none;
    }}
"""

# The menus are deliberately *not* styled here, unlike the tooltips and the
# scroll bars: the moment a style sheet matches a QMenu, Qt hands the whole
# widget to the style sheet style, which paints the entries itself and never
# asks the application style for them. The celldetective tick and the accent
# pill of a highlighted entry would be lost. They are painted by
# CelldetectiveStyle instead (see ``_draw_menu_item``), frame included.

# The roles a button can take. `primary` is the one action a panel is for
# (Submit), `secondary` an outlined action next to it (Explore table),
# `secondary_plain` the same with a plain label, `chip` a small discrete action
# (the model zoo buttons), `ghost` a borderless icon button, `danger` a
# destructive action, `tool` one of the round icon buttons closing a row and
# `add` a left aligned entry of a list.
BUTTON_ROLES = (
    "primary",
    "secondary",
    "secondary_plain",
    "chip",
    "ghost",
    "tool",
    "danger",
    "add",
)

# Every role carries a transparent border of this width, colored on focus: the
# ring then costs no layout, so a focused button does not move or resize.
FOCUS_WIDTH = 2


def button_style(role: str = "primary") -> str:
    """
    Return the style sheet of a button role.

    Parameters
    ----------
    role : str
        One of :data:`BUTTON_ROLES`.

    Returns
    -------
    str
        The style sheet to give to the buttons of that role.

    Raises
    ------
    ValueError
        If the role is not one of :data:`BUTTON_ROLES`.
    """

    if role not in BUTTON_ROLES:
        raise ValueError(f"Unknown button role '{role}', expected one of {BUTTON_ROLES}.")

    focus = f"QPushButton:focus {{ border-color: {ACCENT_PRESSED}; }}"

    if role == "primary":
        return f"""
            QPushButton {{
                background-color: {CELLDETECTIVE_BLUE};
                color: white;
                border: {FOCUS_WIDTH}px solid transparent;
                border-radius: 13px;
                padding: 5px 14px;
                font-weight: bold;
                font-size: 12px;
            }}
            QPushButton:hover {{ background-color: {ACCENT_HOVER}; }}
            QPushButton:pressed {{ background-color: {ACCENT_PRESSED}; }}
            QPushButton:disabled {{
                background-color: {DISABLED_BG};
                color: {DISABLED_FG};
            }}
            {focus}
        """

    if role in ("secondary", "secondary_plain"):
        label = CELLDETECTIVE_BLUE if role == "secondary" else INK_COLOR
        weight = "bold" if role == "secondary" else "normal"

        return f"""
            QPushButton {{
                background-color: transparent;
                border: 1.6px solid {CELLDETECTIVE_BLUE};
                color: {label};
                border-radius: 13px;
                padding: 6px 14px;
                font-weight: {weight};
                font-size: 12px;
            }}
            QPushButton:hover {{ background-color: {ACCENT_SOFT}; }}
            QPushButton:pressed {{
                background-color: {ACCENT_SOFT_STRONG};
                border-color: {ACCENT_PRESSED};
            }}
            QPushButton:disabled {{
                border-color: {DISABLED_BG};
                color: {DISABLED_INK};
            }}
            QPushButton:focus {{
                border-color: {ACCENT_PRESSED};
                background-color: {ACCENT_SOFT};
            }}
        """

    if role == "chip":
        return f"""
            QPushButton {{
                background-color: {SURFACE_COLOR};
                color: {INK_COLOR};
                border: 1px solid {SURFACE_BORDER};
                border-radius: 11px;
                padding: 3px 11px;
                font-size: 10px;
            }}
            QPushButton:hover {{
                background-color: {SURFACE_HOVER};
                border-color: #CBD4DC;
            }}
            QPushButton:pressed {{ background-color: {SURFACE_PRESSED}; }}
            QPushButton:disabled {{
                background-color: #F5F7F9;
                color: {DISABLED_INK};
                border-color: #EDF0F3;
            }}
            {focus}
        """

    if role == "ghost":
        return f"""
            QPushButton {{
                background-color: transparent;
                color: {INK_COLOR};
                border: {FOCUS_WIDTH}px solid transparent;
                border-radius: 14px;
                padding: 3px;
                font-size: 9px;
            }}
            QPushButton:hover {{ background-color: {GHOST_HOVER}; }}
            QPushButton:pressed {{ background-color: {GHOST_PRESSED}; }}
            QPushButton:checked {{
                background-color: {ACCENT_SOFT_STRONG};
                color: {ACCENT_PRESSED};
            }}
            {focus}
        """

    if role == "tool":
        # No frank grey behind the icon here, unlike `ghost`: a tool button
        # recolors its own icon under the mouse (see ToolButton), which says
        # enough on its own, and a light disc sits better under a small glyph
        # than the grey block it would otherwise need.
        return f"""
            QPushButton {{
                background-color: transparent;
                border: {FOCUS_WIDTH}px solid transparent;
                border-radius: {TOOL_BUTTON_SIZE // 2}px;
                padding: 0px;
            }}
            QPushButton:hover {{ background-color: {ACCENT_SOFT}; }}
            QPushButton:pressed {{ background-color: {ACCENT_SOFT_STRONG}; }}
            QPushButton:checked {{ background-color: {ACCENT_SOFT_STRONG}; }}
            QPushButton:disabled {{ background-color: transparent; }}
            {focus}
        """

    if role == "danger":
        return f"""
            QPushButton {{
                background-color: transparent;
                border: 1.6px solid {DANGER_COLOR};
                color: {DANGER_COLOR};
                border-radius: 13px;
                padding: 6px 14px;
                font-weight: bold;
                font-size: 12px;
            }}
            QPushButton:hover {{ background-color: {DANGER_SOFT}; }}
            QPushButton:pressed {{ background-color: {DANGER_SOFT_STRONG}; }}
            QPushButton:disabled {{
                border-color: {DISABLED_BG};
                color: {DISABLED_INK};
            }}
            QPushButton:focus {{
                border-color: {DANGER_COLOR};
                background-color: {DANGER_SOFT};
            }}
        """

    return f"""
        QPushButton {{
            background-color: transparent;
            color: {INK_COLOR};
            border: {FOCUS_WIDTH}px solid transparent;
            border-radius: 13px;
            padding: 5px 7px;
            font-size: 12px;
            text-align: left;
        }}
        QPushButton:hover {{ background-color: {GHOST_HOVER}; }}
        QPushButton:pressed {{ background-color: {GHOST_PRESSED}; }}
        {focus}
    """


class Styles(object):

    def __init__(self):
        """Initialize the Styles class."""

        self.init_button_styles()
        self.init_tab_styles()
        self.init_label_styles()

        self.help_color = "#1958b7"

        self.celldetective_blue = CELLDETECTIVE_BLUE
        self.celldetective_logo_path = os.sep.join(
            [get_software_location(), "celldetective", "icons", "logo.png"]
        )
        self.celldetective_icon = QIcon(self.celldetective_logo_path)

        self.action_lbl_style_sheet = """
			font-size: 10px;
			padding-left: 10px;
			"""


    def init_button_styles(self):
        """
        Initialize button styles.

        The styles are built from the roles of :func:`button_style`. The names
        the software has always used are kept as aliases of those roles, since
        they are set in about two hundred places.
        """

        self.button_primary = button_style("primary")
        self.button_secondary = button_style("secondary")
        self.button_secondary_plain = button_style("secondary_plain")
        self.button_chip = button_style("chip")
        self.button_ghost = button_style("ghost")
        self.button_danger = button_style("danger")

        self.button_style_sheet = self.button_primary
        self.button_style_sheet_2 = self.button_secondary
        self.button_style_sheet_5 = self.button_secondary_plain
        self.button_style_sheet_3 = self.button_chip
        self.button_style_sheet_2_not_done = self.button_danger
        self.button_select_all = self.button_ghost

        # The indicator itself is painted by CelldetectiveStyle, hover included,
        # so no ::indicator rule here: it would draw a square behind it.
        self.menu_check_style = """
			QCheckBox {
				font-size: 10px;
				padding-left: 10px;
				padding-top: 5px;
			}
		"""

        self.button_add = button_style("add")

    def init_tab_styles(self):
        """Initialize tab styles."""

        self.qtab_style = """
			QTabWidget::pane {
			border: 1px solid #B8B8B8;
			background: white;
		}

		QTabWidget::tab-bar:top {
			top: 1px;
		}

		QTabWidget::tab-bar:bottom {
			bottom: 3px solid blue;
		}

		QTabWidget::tab-bar:left {
			right: 1px;
		}

		QTabWidget::tab-bar:right {
			left: 1px;
		}

		QTabBar::tab {
			border: 1px solid #B8B8B8;
		}

		QTabBar::tab:selected {
			background: white;
		}

		QTabBar::tab:!selected {
			background: silver;
		}

		QTabBar::tab:!selected:hover {
			background: #999;
		}

		QTabBar::tab:top:!selected {
			margin-top: 3px;
		}

		QTabBar::tab:bottom:!selected {
			margin-bottom: 3px;
		}

		QTabBar::tab:top, QTabBar::tab:bottom {
			min-width: 8ex;
			margin-right: -1px;
			padding: 5px 10px 5px 10px;
		}

		QTabBar::tab:top:selected {
			border-bottom: 4px solid #1565c0;
		}


		QTabBar::tab:top:last, QTabBar::tab:bottom:last,
		QTabBar::tab:top:only-one, QTabBar::tab:bottom:only-one {
			margin-right: 0;
		}

		QTabBar::tab:left:!selected {
			margin-right: 3px;
		}

		QTabBar::tab:right:!selected {
			margin-left: 3px;
		}

		QTabBar::tab:left, QTabBar::tab:right {
			min-height: 8ex;
			margin-bottom: -1px;
			padding: 10px 5px 10px 5px;
		}

		QTabBar::tab:left:selected {
			border-left-color: none;
		}

		QTabBar::tab:right:selected {
			border-right-color: none;
		}

		QTabBar::tab:left:last, QTabBar::tab:right:last,
		QTabBar::tab:left:only-one, QTabBar::tab:right:only-one {
			margin-bottom: 0;
		}
		"""

    def init_label_styles(self):
        """Initialize label styles."""

        self.block_title = """
			font-weight: bold;
			padding: 0px;
		"""
