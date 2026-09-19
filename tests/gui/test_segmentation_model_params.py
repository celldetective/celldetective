import pytest
from unittest.mock import patch, MagicMock
from PyQt5.QtWidgets import QMessageBox

from celldetective.gui.settings._segmentation_model_params import SegModelParamsWidget

def test_seg_model_params_widget_abort_on_missing_model(qtbot):
    """Test that SegModelParamsWidget aborts cleanly when model cannot be located."""
    mock_parent = MagicMock()
    mock_parent.parent_window = MagicMock()
    mock_parent.parent_window.locate_image = MagicMock()
    mock_parent.parent_window.current_stack = None

    # Patch QMessageBox.exec to return immediately without showing UI
    with patch.object(QMessageBox, "exec", return_value=QMessageBox.Ok) as mock_exec:
        with pytest.raises(ValueError) as excinfo:
            widget = SegModelParamsWidget(parent_window=mock_parent, model_name="NON_EXISTENT_MODEL_NAME")
        
        # Verify the exception message
        assert "NON_EXISTENT_MODEL_NAME could not be located or loaded" in str(excinfo.value)
        # Verify that QMessageBox.exec was indeed called to alert the user
        mock_exec.assert_called_once()


class TestCellSizeRow:
    """
    The cell size is what a model is rescaled against, so every model that has
    a trained size must offer it -- including a generic Cellpose model, which
    records that size in pixels rather than in microns.

    The row is also what decides whether a cell size is saved at all. It used to
    be built for every model and merely hidden for those without one, so
    ``set_selected_channels_for_segmentation`` -- which tests for the field --
    wrote its placeholder 40 um as though the user had asked for it.
    """

    def _widget(self, qtbot, config, monkeypatch, exp_channels=("brightfield_channel",)):
        from celldetective.gui.settings import _segmentation_model_params as smp

        monkeypatch.setattr(smp.SegModelParamsWidget, "locate_model_path", lambda s: None)

        parent = MagicMock()
        parent.parent_window.locate_image = MagicMock()
        parent.parent_window.exp_channels = list(exp_channels)

        widget = SegModelParamsWidget.__new__(SegModelParamsWidget)
        # Built by hand rather than through __init__: the real constructor reads a
        # model off disk, and what is under test is only how the config is read.
        super(SegModelParamsWidget, widget).__init__()
        widget.parent_window = parent
        widget.attr_parent = parent.parent_window
        widget.input_config = config
        widget.required_channels = config.get("channels", [])
        widget.model_name = "test-model"
        from PyQt5.QtGui import QDoubleValidator
        from PyQt5.QtWidgets import QVBoxLayout

        widget.onlyFloat = QDoubleValidator()
        widget.layout = QVBoxLayout()
        widget.populate_widgets()
        widget.setLayout(widget.layout)
        qtbot.addWidget(widget)
        return widget

    def test_a_declared_cell_size_is_offered(self, qtbot, monkeypatch):
        widget = self._widget(
            qtbot,
            {"channels": ["brightfield_channel"], "cell_size_um": 9.211},
            monkeypatch,
        )
        assert hasattr(widget, "diameter_le")
        assert float(widget.diameter_le.get_threshold()) == pytest.approx(9.211)

    def test_a_generic_cellpose_model_is_offered_one_too(self, qtbot, monkeypatch):
        """Derived from the pixel diameter it was trained on: 30 px x 0.5 um/px."""
        widget = self._widget(
            qtbot,
            {
                "channels": ["brightfield_channel"],
                "model_type": "cellpose",
                "diameter": 30.0,
                "spatial_calibration": 0.5,
            },
            monkeypatch,
        )
        assert hasattr(widget, "diameter_le")
        assert float(widget.diameter_le.get_threshold()) == pytest.approx(15.0)

    def test_the_row_reopens_on_the_size_last_saved(self, qtbot, monkeypatch):
        widget = self._widget(
            qtbot,
            {
                "channels": ["brightfield_channel"],
                "cell_size_um": 9.211,
                "target_cell_size_um": 25.0,
            },
            monkeypatch,
        )
        assert float(widget.diameter_le.get_threshold()) == pytest.approx(25.0)

    def test_a_model_without_a_trained_size_has_no_field_to_save(
        self, qtbot, monkeypatch
    ):
        """
        No field means no cell size is written when the dialog is applied, rather
        than a placeholder nobody entered being saved and rescaling later runs.
        """
        widget = self._widget(
            qtbot,
            {"channels": ["brightfield_channel"], "model_type": "stardist"},
            monkeypatch,
        )
        assert not hasattr(widget, "diameter_le")
