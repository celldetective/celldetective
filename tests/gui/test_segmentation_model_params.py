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
