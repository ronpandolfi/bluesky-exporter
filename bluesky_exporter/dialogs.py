from typing import Iterable, List
from pyqtgraph import parametertree as pt
from qtpy.QtWidgets import QDialog, QHBoxLayout, QVBoxLayout, QDialogButtonBox, QPushButton, QLabel
from qtpy.QtCore import Qt, QSettings


class ParameterDialog(QDialog):
    """
    Dialog for modifying parameters
    """

    # _default_parameter_state = pt.parameterTypes.GroupParameter(children=[], name='blah', ).saveState()
    #
    # _qsettings_key = 'bluesky-exporter'
    # _parameter_state = QSettings().value(_qsettings_key, defaultValue=_default_parameter_state)
    # parameter = pt.parameterTypes.GroupParameter(name='blah')
    # parameter.restoreState(_parameter_state)

    def __init__(self, children: List[pt.Parameter], message=None, parent=None, window_flags=Qt.WindowFlags()):
        super(ParameterDialog, self).__init__(parent, window_flags)

        self.parameter_tree = pt.ParameterTree(showHeader=False)
        self.parameter = pt.parameterTypes.GroupParameter(name='blah', children=children)
        # TODO: decide how to re-use overrides?
        # group.restoreState(self.parameter.saveState(), addChildren=False, removeChildren=False)
        self.parameter_tree.setParameters(self.parameter, showTop=False)

        accept_button = QPushButton("&Ok")

        self.buttons = QDialogButtonBox(Qt.Horizontal)
        # Add calibration button that accepts the dialog (closes with 1 status)
        self.buttons.addButton(accept_button, QDialogButtonBox.AcceptRole)
        # Add a cancel button that will reject the dialog (closes with 0 status)
        self.buttons.addButton(QDialogButtonBox.Cancel)

        self.buttons.rejected.connect(self.reject)
        self.buttons.accepted.connect(self.accept)

        outer_layout = QVBoxLayout()

        if message is not None:
            message_label = QLabel(message)
            outer_layout.addWidget(message_label)

        outer_layout.addWidget(self.parameter_tree)
        outer_layout.addWidget(self.buttons)
        outer_layout.setSpacing(0)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(outer_layout)

    def get_parameters(self):
        return {key: value['value'] for key, value in self.parameter.saveState('user')['children'].items()}

    def accept(self):
        super(ParameterDialog, self).accept()
        # QSettings().setValue(self._qsettings_key, self.parameter.saveState())
