import os
from typing import List

from qtpy.QtCore import QSize, QRectF
from pyqtgraph import parametertree as pt
from pyqtgraph.parametertree.parameterTypes import SimpleParameter
from qtpy.QtWidgets import QDialog, QVBoxLayout, QDialogButtonBox, QPushButton, QLabel, QMessageBox
from qtpy.QtCore import Qt

from .parameter_types import RectROIParameter


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


class ROIDialog(ParameterDialog):
    def __init__(self, image, message='ROI'):
        self.image = image
        super(ROIDialog, self).__init__(children=[RectROIParameter(name='ROI', value=image, message=message),
                                                  SimpleParameter(name='Apply to all', value=True, type='bool')])
        self.parameter_tree.sizeHint = lambda *_: QSize(880, 800)
        # self.layout().setSizeConstraint(QLayout.SetFixedSize)

    @property
    def roi(self):
        return self.parameter.children()[0].children()[0].roi

    def accept(self):
        if not QRectF(0,0, *self.image.shape).contains(QRectF(*self.roi.pos(), *self.roi.size())):
            msg = QMessageBox()
            msg.setText('The selected region extends beyond the bounds of the image. This may cause errors during export. Continue?')
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
            msg.setDefaultButton(QMessageBox.Cancel)
            result = msg.exec_()
            if result != QMessageBox.Yes:
                return
        super().accept()


def confirm_writable(filename):
    if not overwrite_if_exists(filename):
        raise InterruptedError('Could not save to existing path. Cancelled by user.')
    if not check_writable(os.path.dirname(filename)):
        raise PermissionError('Could not write to directory.')


def check_writable(filename):
    return os.access('/path/to/folder', os.W_OK)


def overwrite_if_exists(filename):
    if os.path.exists(filename):
        return overwrite_dialog(filename)
    else:
        return True


def overwrite_dialog(filename):
    btn = QMessageBox.warning(None,
                        "Confirm Save As",
                        f"{os.path.basename(filename)} already exists.\n Do you want to replace it?",
                        buttons=QMessageBox.Yes | QMessageBox.No,
                        defaultButton=QMessageBox.No)
    return btn == QMessageBox.Yes


if __name__ == '__main__':
    import numpy as np
    from qtpy.QtWidgets import QApplication

    qapp = QApplication([])

    dialog = ROIDialog(np.fromfunction(lambda y, x: y, shape=(100, 200)))
    dialog.exec_()

    qapp.exec_()


