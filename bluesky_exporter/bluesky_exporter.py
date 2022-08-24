import itertools
import shutil
import time
from collections import Counter, deque
from pathlib import Path
import re
import unicodedata

from PyQt5.QtCore import QSize, QSettings
from PyQt5.QtWidgets import QMessageBox, QProgressBar, QSplitter
from qtpy.QtWidgets import QFormLayout, QLineEdit, QPushButton, QVBoxLayout, QToolButton, QFileDialog, QCheckBox, \
    QComboBox
from qtpy.QtWidgets import QApplication, QSizePolicy
from qtpy.QtCore import Qt
from databroker import Broker
from qtpy.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QGroupBox
from qtmodern import styles
from xicam.core.threads import QThreadFutureIterator, invoke_in_main_thread

from xicam.gui.bluesky.databroker_catalog_plugin import SearchingCatalogController
from xicam.gui.widgets.metadataview import MetadataWidget

from .converters import Converter, dialog_relay

__all__ = ['main']

# Consistently styling the pyqtgraph parametertrees across styles with reasonable colors
pyqtgraph_parametertree_fixes = """
QTreeView::item:has-children {
    background: palette(dark);
    color: palette(light);
}
"""

settings = QSettings('camera', 'bluesky-exporter')


class ExportSettings(QGroupBox):

    def __init__(self, *args, **kwargs):
        super(ExportSettings, self).__init__('Export Settings', *args, **kwargs)
        self.setLayout(form_layout := QFormLayout())

        self.export_directory_path = QLineEdit(settings.value('export_dir_path'))
        self.export_directory_button = QToolButton()
        self.export_directory_button.setText('...')
        self.export_directory_path.textChanged.connect(lambda text: settings.setValue('export_dir_path', text))
        # self.munge_filenames = QCheckBox('Use sample name as filename')
        # self.munge_filenames.setChecked(True)
        self.converter = QComboBox()
        for name, converter in Converter.converter_classes.items():
            self.converter.addItem(name, converter)
        self.converter.setCurrentText(settings.value('converter_name'))
        self.converter.currentTextChanged.connect(lambda text: settings.setValue('converter_name', text))

        export_directory_layout = QHBoxLayout()
        export_directory_layout.addWidget(self.export_directory_path)
        export_directory_layout.addWidget(self.export_directory_button)

        form_layout.addRow('Export Directory:', export_directory_layout)
        form_layout.addRow('Data conversion:', self.converter)
        # form_layout.addWidget(self.munge_filenames)

        self.export_directory_button.clicked.connect(self.choose_directory)

    def choose_directory(self):
        path = str(QFileDialog.getExistingDirectory(self, "Select Export Directory"))
        if path:
            self.export_directory_path.setText(path)


class Exporter(QSplitter):
    def __init__(self, broker, *args, **kwargs):
        super(Exporter, self).__init__()

        self.setOrientation(Qt.Horizontal)
        self.setStretchFactor(0, 3)
        self.setStretchFactor(1, 1)

        left_widget = QWidget()
        left_widget.setLayout(vlayout := QVBoxLayout())
        self.addWidget(left_widget)

        self.export_settings_widget = ExportSettings()
        self.browser_widget = SearchingCatalogController(broker)
        self.metadata_widget = MetadataWidget()
        self.catalog_progress_bar = QProgressBar()
        self.export_progress_bar = QProgressBar()
        self.catalog_progress_bar.setFormat('Completed catalog %v of %m...')
        self.catalog_progress_bar.hide()
        self.export_progress_bar.hide()

        vlayout.addWidget(self.export_settings_widget)
        vlayout.addWidget(self.browser_widget)
        vlayout.addWidget(self.catalog_progress_bar)
        vlayout.addWidget(self.export_progress_bar)
        self.addWidget(self.metadata_widget)

        self.browser_widget.sigOpen.connect(self.start_export)
        self.browser_widget.sigPreview.connect(self.metadata_widget.show_catalog)

        self.browser_widget.open_button.setText('Export')

        self.export_thread = QThreadFutureIterator(self.export,
                                                   yield_slot=self.show_progress,
                                                   finished_slot=self.export_finished,
                                                   except_slot=self.export_finished)
        self.export_queue = deque()
        self.completed_counter = 0

    def start_export(self, catalog):
        # self.browser_widget.open_button.setEnabled(False)
        self.export_queue.append(catalog)
        if not self.export_thread.running:
            self.export_thread.start()
            self.export_progress_bar.setValue(0)
            self.export_progress_bar.setMaximum(0)
            self.catalog_progress_bar.setValue(0)
            self.catalog_progress_bar.setMaximum(0)
        else:
            mx = len(self.export_queue) + self.completed_counter + 1
            self.catalog_progress_bar.setMaximum(mx)
            self.catalog_progress_bar.show()
        # self.catalog_progress_bar.show()
        self.export_progress_bar.show()
        self.export_settings_widget.setDisabled(True)

    def export_finished(self):
        # self.browser_widget.open_button.setEnabled(True)
        self.catalog_progress_bar.hide()
        self.export_progress_bar.hide()
        self.export_settings_widget.setEnabled(True)

    def show_progress(self, value, max, catalog_value, catalog_max):
        self.export_progress_bar.setMaximum(max)
        self.export_progress_bar.setValue(value)
        if catalog_max > 1:
            self.catalog_progress_bar.show()
            self.catalog_progress_bar.setValue(catalog_value)
            self.catalog_progress_bar.setMaximum(catalog_max)

    def export(self):
        export_dir = self.export_settings_widget.export_directory_path.text()

        converter = self.export_settings_widget.converter.currentData()(export_dir)

        while not getattr(converter, 'ready', True):
            time.sleep(.1)

        if not export_dir:
            invoke_in_main_thread(QMessageBox.information, self, 'Cannot export', 'Select an export directory.')
            return

        self.completed_counter = 0

        while self.export_queue:
            catalog = self.export_queue.pop()

            for y in converter.convert_run(catalog):
                if isinstance(y, tuple):
                    queue_length = len(self.export_queue) + self.completed_counter + 1
                    yield *y, self.completed_counter, queue_length
                else:
                    print(y)

            self.completed_counter += 1

        # resource_counter = itertools.count()
        #
        # for name, doc in catalog.canonical(fill='no'):
        #     if name == 'start':
        #         sample_name = slugify(doc['sample_name'])
        #
        #     elif name == 'resource':
        #         src_path = Path(doc['root']) / Path(doc['resource_path'])
        #         dest_path = (Path(export_dir) / Path(f"{sample_name}_{next(resource_counter)}")).with_suffix(Path(doc['resource_path']).suffix)
        #         shutil.copy2(src_path, dest_path)


class ExporterWindow(QMainWindow):
    def __init__(self, broker, *args, **kwargs):
        super(ExporterWindow, self).__init__(*args, **kwargs)
        self.exporter = Exporter(broker)
        self.setCentralWidget(self.exporter)
        self.setWindowTitle('Bluesky Exporter')
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)

        dialog_relay.sigShowDialog.connect(self.show_dialog)

    def sizeHint(self):
        return QSize(1300, 800)

    def show_dialog(self, dialog_callable, lock):
        dialog = dialog_callable()
        dialog.exec_()
        lock.return_value(dialog)
        lock.release()


def main():
    import sys
    if len(sys.argv) > 1:
        broker_name = sys.argv[1]
    else:
        broker_name = 'local'

    db = Broker.named(broker_name).v2

    app = QApplication([])
    styles.dark(app)
    app.setStyleSheet(pyqtgraph_parametertree_fixes)

    w = ExporterWindow(db)

    w.show()

    exit(app.exec_())


if __name__ == '__main__':
    main()
