from pyqtgraph.parametertree import ParameterTree
from pyqtgraph.parametertree.parameterTypes import GroupParameter
from xicam.gui.patches.PyQtGraph import ImageParameter, ImageParameterItem
from xicam.gui.widgets.ROI import BetterRectROI


class ROIParameterItem(ImageParameterItem):
    def __init__(self, *args, **kwargs):
        super(ROIParameterItem, self).__init__(*args, **kwargs)

    def makeWidget(self):
        widget = super(ROIParameterItem, self).makeWidget()
        widget.addItem(self.param.roi)
        widget.view.invertY(False)
        widget.setFixedSize(800, 800)
        return widget


class ROIParameter(ImageParameter):
    itemClass = ROIParameterItem

    def __init__(self, roi, *args, **kwargs):
        self.roi = roi
        super(ROIParameter, self).__init__(*args, **kwargs)


class RectROIParameter(GroupParameter):
    def __init__(self, value, message='ROI', *args, **kwargs):
        super(RectROIParameter, self).__init__(*args, **kwargs)
        children = [ROIParameter(name=message,
                                 value=value,
                                 roi=BetterRectROI(pos=(0, 0),
                                                   size=(value.shape[-1],
                                                         value.shape[-2]),
                                                   pen='r',
                                                   scaleSnap=True,
                                                   translateSnap=True),
                                 expanded=True)]
        self.addChildren(children)


if __name__ == "__main__":
    import numpy as np
    from qtpy.QtWidgets import QApplication

    qapp = QApplication([])

    from xicam.Acquire.devices.fastccd import ProductionCamTriggered

    d = ProductionCamTriggered('ES7011:FastCCD:', name='fastccd')
    p = RectROIParameter(name='ROI', value=np.fromfunction(lambda y, x: y, shape=(100, 200)))
    w = ParameterTree()
    w.setParameters(p)
    w.show()

    qapp.exec_()
