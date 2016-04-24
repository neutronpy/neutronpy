# -*- coding: utf-8 -*-
r'''GUI for resolution calculations

TESTING ONLY FOR NOW
'''
import os
import sys
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QSizePolicy, QVBoxLayout, QTextEdit
from .instrument import Instrument, GetTau
from .energy import Energy
from .instrument.plot import PlotInstrument

matplotlib.rc('font', **{'family': 'serif', 'serif': 'Times New Roman', 'size': 9})
matplotlib.rc('lines', markersize=2, linewidth=0.5)


class MyMplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=261, height=201, dpi=100, qslice='QxQy', projections=dict(), u=[1, 0, 0], v=[0, 1, 0]):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='w', edgecolor='k')
        self.fig.subplots_adjust(bottom=0.25, left=0.25)

        self.axes = self.fig.add_subplot(111)
        self.axes.hold(True)

        self.compute_initial_figure(self.axes, qslice, projections, u, v)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self, qslice, projections, u, v):
        pass


class MyStaticMplCanvas(MyMplCanvas, PlotInstrument):
    def __init__(self, *args, **kwargs):
        super(MyStaticMplCanvas, self).__init__(*args, **kwargs)

    def compute_initial_figure(self, axis, qslice, projections, u, v):
        self.plot_slice(axis, qslice, projections, u, v)


class MainWindow(QMainWindow):
    r'''Main Window of Resolution Calculator

    '''
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        uic.loadUi(os.path.join(os.path.dirname(__file__), 'ui/resolution.ui'), self)

        self.qxqyplot = QVBoxLayout(self.qx_qy_plot_widget)
        self.qxwplot = QVBoxLayout(self.qx_w_plot_widget)
        self.qywplot = QVBoxLayout(self.qy_w_plot_widget)

        self.text_output.setFontPointSize(10)
        self.text_output.setLineWrapMode(QTextEdit.NoWrap)

        self.dir_dict = {'Clockwise': 1, 'Counter-Clockwise': -1}
        self.infin_dict = {'ki': 1, 'kf': -1}
        self.edrop_dict = {'energy (meV)': float(self.energy_input.text()),
                           'wavelength (A)': Energy(wavelength=float(self.energy_input.text())).energy,
                           'wave vector (A-1)': Energy(wavevector=float(self.energy_input.text())).energy}
        self.method_dict = {'Cooper-Nathans': 0, 'Popovici': 1}
        self.moncor_dict = {'On': 1, 'Off': 0}

        self.load_instrument()

        self.load_signals()

    def load_instrument(self):
        self.instrument = Instrument()

        self.instrument.sample.a, self.instrument.sample.b, self.instrument.sample.c = [float(i) for i in self.abc_input.text().split(',')]
        self.instrument.sample.alpha, self.instrument.sample.beta, self.instrument.sample.gamma = [float(i) for i in self.abg_input.text().split(',')]
        self.instrument.sample.height = float(self.sample_height_input.text())
        self.instrument.sample.width = float(self.sample_width_input.text())
        self.instrument.sample.depth = float(self.sample_depth_input.text())
        self.instrument.sample.u = [float(i) for i in self.sample_u_input.text().split(',')]
        self.instrument.sample.v = [float(i) for i in self.sample_v_input.text().split(',')]
        self.instrument.sample.dir = self.dir_dict[self.sample_dir_select.currentText()]
        self.instrument.sample.mosaic = float(self.sample_mosaic_input.text())
        self.instrument.sample.vmosaic = float(self.sample_vmosaic_input.text())
        self.instrument.sample.shape_type = self.sample_shape_dropdown.currentText().lower()

        if self.mono_select_dropdown.currentText() == 'Custom':
            self.instrument.mono.tau = 2 * np.pi / float(self.mono_select_input)
        else:
            self.instrument.mono.tau = GetTau(self.mono_select_dropdown.currentText())
            self.mono_select_input.setText('{0:.3f}'.format(2. * np.pi / GetTau(self.mono_select_dropdown.currentText())))
        self.instrument.mono.mosaic = float(self.mono_mosaic_input.text())
        self.instrument.mono.vmosaic = float(self.mono_vmosaic_input.text())
        self.instrument.mono.dir = self.dir_dict[self.mono_dir_select.currentText()]
        self.instrument.mono.height = float(self.mono_height_input.text())
        self.instrument.mono.width = float(self.mono_width_input.text())
        self.instrument.mono.depth = float(self.mono_depth_input.text())

        if self.ana_select_dropdown.currentText() == 'Custom':
            self.instrument.ana.tau = 2 * np.pi / float(self.ana_select_input)
        else:
            self.instrument.ana.tau = GetTau(self.ana_select_dropdown.currentText())
            self.ana_select_input.setText('{0:.3f}'.format(2. * np.pi / GetTau(self.ana_select_dropdown.currentText())))
        self.instrument.ana.mosaic = float(self.ana_mosaic_input.text())
        self.instrument.ana.vmosaic = float(self.ana_vmosaic_input.text())
        self.instrument.ana.dir = self.dir_dict[self.ana_dir_select.currentText()]
        self.instrument.ana.height = float(self.ana_height_input.text())
        self.instrument.ana.width = float(self.ana_width_input.text())
        self.instrument.ana.depth = float(self.ana_depth_input.text())

        self.instrument.efixed = self.edrop_dict[self.energy_dropdown.currentText()]
        self.instrument.infin = self.infin_dict[self.fixed_kikf_dropdown.currentText()]
        self.instrument.hcol = [float(i) for i in self.hcols_input.text().split(',')]
        self.instrument.vcol = [float(i) for i in self.vcols_input.text().split(',')]
        self.instrument.arms = [float(i) for i in self.arms_input.text().split(',')]

        self.instrument.guide.height = float(self.guide_height_input.text())
        self.instrument.guide.width = float(self.guide_width_input.text())

        self.instrument.detector.height = float(self.detector_height_input.text())
        self.instrument.detector.width = float(self.detector_width_input.text())

        self.instrument.moncor = self.moncor_dict[self.moncor_dropdown.currentText()]
        self.instrument.method = self.method_dict[self.method_dropdown.currentText()]

        if self.mono_hcurve_input.text() != 'None':
            self.instrument.mono.rh = float(self.mono_hcurve_input.text())
        if self.mono_vcurve_input.text() != 'None':
            self.instrument.mono.rv = float(self.mono_vcurve_input.text())

        if self.ana_hcurve_input.text() != 'None':
            self.instrument.ana.rh = float(self.ana_hcurve_input.text())
        if self.mono_vcurve_input.text() != 'None':
            self.instrument.ana.rh = float(self.ana_vcurve_input.text())

        self.q = [float(i) for i in self.q_input.text().split(',')]

        self.instrument.calc_resolution(self.q)
        self.instrument.calc_projections(self.q)

        # TEST CODE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        try:
            self.clearLayout(self.qxqyplot)
            self.clearLayout(self.qxwplot)
            self.clearLayout(self.qywplot)
        except:
            pass

        qxqy = MyStaticMplCanvas(self.qx_qy_plot_widget, width=261, height=201, dpi=100, qslice='QxQy', projections=self.instrument.projections, u=self.instrument.orient1, v=self.instrument.orient2)
        self.qxqyplot.addWidget(qxqy)

        qxw = MyStaticMplCanvas(self.qx_w_plot_widget, width=261, height=201, dpi=100, qslice='QxW', projections=self.instrument.projections, u=self.instrument.orient1, v=self.instrument.orient2)
        self.qxwplot.addWidget(qxw)

        qyw = MyStaticMplCanvas(self.qy_w_plot_widget, width=261, height=201, dpi=100, qslice='QyW', projections=self.instrument.projections, u=self.instrument.orient1, v=self.instrument.orient2)
        self.qywplot.addWidget(qyw)

        self.text_output.setText(self.instrument.description_string())

    def load_signals(self):
        self.method_dropdown.currentIndexChanged.connect(self.load_instrument)
        self.mono_dir_select.currentIndexChanged.connect(self.load_instrument)
        self.sample_dir_select.currentIndexChanged.connect(self.load_instrument)
        self.ana_dir_select.currentIndexChanged.connect(self.load_instrument)
        self.mono_select_dropdown.currentIndexChanged.connect(self.load_instrument)
        self.ana_select_dropdown.currentIndexChanged.connect(self.load_instrument)
        self.moncor_dropdown.currentIndexChanged.connect(self.load_instrument)
        self.fixed_kikf_dropdown.currentIndexChanged.connect(self.load_instrument)

        self.energy_input.editingFinished.connect(self.load_instrument)
        self.mono_select_input.editingFinished.connect(self.load_instrument)
        self.mono_mosaic_input.editingFinished.connect(self.load_instrument)
        self.mono_vmosaic_input.editingFinished.connect(self.load_instrument)
        self.mono_height_input.editingFinished.connect(self.load_instrument)
        self.mono_width_input.editingFinished.connect(self.load_instrument)
        self.mono_depth_input.editingFinished.connect(self.load_instrument)
        self.mono_hcurve_input.editingFinished.connect(self.load_instrument)
        self.mono_vcurve_input.editingFinished.connect(self.load_instrument)

        self.ana_select_input.editingFinished.connect(self.load_instrument)
        self.ana_mosaic_input.editingFinished.connect(self.load_instrument)
        self.ana_vmosaic_input.editingFinished.connect(self.load_instrument)
        self.ana_height_input.editingFinished.connect(self.load_instrument)
        self.ana_width_input.editingFinished.connect(self.load_instrument)
        self.ana_depth_input.editingFinished.connect(self.load_instrument)
        self.ana_hcurve_input.editingFinished.connect(self.load_instrument)
        self.ana_vcurve_input.editingFinished.connect(self.load_instrument)

        self.abc_input.editingFinished.connect(self.load_instrument)
        self.abg_input.editingFinished.connect(self.load_instrument)
        self.sample_mosaic_input.editingFinished.connect(self.load_instrument)
        self.sample_vmosaic_input.editingFinished.connect(self.load_instrument)
        self.sample_height_input.editingFinished.connect(self.load_instrument)
        self.sample_width_input.editingFinished.connect(self.load_instrument)
        self.sample_depth_input.editingFinished.connect(self.load_instrument)
        self.sample_u_input.editingFinished.connect(self.load_instrument)
        self.sample_v_input.editingFinished.connect(self.load_instrument)
        self.sample_shape_dropdown.currentIndexChanged.connect(self.load_instrument)

        self.hcols_input.editingFinished.connect(self.load_instrument)
        self.vcols_input.editingFinished.connect(self.load_instrument)
        self.arms_input.editingFinished.connect(self.load_instrument)

        self.guide_height_input.editingFinished.connect(self.load_instrument)
        self.guide_width_input.editingFinished.connect(self.load_instrument)
        self.detector_height_input.editingFinished.connect(self.load_instrument)
        self.detector_width_input.editingFinished.connect(self.load_instrument)

        self.energy_input.returnPressed.connect(self.load_instrument)

    def clearLayout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            child.widget().deleteLater()


def launch():
    app = QApplication(sys.argv)

    w = MainWindow()
    w.show()

    sys.exit(app.exec_())
