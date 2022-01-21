# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------
# Author: Gregoire Vandenschrick
# Date:   20/01/2022
# -----------------------------------------------------------------------------
# Parts of this is based on the following:
#
# https://github.com/soilstack/react_diffuse/
# which uses Matplotlib and hence is quite slow.
# I reused its systems.py and the setup_grid function.
#
# https://github.com/pmneila/jsexp
# while it's using Javascript and THREE.js, it was a source of understanding
# and inspiration, for the brush tool for example.
#
# https://github.com/glumpy/glumpy/blob/master/examples/grayscott.py
#
# which I discovered is very close to this:
# http://vispy.org/examples/demo/gloo/terrain.html
#
# Primary Sources
# 1. [Karl Sims Original](http://karlsims.com/rd.html)
# 1. [Detailed discussion of Gray-Scott Model](http://mrob.com/pub/comp/xmorphia/)
# 1. [Coding Train Reaction-Diffusion (based on Karl Sims)](https://www.youtube.com/watch?v=BV9ny785UNc&t=2100s)
# 1. [Pearson canonical labelling of systems](https://arxiv.org/abs/patt-sol/9304003)
#
# This alternative version display the patterns using a 3D plane that can be oriented
# I followed a good part of the lighting explanations found here for the 3D render fragment:
# http://learnwebgl.brown37.net/index.html
# -----------------------------------------------------------------------------
"""
    Gray-Scott reaction-diffusion model
    -----------------------------------

    3D presentation of a model of Reaction-Diffusion.
    Qt GUI to ease the manipulation of all parameters. Still keys should work
    when the main window is active.
    For more use help, type python3 gs3DQt.py -h.
"""

################################################################################
try:
    from sip import setapi
    setapi("QVariant", 2)
    setapi("QString", 2)
except ImportError:
    pass


# To switch between PyQt5 and PySide2 bindings just change the from import
from PySide6 import QtCore, QtWidgets

from PySide6.QtGui import QPainter, QPainterPath, QBrush, QPen, QColor
from PySide6.QtCore import Qt, QRectF, Slot, Signal

import sys

# from math import pi
import argparse
import textwrap

from gs3D_lib import (GrayScottModel, MainRenderer, Canvas)

# ? Use of this ?
# gl.use_gl('gl+')
# app.use_app('pyside6')

# Provide automatic signal function selection for PyQt5/PySide2
pyqtsignal = QtCore.pyqtSignal if hasattr(QtCore, 'pyqtSignal') else QtCore.Signal

################################################################################


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self,
                 size=(1024, 1024),
                 modelSize=(512,512),
                 specie='alpha_left',
                 cmap='honolulu_r',
                 verbose=False):
        QtWidgets.QMainWindow.__init__(self)

        self.resize(size[0], size[1])
        self.setWindowTitle('3D Gray-Scott Reaction-Diffusion - GregVDS')

        self.canvas = Canvas(size, modelSize, specie, cmap, verbose)
        self.canvas.create_native()
        self.canvas.native.setParent(self)
        self.canvas.measure_fps(0.1, self.show_fps)

        splitter1 = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter1.addWidget(self.canvas.native)

        self.setCentralWidget(splitter1)

        self.createMenuBar()
        self.createModelDock()
        self.createLightingDock()
        self.createPearsonPatternDetailDock()

        self.initializeGui()
        self.connectSignals()

        # FPS message in statusbar:
        self.status = self.statusBar()
        self.status_label = QtWidgets.QLabel('...')
        self.status.addWidget(self.status_label)

    def createMenuBar(self):
        self.menuBar = QtWidgets.QMenuBar()
        self.panelMenu = QtWidgets.QMenu("Panels")
        self.menuBar.addMenu(self.panelMenu)

    def createLightingDock(self):
        self.lightingDock = QtWidgets.QDockWidget('Lighting settings', self)
        self.lightingDock.setFloating(True)

        groupBox = QtWidgets.QGroupBox(self.lightingDock)
        topBox = QtWidgets.QVBoxLayout(groupBox)

        # TODO connect signals of all these auto-generated widgets...
        for lightType in MainRenderer.lightingDictionnary.keys():
            lightTypeBox = QtWidgets.QGroupBox(lightType, self.lightingDock)
            lightTypeLayout = QtWidgets.QGridLayout()
            paramCount = 0
            for param in MainRenderer.lightingDictionnary[lightType].keys():
                lightTypeParam = MainRenderer.lightingDictionnary[lightType][param]
                if lightTypeParam[1] == "bool":
                    lightTypeBox.setCheckable(True)
                    lightTypeBox.setChecked(lightTypeParam[0])

                    lightTypeBox.clicked.connect(self.canvas.mainRenderer.toggleAmbientLight)
                    # lightTypeBox.clicked.emit(self.title)

                elif lightTypeParam[1] == "float":
                    lightTypeLayout.addWidget(QtWidgets.QLabel(param, lightTypeBox), paramCount, 0)
                    lightTypeDoubleSpinBox = QtWidgets.QDoubleSpinBox(lightTypeBox)
                    lightTypeDoubleSpinBox.setDecimals(5)
                    lightTypeDoubleSpinBox.setMinimum(lightTypeParam[2])
                    lightTypeDoubleSpinBox.setMaximum(lightTypeParam[3])
                    lightTypeDoubleSpinBox.setValue(lightTypeParam[0])
                    lightTypeDoubleSpinBox.setSingleStep((lightTypeParam[3] - lightTypeParam[2])/1000)
                    lightTypeLayout.addWidget(lightTypeDoubleSpinBox, paramCount, 1)
                    paramCount += 1
                elif lightTypeParam[1] == "int":
                    lightTypeLayout.addWidget(QtWidgets.QLabel(param, lightTypeBox), paramCount, 0)
                    lightTypeSpinBox = QtWidgets.QSpinBox()
                    lightTypeSpinBox.setMinimum(lightTypeParam[2])
                    lightTypeSpinBox.setMaximum(lightTypeParam[3])
                    lightTypeSpinBox.setValue(lightTypeParam[0])
                    lightTypeLayout.addWidget(lightTypeSpinBox, paramCount, 1)
                    paramCount += 1
                elif lightTypeParam[1] == "color":
                    lightTypeLayout.addWidget(QtWidgets.QLabel(param, lightTypeBox), paramCount, 0)
                    color = QColor(lightTypeParam[0][0]*255,
                                   lightTypeParam[0][1]*255,
                                   lightTypeParam[0][2]*255)
                    lightTypeLayout.addWidget(RoundedButton("", 1, QColor(0,0,0), color), paramCount, 1)
                    paramCount += 1
            lightTypeBox.setLayout(lightTypeLayout)
            topBox.addWidget(lightTypeBox)

        displayBox = QtWidgets.QGroupBox("Display", self.lightingDock)
        displayLayout = QtWidgets.QGridLayout(displayBox)
        self.normalRadioButton = QtWidgets.QRadioButton('Normal', self.lightingDock)
        self.shadowRadioButton = QtWidgets.QRadioButton('Shadowmap', self.lightingDock)
        self.normalRadioButton.setChecked(True)
        displayLayout.addWidget(self.normalRadioButton, 0, 0)
        displayLayout.addWidget(self.shadowRadioButton, 0, 1)
        self.resetCameraButton = QtWidgets.QPushButton("Reset camera", self.lightingDock)
        self.resetShadowButton = QtWidgets.QPushButton("Reset shadow", self.lightingDock)
        displayLayout.addWidget(self.resetCameraButton, 1, 0)
        displayLayout.addWidget(self.resetShadowButton, 1, 1)
        displayBox.setLayout(displayLayout)

        topBox.addWidget(displayBox)
        topBox.addStretch(1)
        groupBox.setLayout(topBox)

        self.lightingDock.setWidget(groupBox)
        # self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.lightingDock)

        self.panelMenu.addAction(self.lightingDock.toggleViewAction())

    def createModelDock(self):
        self.modelDock = QtWidgets.QDockWidget('Model settings', self)
        self.modelDock.setFloating(True)
        groupBox = QtWidgets.QGroupBox(self.modelDock)
        topBox = QtWidgets.QVBoxLayout(groupBox)

        pearsonsBox = QtWidgets.QGroupBox("Pearson' pattern", self.modelDock)
        pearsonsLayout = QtWidgets.QVBoxLayout()
        self.pearsonsPatternsComboBox = QtWidgets.QComboBox(self.modelDock)
        patterns = []
        for key in GrayScottModel.speciesDictionnary.keys():
            patterns.append(GrayScottModel.speciesDictionnary[key])
        for key in GrayScottModel.speciesDictionnaryShifted.keys():
            patterns.append(GrayScottModel.speciesDictionnaryShifted[key])
        patterns.sort()
        self.pearsonsPatternsComboBox.addItems(patterns)
        pearsonsLayout.addWidget(self.pearsonsPatternsComboBox)
        pearsonsBox.setLayout(pearsonsLayout)

        reagentBox = QtWidgets.QGroupBox("Reagent displayed", self.modelDock)
        reagentLayout = QtWidgets.QHBoxLayout(reagentBox)
        self.uReagentRadioButton = QtWidgets.QRadioButton('U', self.modelDock)
        self.vReagentRadioButton = QtWidgets.QRadioButton('V', self.modelDock)
        self.vReagentRadioButton.setChecked(True)
        reagentLayout.addWidget(self.uReagentRadioButton)
        reagentLayout.addWidget(self.vReagentRadioButton)
        reagentBox.setLayout(reagentLayout)

        colorMapBox = QtWidgets.QGroupBox("Colormap used", self.modelDock)
        colorMapLayout = QtWidgets.QVBoxLayout()
        self.colorsComboBox = QtWidgets.QComboBox(self.modelDock)
        colors = []
        for key in MainRenderer.colormapDictionnary.keys():
            colors.append(MainRenderer.colormapDictionnary[key])
        for key in MainRenderer.colormapDictionnaryShifted.keys():
            colors.append(MainRenderer.colormapDictionnaryShifted[key])
        colors.sort()
        self.colorsComboBox.addItems(colors)
        colorMapLayout.addWidget(self.colorsComboBox)
        colorMapBox.setLayout(colorMapLayout)

        controlBox = QtWidgets.QGroupBox("Controls", self.modelDock)
        controlLayout = QtWidgets.QVBoxLayout()
        self.resetButton = QtWidgets.QPushButton("Reset", self.modelDock)
        controlLayout.addWidget(self.resetButton)
        cyclesBox = QtWidgets.QGroupBox("Additional cycles/frame", self.modelDock)
        cyclesLayout = QtWidgets.QHBoxLayout()
        self.lessCycles = QtWidgets.QPushButton("-", self.modelDock)
        self.cycles = QtWidgets.QSpinBox(self.modelDock)
        self.cycles.setReadOnly(True)
        self.cycles.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.cycles.setValue(self.canvas.grayScottModel.cycle)
        self.moreCycles = QtWidgets.QPushButton("+", self.modelDock)
        cyclesLayout.addWidget(self.lessCycles)
        cyclesLayout.addWidget(self.cycles)
        cyclesLayout.addWidget(self.moreCycles)
        cyclesBox.setLayout(cyclesLayout)
        controlLayout.addWidget(cyclesBox)

        controlBox.setLayout(controlLayout)

        topBox.addWidget(pearsonsBox)
        topBox.addWidget(reagentBox)
        topBox.addWidget(controlBox)
        topBox.addWidget(colorMapBox)
        topBox.addStretch(1)

        groupBox.setLayout(topBox)

        self.modelDock.setWidget(groupBox)
        # self.addDockWidget(QtCore.Qt.TopDockWidgetArea, self.modelDock)

        self.panelMenu.addAction(self.modelDock.toggleViewAction())

    def createPearsonPatternDetailDock(self):
        self.pPDetailsDock = QtWidgets.QDockWidget('Pearson\' pattern Details', self)
        self.pPDetailsDock.setFloating(True)

        box = QtWidgets.QGroupBox("", self.pPDetailsDock)
        layout = QtWidgets.QVBoxLayout()
        self.pPDetailsLabel = QtWidgets.QLabel()
        layout.addWidget(self.pPDetailsLabel)
        box.setLayout(layout)

        self.pPDetailsDock.setWidget(box)

        self.panelMenu.addAction(self.pPDetailsDock.toggleViewAction())

    def initializeGui(self):
        self.colorsComboBox.setCurrentText(self.canvas.mainRenderer.cmapName)
        self.pearsonsPatternsComboBox.setCurrentText(self.canvas.grayScottModel.specie)
        self.pPDetailsLabel.setText(self.canvas.grayScottModel.getPearsonPatternDescription())

    def connectSignals(self):
        self.colorsComboBox.textActivated[str].connect(self.canvas.mainRenderer.setColorMap)
        self.colorsComboBox.textActivated[str].emit(self.colorsComboBox.currentText())

        self.pearsonsPatternsComboBox.textActivated[str].connect(self.canvas.grayScottModel.setSpecie)
        self.pearsonsPatternsComboBox.textActivated[str].connect(self.setPearsonsPatternDetails)
        self.pearsonsPatternsComboBox.textActivated[str].emit(self.pearsonsPatternsComboBox.currentText())

        self.vReagentRadioButton.toggled.connect(self.canvas.mainRenderer.switchReagent)

        self.normalRadioButton.toggled.connect(self.canvas.switchDisplay)

        self.resetButton.clicked.connect(self.canvas.grayScottModel.initializeGrid)
        self.lessCycles.clicked.connect(self.canvas.grayScottModel.decreaseCycle)
        self.lessCycles.clicked.connect(self.updateCycle)
        self.moreCycles.clicked.connect(self.canvas.grayScottModel.increaseCycle)
        self.moreCycles.clicked.connect(self.updateCycle)
        self.resetCameraButton.clicked.connect(self.canvas.mainRenderer.resetCamera)
        self.resetShadowButton.clicked.connect(self.canvas.mainRenderer.resetLight)

    def setPearsonsPatternDetails(self):
        self.pPDetailsLabel.setText(self.canvas.grayScottModel.getPearsonPatternDescription())

    def updateCycle(self):
        self.cycles.setValue(self.canvas.grayScottModel.cycle)

    def show_fps(self, fps):
        msg = " FPS - %0.2f" % float(fps)
        # NOTE: We can't use showMessage in PyQt5 because it causes
        #       a draw event loop (show_fps for every drawing event,
        #       showMessage causes a drawing event, and so on).
        self.status_label.setText(msg)

        self.canvas.visible = True


class RoundedButton(QtWidgets.QPushButton):
    def __init__(self, text, bordersize, outlineColor, fillColor):
        super(RoundedButton, self).__init__()
        self.bordersize = bordersize
        self.outlineColor = outlineColor
        self.fillColor = fillColor
        self.setText(text)
        self.colorDialog = QtWidgets.QColorDialog(fillColor, self)
        self.clicked.connect(self.changeColor)

    def changeColor(self):
        color = self.colorDialog.getColor(self.fillColor)
        if color.isValid():
            self.fillColor = color

    def paintEvent(self, event):
        # Create the painter
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        # Create the path
        path = QPainterPath()
        # Set painter colors to given values.
        pen = QPen(self.outlineColor, self.bordersize)
        painter.setPen(pen)
        brush = QBrush(self.fillColor)
        painter.setBrush(brush)

        rect = QRectF(event.rect())
        # Slighly shrink dimensions to account for bordersize.
        rect.adjust(self.bordersize/2, self.bordersize/2, -self.bordersize/2, -self.bordersize/2)

        # Add the rect to path.
        path.addRoundedRect(rect, 10, 10)
        painter.setClipPath(path)

        # Fill shape, draw the border and center the text.
        painter.fillPath(path, painter.brush())
        painter.strokePath(path, painter.pen())
        painter.drawText(rect, Qt.AlignCenter, self.text())

################################################################################
# def fun(x):
#     c.title = c.title2 +' - FPS: %0.1f' % x


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=textwrap.dedent(Canvas.getCommandsDocs()),
                                     epilog= textwrap.dedent("""Examples:
    python3 gs3DQt.py
    python3 gs3DQt.py -c osmort
    python3 gs3DQt.py -s 512 -p kappa_left -c oslo
    python3 gs3DQt.py -s 512 -w 800 -p alpha_left -c detroit"""),
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-s",
                        "--Size",
                        type=int,
                        default=512,
                        help="Size of model")
    parser.add_argument("-w",
                        "--Window",
                        type=int,
                        default=1024,
                        help="Size of window")
    parser.add_argument("-p",
                        "--Pattern",
                        choices={**GrayScottModel.speciesDictionnary,
                                 **GrayScottModel.speciesDictionnaryShifted}.values(),
                        default="alpha_left",
                        help="Pearson\' pattern")
    parser.add_argument("-c",
                        "--Colormap",
                        choices={**MainRenderer.colormapDictionnary,
                                 **MainRenderer.colormapDictionnaryShifted}.values(),
                        default="honolulu_r",
                        help="Colormap used")
    parser.add_argument("-v",
                        "--Verbose",
                        action='store_true',
                        help="outputs comment on changes")

    args = parser.parse_args()

    appQt = QtWidgets.QApplication(sys.argv)
    win = MainWindow(modelSize=(args.Size, args.Size),
                     size=(args.Window, args.Window),
                     specie=args.Pattern,
                     cmap=args.Colormap,
                     verbose=args.Verbose)
    win.show()
    appQt.exec()
