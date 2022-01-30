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
from PySide6.QtCore import Qt, QRectF, QPointF, Slot
from PySide6.QtCharts import QChartView, QChart, QScatterSeries

import sys
import math

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

        self.canvas = Canvas(size, modelSize, specie, cmap, verbose, isotropic=False)
        self.canvas.create_native()
        self.canvas.native.setParent(self)
        self.canvas.measure_fps(1.0, self.show_fps)

        splitter1 = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter1.addWidget(self.canvas.native)

        self.setCentralWidget(splitter1)

        self.createMenuBar()
        self.createPearsonPatternDetailDock(visible=True)
        self.createModelDock()
        self.createDisplayDock(visible=False)
        self.createLightingDock(visible=False)

        # FPS message in statusbar:
        self.status = self.statusBar()
        self.status_label = QtWidgets.QLabel('...')
        self.status.addWidget(self.status_label)

    def createMenuBar(self):
        """
        Creates menubar and menus.
        """
        self.menuBar = QtWidgets.QMenuBar()
        self.panelMenu = QtWidgets.QMenu("Panels")
        self.menuBar.addMenu(self.panelMenu)

    def createLightingDock(self, visible=True):
        """
        Creates dock with lighting parameters of the 3D model.
        These are read from MainRenderer lightingDictionnary and produce the
        appropriate groupbox and widgets in it.
        """
        self.lightingDock = QtWidgets.QDockWidget('Lighting settings', self)
        self.lightingDock.setFloating(True)
        self.lightingDock.setVisible(visible)
        topBox = QtWidgets.QGroupBox(self.lightingDock)
        topLayout = QtWidgets.QVBoxLayout(topBox)

        # --------------------------------------
        for lightType in MainRenderer.lightingDictionnary.keys():
            paramCount = 0
            lightTypeBox = LightTypeGroupBox(lightType, self, 'on')
            lightTypeLayout = QtWidgets.QGridLayout()
            for param in MainRenderer.lightingDictionnary[lightType].keys():
                lightTypeParam = MainRenderer.lightingDictionnary[lightType][param]
                if lightTypeParam[1] == "bool":
                    lightTypeBox.setCheckable(True)
                    lightTypeBox.setChecked(lightTypeParam[0])
                elif lightTypeParam[1] == "float":
                    lightTypeLayout.addWidget(QtWidgets.QLabel(param, lightTypeBox), paramCount, 0)
                    lightParamValueLabel = QtWidgets.QLabel(str(lightTypeParam[0]), lightTypeBox)
                    lightTypeLayout.addWidget(lightParamValueLabel, paramCount, 1)
                    paramCount += 1
                    lightParamSlider = LightParamSlider(Qt.Horizontal, self, lightParamValueLabel, lightType, param)
                    lightParamSlider.setMinimum(lightTypeParam[2])
                    lightParamSlider.setMaximum(lightTypeParam[3])
                    lightParamSlider.setValue(lightTypeParam[0])
                    lightParamSlider.valueChanged.connect(lightParamSlider.updateLighting)
                    lightTypeLayout.addWidget(lightParamSlider, paramCount, 0, 1, 2)
                    paramCount += 1
                elif lightTypeParam[1] == "int":
                    lightTypeLayout.addWidget(QtWidgets.QLabel(param, lightTypeBox), paramCount, 0)
                    lightSpinBox = LightParamSpinBox(self, lightType, param)
                    lightSpinBox.setMinimum(lightTypeParam[2])
                    lightSpinBox.setMaximum(lightTypeParam[3])
                    lightSpinBox.setValue(lightTypeParam[0])
                    lightSpinBox.setWrapping(True)
                    lightSpinBox.valueChanged.connect(lightSpinBox.updateLighting)
                    lightTypeLayout.addWidget(lightSpinBox, paramCount, 1)
                    paramCount += 1
                elif lightTypeParam[1] == "color":
                    lightTypeLayout.addWidget(QtWidgets.QLabel(param, lightTypeBox), paramCount, 0)
                    color = QColor(lightTypeParam[0][0]*255,
                                   lightTypeParam[0][1]*255,
                                   lightTypeParam[0][2]*255)
                    colorButton = RoundedButton(lightType, self, 1, QColor(0,0,0), color)
                    lightTypeLayout.addWidget(colorButton, paramCount, 1)
                    paramCount += 1
            lightTypeBox.setLayout(lightTypeLayout)
            topLayout.addWidget(lightTypeBox)

        # --------------------------------------
        topLayout.addStretch(1)
        topBox.setLayout(topLayout)
        self.lightingDock.setWidget(topBox)
        self.panelMenu.addAction(self.lightingDock.toggleViewAction())

    def createDisplayDock(self, visible=True):
        """
        Creates dock with Disply parameters of the 3D model.
        Here are defined the colormap, background, choice of reagents and Camera
        widgets.
        """
        self.displayDock = QtWidgets.QDockWidget('Display settings', self)
        self.displayDock.setFloating(True)
        self.displayDock.setVisible(visible)
        topBox = QtWidgets.QGroupBox(self.displayDock)
        topLayout = QtWidgets.QVBoxLayout(topBox)

        # --------------------------------------
        colorMapBox = QtWidgets.QGroupBox("Colormap", self.modelDock)
        colorMapLayout = QtWidgets.QVBoxLayout()
        self.colorsComboBox = QtWidgets.QComboBox(self.displayDock)
        colors = []
        for key in MainRenderer.colormapDictionnary.keys():
            colors.append(MainRenderer.colormapDictionnary[key])
        for key in MainRenderer.colormapDictionnaryShifted.keys():
            colors.append(MainRenderer.colormapDictionnaryShifted[key])
        colors.sort()
        self.colorsComboBox.addItems(colors)
        self.colorsComboBox.setCurrentText(self.canvas.mainRenderer.cmapName)
        self.colorsComboBox.textActivated[str].connect(self.canvas.mainRenderer.setColorMap)
        self.colorsComboBox.textActivated[str].emit(self.colorsComboBox.currentText())
        colorMapLayout.addWidget(self.colorsComboBox)
        colorMapBox.setLayout(colorMapLayout)
        topLayout.addWidget(colorMapBox)

        # --------------------------------------
        backgroundBox = QtWidgets.QGroupBox("Background", self.modelDock)
        backgroundLayout = QtWidgets.QGridLayout()
        backgroundLayout.addWidget(QtWidgets.QLabel("color", backgroundBox), 0, 0)
        canvasColor = self.canvas.backgroundColor
        color = QColor(canvasColor[0]*255,
                       canvasColor[1]*255,
                       canvasColor[2]*255)
        colorButton = RoundedButton("background", self, 1, QColor(0,0,0), color)
        backgroundLayout.addWidget(colorButton, 0, 1)
        backgroundBox.setLayout(backgroundLayout)
        topLayout.addWidget(backgroundBox)

        # --------------------------------------
        reagentBox = QtWidgets.QGroupBox("Reagent", self.displayDock)
        reagentLayout = QtWidgets.QHBoxLayout(reagentBox)
        self.uReagentRadioButton = QtWidgets.QRadioButton('U', self.displayDock)
        self.vReagentRadioButton = QtWidgets.QRadioButton('V', self.displayDock)
        self.vReagentRadioButton.setChecked(True)
        self.vReagentRadioButton.toggled.connect(self.canvas.mainRenderer.switchReagent)
        reagentLayout.addWidget(self.uReagentRadioButton)
        reagentLayout.addWidget(self.vReagentRadioButton)
        reagentBox.setLayout(reagentLayout)
        topLayout.addWidget(reagentBox)

        # --------------------------------------
        displayBox = QtWidgets.QGroupBox("Camera", self.displayDock)
        displayLayout = QtWidgets.QGridLayout(displayBox)
        self.normalRadioButton = QtWidgets.QRadioButton('Normal', self.displayDock)
        self.shadowRadioButton = QtWidgets.QRadioButton('Shadowmap', self.displayDock)
        self.normalRadioButton.setChecked(True)
        self.normalRadioButton.toggled.connect(self.canvas.switchDisplay)
        displayLayout.addWidget(self.normalRadioButton, 0, 0)
        displayLayout.addWidget(self.shadowRadioButton, 0, 1)
        self.resetCameraButton = QtWidgets.QPushButton("Reset camera", self.displayDock)
        self.resetCameraButton.clicked.connect(self.canvas.mainRenderer.resetCamera)
        self.resetShadowButton = QtWidgets.QPushButton("Reset shadow", self.displayDock)
        self.resetShadowButton.clicked.connect(self.canvas.mainRenderer.resetLight)
        displayLayout.addWidget(self.resetCameraButton, 1, 0)
        displayLayout.addWidget(self.resetShadowButton, 1, 1)
        displayBox.setLayout(displayLayout)
        topLayout.addWidget(displayBox)

        # --------------------------------------
        topLayout.addStretch(1)
        topBox.setLayout(topLayout)
        self.displayDock.setWidget(topBox)
        self.panelMenu.addAction(self.displayDock.toggleViewAction())

    def createModelDock(self, visible=True):
        """
        Creates dock that holds parameters of the reaction-diffusion and Some
        parameters of the cycling of it.
        """
        self.modelDock = QtWidgets.QDockWidget('Model settings', self)
        self.modelDock.setFloating(True)
        self.modelDock.setVisible(visible)
        topBox = QtWidgets.QGroupBox(self.modelDock)
        topLayout = QtWidgets.QVBoxLayout(topBox)

        # --------------------------------------
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
        self.pearsonsPatternsComboBox.setCurrentText(self.canvas.grayScottModel.specie)
        self.pearsonsPatternsComboBox.textActivated[str].connect(self.canvas.grayScottModel.setSpecie)
        self.pearsonsPatternsComboBox.textActivated[str].connect(self.setSelectedPearsonsPatternDetails)
        self.pearsonsPatternsComboBox.textActivated[str].emit(self.pearsonsPatternsComboBox.currentText())
        self.pearsonsPatternsComboBox.textActivated[str].connect(self.setFeedKillDials)
        self.pearsonsPatternsComboBox.textHighlighted[str].connect(self.setHighlightedPearsonsPatternDetails)
        # self.pearsonsPatternsComboBox.textHighlighted[str].emit(self.pearsonsPatternsComboBox.currentText())
        pearsonsLayout.addWidget(self.pearsonsPatternsComboBox)
        pearsonsBox.setLayout(pearsonsLayout)
        topLayout.addWidget(pearsonsBox)

        # --------------------------------------
        fkBox = QtWidgets.QGroupBox(self.modelDock)
        fkLayout = QtWidgets.QVBoxLayout()

        fBox = QtWidgets.QGroupBox(self.modelDock)
        fLayout = QtWidgets.QGridLayout()
        fLayout.addWidget(QtWidgets.QLabel("feed", fBox), 0, 0)
        feedParamLabel = QtWidgets.QLabel("", fBox)
        fLayout.addWidget(feedParamLabel, 0, 1)
        self.feedParamSlider = ParamSlider(Qt.Horizontal, self, feedParamLabel, "feed", 1000.0)
        self.feedParamSlider.setMinimum(self.canvas.grayScottModel.fMin)
        self.feedParamSlider.setMaximum(self.canvas.grayScottModel.fMax)
        self.feedParamSlider.setValue(self.canvas.grayScottModel.baseParams[2])
        self.feedParamSlider.updateParam(0)
        self.feedParamSlider.valueChanged.connect(self.feedParamSlider.updateParam)
        self.feedParamSlider.sliderMoved.connect(self.setCurrentFKInChart)
        fLayout.addWidget(self.feedParamSlider, 1, 0, 1, 2)
        fLayout.addWidget(QtWidgets.QLabel("∂feed/∂x", fBox), 2, 0)
        dFeedParamLabel = QtWidgets.QLabel("", fBox)
        fLayout.addWidget(dFeedParamLabel, 2, 1)
        self.dFeedParamSlider = ParamSlider(Qt.Horizontal, self, dFeedParamLabel, "dFeed", 1000.0)
        self.dFeedParamSlider.setMinimum(0.0)
        self.dFeedParamSlider.setMaximum(0.008)
        self.dFeedParamSlider.setValue(0.0)
        self.dFeedParamSlider.updateParam(0)
        self.dFeedParamSlider.valueChanged.connect(self.dFeedParamSlider.updateParam)
        self.dFeedParamSlider.valueChanged.connect(self.setDFeedInChart)
        fLayout.addWidget(self.dFeedParamSlider, 3, 0, 1, 2)
        fBox.setLayout(fLayout)

        kBox = QtWidgets.QGroupBox(self.modelDock)
        kLayout = QtWidgets.QGridLayout()
        kLayout.addWidget(QtWidgets.QLabel("kill", kBox), 0, 0)
        killParamLabel = QtWidgets.QLabel("", kBox)
        kLayout.addWidget(killParamLabel, 0, 1)
        self.killParamSlider = ParamSlider(Qt.Horizontal, self, killParamLabel, "kill", 1000.0)
        self.killParamSlider.setMinimum(self.canvas.grayScottModel.kMin)
        self.killParamSlider.setMaximum(self.canvas.grayScottModel.kMax)
        self.killParamSlider.setValue(self.canvas.grayScottModel.baseParams[3])
        self.killParamSlider.updateParam(0)
        self.killParamSlider.valueChanged.connect(self.killParamSlider.updateParam)
        self.killParamSlider.sliderMoved.connect(self.setCurrentFKInChart)
        kLayout.addWidget(self.killParamSlider, 1, 0, 1, 2)
        kLayout.addWidget(QtWidgets.QLabel("∂kill/∂y", kBox), 2, 0)
        dKillParamLabel = QtWidgets.QLabel("", kBox)
        kLayout.addWidget(dKillParamLabel, 2, 1)
        self.dKillParamSlider = ParamSlider(Qt.Horizontal, self, dKillParamLabel, "dKill", 1000.0)
        self.dKillParamSlider.setMinimum(0.0)
        self.dKillParamSlider.setMaximum(0.004)
        self.dKillParamSlider.setValue(0.0)
        self.dKillParamSlider.updateParam(0)
        self.dKillParamSlider.valueChanged.connect(self.dKillParamSlider.updateParam)
        self.dKillParamSlider.valueChanged.connect(self.setDKillInChart)
        kLayout.addWidget(self.dKillParamSlider, 3, 0, 1, 2)
        kBox.setLayout(kLayout)

        dUBox = QtWidgets.QGroupBox(self.modelDock)
        dULayout = QtWidgets.QGridLayout()
        dULayout.addWidget(QtWidgets.QLabel("dU", dUBox), 0, 0)
        dUParamLabel = QtWidgets.QLabel("", dUBox)
        dULayout.addWidget(dUParamLabel, 0, 1)
        self.dUParamSlider = ParamSlider(Qt.Horizontal, self, dUParamLabel, "dU", 1000.0)
        self.dUParamSlider.setMinimum(self.canvas.grayScottModel.dUMin)
        self.dUParamSlider.setMaximum(self.canvas.grayScottModel.dUMax)
        self.dUParamSlider.setValue(self.canvas.grayScottModel.baseParams[0])
        self.dUParamSlider.updateParam(0)
        self.dUParamSlider.valueChanged.connect(self.dUParamSlider.updateParam)
        dULayout.addWidget(self.dUParamSlider, 1, 0, 1, 2)
        dUBox.setLayout(dULayout)

        dVBox = QtWidgets.QGroupBox(self.modelDock)
        dVLayout = QtWidgets.QGridLayout()
        dVLayout.addWidget(QtWidgets.QLabel("dV", dVBox), 0, 0)
        dVParamLabel = QtWidgets.QLabel("", dVBox)
        dVLayout.addWidget(dVParamLabel, 0, 1)
        self.dVParamSlider = ParamSlider(Qt.Horizontal, self, dVParamLabel, "dV", 1000.0)
        self.dVParamSlider.setMinimum(self.canvas.grayScottModel.dVMin)
        self.dVParamSlider.setMaximum(self.canvas.grayScottModel.dVMax)
        self.dVParamSlider.setValue(self.canvas.grayScottModel.baseParams[1])
        self.dVParamSlider.updateParam(0)
        self.dVParamSlider.valueChanged.connect(self.dVParamSlider.updateParam)
        dVLayout.addWidget(self.dVParamSlider, 1, 0, 1, 2)
        dVBox.setLayout(dVLayout)

        fkLayout.addWidget(fBox)
        fkLayout.addWidget(kBox)
        fkLayout.addWidget(dUBox)
        fkLayout.addWidget(dVBox)
        fkBox.setLayout(fkLayout)
        topLayout.addWidget(fkBox)

        # --------------------------------------
        controlBox = QtWidgets.QGroupBox("Controls", self.modelDock)
        controlLayout = QtWidgets.QVBoxLayout()
        self.resetButton = QtWidgets.QPushButton("Reset", self.modelDock)
        self.resetButton.clicked.connect(self.canvas.grayScottModel.initializeGrid)
        controlLayout.addWidget(self.resetButton)
        cyclesBox = QtWidgets.QGroupBox("Additional cycles/frame", self.modelDock)
        cyclesLayout = QtWidgets.QHBoxLayout()
        self.lessCycles = QtWidgets.QPushButton("-", self.modelDock)
        self.lessCycles.clicked.connect(self.canvas.grayScottModel.decreaseCycle)
        self.lessCycles.clicked.connect(self.updateCycle)
        self.cycles = QtWidgets.QLabel(self.modelDock)
        self.cycles.setText(str(2*self.canvas.grayScottModel.cycle))
        self.cycles.setAlignment(Qt.AlignCenter | Qt.AlignCenter)
        self.moreCycles = QtWidgets.QPushButton("+", self.modelDock)
        self.moreCycles.clicked.connect(self.canvas.grayScottModel.increaseCycle)
        self.moreCycles.clicked.connect(self.updateCycle)
        cyclesLayout.addWidget(self.lessCycles)
        cyclesLayout.addWidget(self.cycles)
        cyclesLayout.addWidget(self.moreCycles)
        cyclesBox.setLayout(cyclesLayout)
        controlLayout.addWidget(cyclesBox)
        controlBox.setLayout(controlLayout)
        topLayout.addWidget(controlBox)

        # --------------------------------------
        topLayout.addStretch(1)
        topBox.setLayout(topLayout)
        self.modelDock.setWidget(topBox)
        self.panelMenu.addAction(self.modelDock.toggleViewAction())

    def createPearsonPatternDetailDock(self, visible=True):
        """
        Creates dock that display a 'phase diagram' showing all the Pearson'
        patterns aswell as a short description of the current used one below it.
        WIP this should in the end highlight in the diagram the one used
        WIP the diagram could also let one be picked, or any values pair(kill - feed)
        where clicked
        WIP diagram should plot greek letter and not plain circles
        """
        self.pPDetailsDock = QtWidgets.QDockWidget('Pearson\' pattern Details', self)
        self.pPDetailsDock.setFloating(True)
        self.pPDetailsDock.setVisible(visible)
        topBox = QtWidgets.QGroupBox("", self.pPDetailsDock)
        topLayout = QtWidgets.QVBoxLayout(topBox)

        # --------------------------------------
        self.pPDetailsLabel = QtWidgets.QLabel(topBox)
        self.pPDetailsLabel.setText(self.canvas.grayScottModel.getPearsonPatternDescription())

        # --------------------------------------
        self.fkChartView = View(topBox)
        topLayout.addWidget(self.fkChartView)

        # --------------------------------------
        topLayout.addWidget(self.pPDetailsLabel)
        topBox.setLayout(topLayout)
        self.pPDetailsDock.setWidget(topBox)
        self.panelMenu.addAction(self.pPDetailsDock.toggleViewAction())

    @Slot()
    @Slot(str)
    def setSelectedPearsonsPatternDetails(self, type=None):
        """
        Updates the pattern description and highlight the selected one.
        """
        # Hide chart so its dimension does not keep those of the label
        # as they where if they should shrink
        self.hideChartAndUpdateDetails(type)
        self.fkChartView.setSelect(type)
        # Sets the dimensions of the chart folowing the label width
        self.AdjustChartSizeAndShow()

    @Slot()
    @Slot(str)
    def setHighlightedPearsonsPatternDetails(self, type=None):
        """
        Updates the pattern description and highlight the highlighted one.
        """
        # Hide chart so its dimension does not keep those of the label
        # as they where if they should shrink
        self.hideChartAndUpdateDetails(type)
        self.fkChartView.setHighlight(type)
        # Sets the dimensions of the chart folowing the label width
        self.AdjustChartSizeAndShow()

    def hideChartAndUpdateDetails(self, type=None):
        """
        Updates the pattern description part 1.
        """
        self.fkChartView.hide()
        self.pPDetailsLabel.setText(self.canvas.grayScottModel.getPearsonPatternDescription(specie=type))
        self.pPDetailsLabel.adjustSize()
        self.pPDetailsLabel.parent().adjustSize()

    def AdjustChartSizeAndShow(self):
        """
        Updates the pattern description part 2.
        """
        self.fkChartView.setMinimumHeight(self.pPDetailsLabel.size().width())
        self.fkChartView.setMaximumHeight(self.pPDetailsLabel.size().width())
        self.fkChartView.setMinimumWidth(self.pPDetailsLabel.size().width())
        self.fkChartView.setMaximumWidth(self.pPDetailsLabel.size().width())
        self.fkChartView.adjustSize()
        self.fkChartView.show()
        self.pPDetailsDock.adjustSize()

    def setFeedKillDials(self):
        """
        Set the values of model parameters slider after change in the model
        """
        self.feedParamSlider.setValue(self.canvas.grayScottModel.baseParams[2])
        self.killParamSlider.setValue(self.canvas.grayScottModel.baseParams[3])

    @Slot(int)
    def setCurrentFKInChart(self, val):
        """
        Received value is ignored as it is the technical value of the slider.
        """
        self.fkChartView.hide()
        self.fkChartView.setCurrentFKPoint(self.killParamSlider.value(), self.feedParamSlider.value())
        self.fkChartView.show()

    @Slot(int)
    def setDFeedInChart(self, val):
        """
        Received value is ignored as it is the technical value of the slider.
        """
        self.fkChartView.hide()
        self.fkChartView.setDFeed(self.dFeedParamSlider.value())
        self.fkChartView.show()

    @Slot(int)
    def setDKillInChart(self, val):
        """
        Received value is ignored as it is the technical value of the slider.
        """
        self.fkChartView.hide()
        self.fkChartView.setDKill(self.dKillParamSlider.value())
        self.fkChartView.show()

    def updateCycle(self):
        """
        Update label for number of supplementary render cycles per frame
        """
        self.cycles.setText(str(2*self.canvas.grayScottModel.cycle))

    def show_fps(self, fps):
        """
        Shows FPS in status bar.
        """
        msg = " FPS - %0.2f" % float(fps)
        # NOTE: We can't use showMessage in PyQt5 because it causes
        #       a draw event loop (show_fps for every drawing event,
        #       showMessage causes a drawing event, and so on).
        self.status_label.setText(msg)

        self.canvas.visible = True


class LightTypeGroupBox(QtWidgets.QGroupBox):
    """
    Simple GroupBox that keep a reference of the parameter it concerns.
    Its toggling toggles too the proper lighting parameter in the MainRenderer
    of the model.
    """
    def __init__(self, title, parent, param):
        super(LightTypeGroupBox, self).__init__(title, parent)
        self.param = param
        self.parent = parent
        self.toggled.connect(self.updateLighting)

    @Slot(bool)
    def updateLighting(self, state):
        self.parent.canvas.mainRenderer.setLighting(self.title(), self.param, state)


class ParamSlider(QtWidgets.QSlider):
    """
    Simple slider that handles floating values and has a slot to update the
    corresponding parameter in the canvas grayScottModel.
    """
    def __init__(self, orientation, parent, outputLabel, param, resolution):
        super(ParamSlider, self).__init__(orientation, parent)
        super(ParamSlider, self).setSingleStep(1)
        self.param = param
        self.parent = parent
        self.resolution = resolution
        self.outputLabel = outputLabel
        self.outputLabel.setAlignment(Qt.AlignRight | Qt.AlignCenter)
        self.outputFormat = "%1.3f"

    def setMinimum(self, val):
        self.vMin = val
        super(ParamSlider, self).setMinimum(0)

    def setMaximum(self, val):
        self.vMax = val
        if self.vMax < 1e-2:
            self.outputFormat = "%1.4f"
        super(ParamSlider, self).setMaximum(self.resolution)

    def setValue(self, val):
        self.outputLabel.setText(self.outputFormat % val)
        value = int(self.resolution * (val - self.vMin)/(self.vMax - self.vMin))
        super(ParamSlider, self).setValue(value)

    def value(self):
        value = super(ParamSlider, self).value()
        return ((float(value) / self.resolution) * (self.vMax - self.vMin)) + self.vMin

    @Slot(int)
    def updateParam(self, val):
        value = self.value()
        self.outputLabel.setText(self.outputFormat % (value))
        if self.param == "feed":
            self.parent.canvas.grayScottModel.setParams(feed=value)
        elif self.param == "dFeed":
            self.parent.canvas.grayScottModel.setParams(dFeed=value)
        elif self.param == "kill":
            self.parent.canvas.grayScottModel.setParams(kill=value)
        elif self.param == "dKill":
            self.parent.canvas.grayScottModel.setParams(dKill=value)
        elif self.param == "dU":
            self.parent.canvas.grayScottModel.setParams(dU=value)
        elif self.param == "dV":
            self.parent.canvas.grayScottModel.setParams(dV=value)


class LightParamSlider(QtWidgets.QSlider):
    def __init__(self, orientation, parent, outputLabel, lightType, param):
        super(LightParamSlider, self).__init__(orientation, parent)
        super(LightParamSlider, self).setSingleStep(1)
        self.lightType = lightType
        self.param = param
        self.parent = parent
        self.outputLabel = outputLabel
        self.outputLabel.setAlignment(Qt.AlignRight | Qt.AlignCenter)
        self.outputFormat = "%3.2f"

    def setMinimum(self, val):
        self.vMin = val
        super(LightParamSlider, self).setMinimum(0)

    def setMaximum(self, val):
        self.vMax = val
        if self.vMax < 1.0:
            self.outputFormat = "%1.5f"
        elif self.vMax < 10.0:
            self.outputFormat = "%1.2f"
        else:
            self.outputFormat = "%3.0f"
        super(LightParamSlider, self).setMaximum(1000)

    def setValue(self, val):
        self.outputLabel.setText(self.outputFormat % val)
        value = (val - self.vMin)/(self.vMax - self.vMin)
        if (self.vMax + 1)/(self.vMin + 1) > 100:
            value = math.sqrt(math.sqrt(value))
        super(LightParamSlider, self).setValue(int(1000 * value))

    def value(self):
        val = super(LightParamSlider, self).value()
        value = float(val)/1000.0
        if (self.vMax + 1)/(self.vMin + 1) > 100:
            value = value**4
        return (value * (self.vMax - self.vMin)) + self.vMin

    @Slot(int)
    def updateLighting(self, value):
        value = self.value()
        self.outputLabel.setText(self.outputFormat % value)
        self.parent.canvas.mainRenderer.setLighting(self.lightType, self.param, value)


class LightParamDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    def __init__(self, parent, lightType, param):
        super(LightParamDoubleSpinBox, self).__init__(parent)
        self.lightType = lightType
        self.param = param
        self.parent = parent

    @Slot(float)
    def updateLighting(self, value):
        self.parent.canvas.mainRenderer.setLighting(self.lightType, self.param, value)


class LightParamSpinBox(QtWidgets.QSpinBox):
    def __init__(self, parent, lightType, param):
        super(LightParamSpinBox, self).__init__(parent)
        self.lightType = lightType
        self.param = param
        self.parent = parent

    @Slot(int)
    def updateLighting(self, value):
        self.parent.canvas.mainRenderer.setLighting(self.lightType, self.param, value)


class RoundedButton(QtWidgets.QPushButton):
    """
    Button that represents a color and opens a QColorDialog on its click.
    Updates the color of the canvas/mainRenderer according to lightType.
    """
    def __init__(self, text, parent, bordersize, outlineColor, fillColor):
        super(RoundedButton, self).__init__()
        self.bordersize = bordersize
        self.outlineColor = outlineColor
        self.fillColor = fillColor
        self.lightType = text
        self.parent = parent
        self.colorDialog = QtWidgets.QColorDialog(fillColor, self)
        self.clicked.connect(self.changeColor)

    def changeColor(self):
        color = self.colorDialog.getColor(self.fillColor)
        if color.isValid():
            self.fillColor = color
            if self.lightType == 'background':
                self.parent.canvas.setBackgroundColor(color.getRgbF())
            else:
                self.parent.canvas.mainRenderer.setLighting(self.lightType, 'color', color.getRgbF())

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


class FkPoint(QPointF):
    """
    WIP a QpointF that knows its name and symbol to used as marker
    """
    def __init__(self, xpos, ypos, name, symbol):
        super(FkPoint, self).__init__(xpos, ypos)
        self.name = name
        self.symbol = symbol


class View(QChartView):
    """
    A QChartView subclass to splot a scatter chart with individual labels that
    one can be highlighted. This also plot a rectangle showing the ranges of
    kill feed currently modelled.
    # TODO, have a clicked/selected symbol to switch the pattern used,
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.selectedColor = QColor(255, 0, 0)
        self.currentColor = QColor(221, 158, 116)
        self.highlightedColor = QColor(247, 102, 62)

        # Chart
        self.fkChart = QChart()
        self.fkChart.setBackgroundVisible(False)
        self.fkChart.legend().hide()

        # Points (drawn almost invisible, but cannot be hidden or no hover/click
        # would be possible in the chart)
        self.fkPoints = QScatterSeries()
        self.fkPoints.setMarkerSize(12)
        self.fkPoints.setColor(QColor(255, 255, 255, 1))
        self.fkPoints.setBorderColor(QColor(255, 255, 255, 0))
        self.fkPointList = []
        self.fkPointValues = []
        self.fkPointLabels = []
        for specie in GrayScottModel.species.keys():
            feed = GrayScottModel.species[specie][2]
            kill = GrayScottModel.species[specie][3]
            symbol = GrayScottModel.species[specie][5]
            fkPoint = FkPoint(kill, feed, specie, symbol)
            self.fkPoints.append(fkPoint)
            self.fkPointList.append(fkPoint)
            self.fkPointValues.append((kill, feed, symbol, specie))
        self.fkChart.addSeries(self.fkPoints)
        self.fkChart.createDefaultAxes()

        # Axes
        axisX = self.fkChart.axes(orientation=Qt.Horizontal)[0]
        axisX.setTickInterval(0.01)
        axisX.setTickCount(6)
        axisX.setRange(0.03,0.08)
        axisX.setTitleText("Kill")
        axisY = self.fkChart.axes(orientation=Qt.Vertical)[0]
        axisY.setTickInterval(0.02)
        axisY.setTickCount(7)
        axisY.setRange(0.0,0.12)
        axisY.setTitleText("Feed")
        p = self.fkChart.sizePolicy()
        p.setHeightForWidth(True)
        self.fkChart.setSizePolicy(p)
        self.fkChart.setAcceptHoverEvents(True)
        self.setRenderHint(QPainter.Antialiasing)
        self.scene().addItem(self.fkChart)

        self.positionCorrectionPoint = QPointF(4.045275,-4.11746)
        self.highlightedSpecie = None
        self.selectedSpecie = None
        self.currentFKPoint = None
        self.dFeed = 0
        self.dKill = 0

        # self.fkPoints.clicked.connect(self.clickPoint)
        self.fkPoints.hovered.connect(self.hoverPoint)

        self.setMouseTracking(True)

    def resizeEvent(self, event):
        QChartView.resizeEvent(self, event)
        if self.scene():
            self.scene().setSceneRect(QRectF(QPointF(0, 0), event.size()))
            self.fkChart.resize(event.size())

    def paintEvent(self, event):
        QChartView.paintEvent(self, event)
        self.drawCustomLabels(self.fkPoints, 14)
        self.drawCurrentFKPoint(self.fkPoints, 14)
        self.drawDFeedDKillBox(self.fkPoints)

    def drawCustomLabels(self, points, pointSize):
        """
        Writes each symbol at position of each points. These are intended as the
        true representations of the points, not labels of them, hence points are
        drawn almost transparent. Selected and highlighted points are written with
        different colors.
        """
        if points.count() == 0:
            return

        painter = QPainter(self.viewport())
        # font size
        fm = painter.font()
        fm.setPointSize(pointSize)
        painter.setFont(fm)
        # to be restored after highlighted draw
        currentPen = painter.pen()
        currentBrush = painter.brush()
        for i in range(points.count()):
            pointLabel = self.fkPointValues[i][2]
            specie = self.fkPointValues[i][3]
            position = points.at(i)
            if self.selectedSpecie == specie:
                pen = QPen(self.selectedColor)
                brush = QBrush(self.selectedColor)
                painter.setPen(pen)
                painter.setBrush(brush)
            elif self.highlightedSpecie == specie:
                pen = QPen(self.highlightedColor)
                brush = QBrush(self.highlightedColor)
                painter.setPen(pen)
                painter.setBrush(brush)
            painter.drawText(self.fkChart.mapToPosition(position, points)-self.positionCorrectionPoint, pointLabel)
            painter.setPen(currentPen)
            painter.setBrush(currentBrush)

    def drawCurrentFKPoint(self, points, pointSize):
        """
        Draws a cross indicating the current Feed and Kill values modelled.
        This cross is not visible if the values are ones of a selected
        specie, which has already a symbol shown.
        """
        if self.currentFKPoint is None:
            return
        currentKill = self.currentFKPoint.x()
        currentFeed = self.currentFKPoint.y()
        selectedPoint = self.getPointOfSpecie(self.selectedSpecie)
        # There should always be a selected specie, but let's stay on the safe side
        # if there is one, and if current and selected are too close, return
        if selectedPoint is not None:
            selectedKill = selectedPoint.x()
            selectedFeed = selectedPoint.y()
            dKill = abs(currentKill - selectedKill)
            dFeed = abs(currentFeed - selectedFeed)
            if dKill < 1e-04 and dFeed < 1e-04:
                return

        painter = QPainter(self.viewport())
        # font size
        fm = painter.font()
        fm.setPointSize(pointSize)
        painter.setFont(fm)
        # to be restored after highlighted draw
        currentPen = painter.pen()
        currentBrush = painter.brush()
        pen = QPen(self.currentColor)
        brush = QBrush(self.currentColor)
        painter.setPen(pen)
        painter.setBrush(brush)
        painter.drawText(self.fkChart.mapToPosition(self.currentFKPoint, points)-self.positionCorrectionPoint, "+")
        painter.setPen(currentPen)
        painter.setBrush(currentBrush)

    def drawDFeedDKillBox(self, points):
        """
        Draws a box showing the ranges of feed and kill currently modelled.
        This is either plotted around the selected specie, or around the current
        values of feed and kill. There should always be a selected specie, and
        this one is to be replaced if a current values are defined, but if
        something goes wrong, nothing is drawn.
        """
        if self.dKill == 0 and self.dFeed == 0:
            return
        centerPoint = self.getPointOfSpecie(self.selectedSpecie)
        if self.currentFKPoint is not None:
            centerPoint = self.currentFKPoint
        if centerPoint is None:
            return
        halves = QPointF(self.dKill, -self.dFeed)
        topLeft = centerPoint - halves
        topLeft = self.fkChart.mapToPosition(topLeft, points)
        bottomRight = centerPoint + halves
        bottomRight = self.fkChart.mapToPosition(bottomRight, points)
        painter = QPainter(self.viewport())
        # to be restored after highlighted draw
        currentPen = painter.pen()
        currentBrush = painter.brush()
        pen = QPen(self.currentColor)
        brush = QBrush(QColor(255, 255, 255, 0))
        painter.setPen(pen)
        painter.setBrush(brush)
        painter.drawRect(QRectF(topLeft, bottomRight))
        painter.setPen(currentPen)
        painter.setBrush(currentBrush)

    def setSelect(self, specie):
        """
        Sets the selected specie. If one is selected, it is the modelled one
        and no current point should be displayed (as a cross)
        """
        self.selectedSpecie = specie
        self.currentFKPoint = None

    def setHighlight(self, specie):
        """
        Sets the specie to highlight.
        """
        self.highlightedSpecie = specie

    # Currently chart clicking produces a segmentationfault...
    # def clickPoint(self, point):
    #     specie = self.getSpecieOfPoint(point)
    #     self.parent().parent().parent().pearsonsPatternsComboBox.setCurrentText(specie)
    #     self.parent().parent().parent().pearsonsPatternsComboBox.textActivated(specie)

    def setCurrentFKPoint(self, kill, feed):
        """
        Sets the point to draw a cross on to show the current feed and kill modelled.
        """
        self.currentFKPoint = QPointF(kill, feed)

    def setDKill(self, value):
        """
        Sets range of kill currently modelled. Will be the width of the box.
        """
        self.dKill = value

    def setDFeed(self, value):
        """
        Sets range of feed currently modelled. Will be the height of the box.
        """
        self.dFeed = value

    def hoverPoint(self, point, state):
        """
        Hovering on a point will change the details shown and will highlight the
        """
        specie = self.getSpecieOfPoint(point)
        if state and specie is not None:
            # That's so ugly, surely there is a better way to find the method...
            self.parent().parent().parent().setHighlightedPearsonsPatternDetails(specie)

    def getSpecieOfPoint(self, point):
        pointIndex = None
        if point in self.fkPointList:
            pointIndex = self.fkPointList.index(point)
            return self.fkPointValues[pointIndex][3]
        return None

    def getPointOfSpecie(self, specie):
        for i in range(len(self.fkPointValues)):
            if self.fkPointValues[i][3] == specie:
                return self.fkPoints.at(i)
        return None


################################################################################


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
