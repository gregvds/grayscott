# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------
# Author: Gregoire Vandenschrick
# Date:   30/12/2021
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
    For more use help, type python3 gs3D.py -h.
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

import sys

from math import pi
import argparse
import textwrap

import numpy as np

from vispy import app, gloo
from vispy.gloo import gl

from gs3D_lib import (Camera, GrayScottModel, ShadowRenderer, MainRenderer)

# ? Use of this ?
# gl.use_gl('gl+')
# app.use_app('pyside6')

# Provide automatic signal function selection for PyQt5/PySide2
pyqtsignal = QtCore.pyqtSignal if hasattr(QtCore, 'pyqtSignal') else QtCore.Signal

################################################################################
class Canvas(app.Canvas):

    def __init__(self,
                 size=(1024, 1024),
                 modelSize=(512,512),
                 specie='alpha_left',
                 cmap='honolulu_r'):
        app.Canvas.__init__(self,
                            size=size,
                            title='3D Gray-Scott Reaction-Diffusion: - GregVDS',
                            keys='interactive')

        # Create the Gray-Scott model
        # --------------------------------------
        # this contains the 3D grid model, texture and program
        self.grayScottModel = GrayScottModel(canvas=self,
                                             gridSize=modelSize,
                                             specie=specie)

        # Build model
        # --------------------------------------
        model = np.eye(4, dtype=np.float32)

        # Build light camera for shadow view and projection
        # --------------------------------------
        self.lightCamera = Camera(model,
                             eye=[0, 2, 2],
                             target=[0,0,0],
                             up=[0,1,0],
                             shadowCam=True)

        # Build shadow renderer using the model, grayScottModel and lightCamera
        # --------------------------------------
        self.shadowRenderer = ShadowRenderer(self.grayScottModel,
                                             self.lightCamera,
                                             shadowMapSize=2048)

        # Build main camera for view and projection
        # --------------------------------------
        self.camera = Camera(model,
                             eye=[0, 2, -2],
                             target=[0,0,0],
                             up=[0,1,0],
                             fov=24.0,
                             aspect=self.size[0] / float(self.size[1]),
                             near=0.1,
                             far=20.0)

        # Build main renderer using the model, grayScottModel, shadowRenderer and camera
        # --------------------------------------
        self.mainRenderer = MainRenderer(self.grayScottModel,
                                         self.camera,
                                         self.shadowRenderer,
                                         cmap=cmap)

        # Mouse interactions parameters
        # --------------------------------------
        self.pressed = False

        self.displaySwitch = 0

        # ? better computation ?
        # --------------------------------------
        gl.GL_FRAGMENT_PRECISION_HIGH = 1

        # OpenGL initialization
        # --------------------------------------
        gloo.set_state(clear_color=(0.30, 0.30, 0.35, 1.00), depth_test=True,
                       line_width=0.75)

        self.activate_zoom()
        # self.show()

    ############################################################################
    #

    def on_draw(self, event):
        if self.visible:
            # Render next model state into buffer
            with self.grayScottModel.buffer:
                gloo.set_viewport(0, 0, self.grayScottModel.h, self.grayScottModel.w)
                gloo.set_state(depth_test=False,
                               clear_color='black',
                               polygon_offset=(0, 0))
                self.grayScottModel.draw()

            # Render the shadowmap into buffer
            if self.mainRenderer.shadow > 0:
                with self.mainRenderer.buffer:
                    gloo.set_viewport(0, 0, self.shadowRenderer.shadowMapSize, self.shadowRenderer.shadowMapSize)
                    gloo.set_state(depth_test=True,
                                   polygon_offset=(1, 1),
                                   polygon_offset_fill=True)
                    gloo.clear(color=True, depth=True)
                    self.mainRenderer.shadowRenderer.draw()

            # DEBUG
            if self.displaySwitch == 0:
                gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
                gloo.set_state(blend=False, depth_test=True,
                               clear_color=(0.30, 0.30, 0.35, 1.00),
                               blend_func=('src_alpha', 'one_minus_src_alpha'),
                               polygon_offset=(1, 1),
                               polygon_offset_fill=True)
                gloo.clear(color=True, depth=True)
                self.mainRenderer.draw()
            # To check the view from the lightCamera for shadowmap rendering
            else:
                gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
                gloo.set_state(depth_test=True,
                               polygon_offset=(1, 1),
                               polygon_offset_fill=True)
                gloo.clear(color=True, depth=True)
                self.mainRenderer.shadowRenderer.draw()

            # exchange between rg and ba sets in texture
            self.grayScottModel.flipPingpong()
            self.update()

    def on_resize(self, event):
        self.activate_zoom()

    def activate_zoom(self):
        gloo.set_viewport(0, 0, *self.physical_size)
        self.mainRenderer.updateAspect(self.size[0] / float(self.size[1]))

    ############################################################################
    # Mouse and keys interactions

    def on_mouse_wheel(self, event):
        # no Shift modifier key: moves the camera
        self.mainRenderer.moveCamera(dDistance=(event.delta[1])/3.0)
        # Shift modifier key: zoom in out
        self.mainRenderer.zoomCamera((event.delta[0])/3.0)

    def on_mouse_press(self, event):
        self.pressed = True
        self.mousePos = event.pos
        if len(event.modifiers) == 1 and event.modifiers[0] == 'Shift':
            (x, y) = event.pos
            (sx, sy) = self.size
            xpos = x/sx
            ypos = 1 - y/sy
            self.grayScottModel.interact([xpos, ypos], event.button)

    def on_mouse_release(self, event):
        self.pressed = False
        self.grayScottModel.interact([0, 0], 0)

    def on_mouse_move(self, event):
        if(self.pressed):
            if len(event.modifiers) == 0:
                # no Shift modifier key: moves the camera
                dazimuth = (event.pos[0] - self.mousePos[0]) * (2*pi) / self.size[0]
                delevation = (event.pos[1] - self.mousePos[1]) * (2*pi) / self.size[1]
                self.mousePos = event.pos
                self.mainRenderer.moveCamera(dAzimuth=dazimuth, dElevation=delevation)
            elif len(event.modifiers) == 1 and event.modifiers[0] == 'Shift':
                # Shift modifier: interact with V concentrations
                (x, y) = event.pos
                (sx, sy) = self.size
                xpos = x/sx
                ypos = 1 - y/sy
                self.grayScottModel.interact([xpos, ypos], event.button)
            elif len(event.modifiers) == 1 and event.modifiers[0] == 'Control':
                # Control Modifier: moves the light
                dazimuth = (event.pos[0] - self.mousePos[0]) * (2*pi) / self.size[0]
                delevation = (event.pos[1] - self.mousePos[1]) * (2*pi) / self.size[1]
                self.mousePos = event.pos
                self.mainRenderer.moveLight(dAzimuth=dazimuth, dElevation=delevation)

    NO_ACTION = (None, None)

    def on_key_press(self, event):
        """treats all key event that are defined in keyactionDictionnary"""
        eventKey = ''
        eventModifiers = []
        if len(event.text) > 0:
            eventKey = event.text
        else:
            eventKey = event.key.name
        for each in event.modifiers:
            eventModifiers.append(each)
        eventModifiers = tuple(eventModifiers)
        (func, args) = Canvas.keyactionDictionnary.get((eventKey, eventModifiers), self.NO_ACTION)
        if func is not None:
            if hasattr(self, func.__name__):
                func(self, *args)
            elif hasattr(self.grayScottModel, func.__name__):
                func(self.grayScottModel, *args)
            elif hasattr(self.mainRenderer, func.__name__):
                func(self.mainRenderer, *args)
            # elif hasattr(self.shadowRenderer, func.__name__):
            #     func(self.shadowRenderer, event, *args)
            else:
                print("Method %s does not seem to be found..." % str(func))

    ############################################################################
    # Debug/utilities functions

    def switchDisplay(self):
        """
        Toggles between mainRenderer display and shadowRenderer display.
        """
        self.displaySwitch = (self.displaySwitch + 1) % 2

    def measure_fps2(self, window=1, callback=' %1.1f FPS                   '):
        """Measure the current FPS

        Sets the update window, connects the draw event to update_fps
        and sets the callback function.

        Parameters
        ----------
        window : float
            The time-window (in seconds) to calculate FPS. Default 1.0.
        callback : function | str
            The function to call with the float FPS value, or the string
            to be formatted with the fps value and then printed. The
            default is ``'%1.1f FPS'``. If callback evaluates to False, the
            FPS measurement is stopped.
        """
        # Connect update_fps function to draw
        self.events.draw.disconnect(self._update_fps)
        if callback:
            if isinstance(callback, str):
                callback_str = callback  # because callback gets overwritten

                def callback(x):
                    print(callback_str % x, end="\r")

            self._fps_window = window
            self.events.draw.connect(self._update_fps)
            self._fps_callback = callback
        else:
            self._fps_callback = None

    # Dictionnary to map key commands to function
    # --------------------------------------
    keyactionDictionnary = {
        ('=', ()): (switchDisplay, ()),
        (' ', ()): (GrayScottModel.initializeGrid, ()),
        ('/', ('Shift',)): (MainRenderer.switchReagent, ()),
        ('P', ('Control',)): (GrayScottModel.increaseCycle, ()),
        ('O', ('Control',)): (GrayScottModel.decreaseCycle, ()),
        ('@', ()): (MainRenderer.resetCamera, ()),
        ('<', ()): (MainRenderer.resetLight, ()),
        (',', ('Control',)): (MainRenderer.modifyLightCharacteristic, ('ambient',)),
        (';', ('Control',)): (MainRenderer.modifyLightCharacteristic, ('diffuse',)),
        (':', ('Control',)): (MainRenderer.modifyLightCharacteristic, ('specular',)),
        ('=', ('Control',)): (MainRenderer.modifyLightCharacteristic, ('shadow',)),
        ('N', ('Control',)): (MainRenderer.modifyLightCharacteristic, ('attenuation',)),
        ('L', ('Control',)): (MainRenderer.modifyLightCharacteristic, ('shininess', '-')),
        ('M', ('Control',)): (MainRenderer.modifyLightCharacteristic, ('shininess', '+')),
        ('J', ('Control',)): (MainRenderer.modifyLightCharacteristic, ('u_fresnel_power', '-')),
        ('K', ('Control',)): (MainRenderer.modifyLightCharacteristic, ('u_fresnel_power', '+')),
        ('I', ('Control',)): (MainRenderer.modifyLightCharacteristic, ('lightbox',))
    }
    for key in MainRenderer.colormapDictionnary.keys():
        keyactionDictionnary[(key, ())] = (MainRenderer.setColorMap, (MainRenderer.colormapDictionnary[key],))
    for key in MainRenderer.colormapDictionnaryShifted.keys():
        keyactionDictionnary[(key, ('Shift',))] = (MainRenderer.setColorMap, (MainRenderer.colormapDictionnaryShifted[key],))
    for key in GrayScottModel.speciesDictionnary.keys():
        keyactionDictionnary[(key, ())] = (GrayScottModel.setSpecie, (GrayScottModel.speciesDictionnary[key],))
    for key in GrayScottModel.speciesDictionnaryShifted.keys():
        keyactionDictionnary[(key, ('Shift',))] = (GrayScottModel.setSpecie, (GrayScottModel.speciesDictionnaryShifted[key],))

    ############################################################################
    # Output functions

    @staticmethod
    def getCommandsDocs():
        """
        Calss static method to harvest all __doc__
        from methods bound to keys, modifiers.
        Result to be used as description by the parser help.
        """
        commandDoc = ''
        command = ''
        for (key, modifiers) in Canvas.keyactionDictionnary.keys():
            (function, args) = Canvas.keyactionDictionnary[(key, modifiers)]
            # command = "keys '%s' + '%s':" % (modifiers, key)
            command = "Keys "
            for modifier in modifiers:
                command += "'%s' + " % modifier
            command += "'%s': %s(" % (key, function.__name__)
            if len(args) > 0:
                for arg in args:
                    if arg == args[-1]:
                        command += "'%s'" % str(arg)
                    else:
                        command += "'%s', " % str(arg)
            command += ")"
            if function.__doc__ is not None:
                command += textwrap.dedent(function.__doc__)
            command += "\n"
            commandDoc += command
        return commandDoc


################################################################################
class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)

        self.resize(1024, 1024)
        self.setWindowTitle('... TEST Qt + gs3D ...')

        self.canvas = Canvas()
        self.canvas.create_native()
        self.canvas.native.setParent(self)
        print(self.canvas.native.parent)
        self.canvas.measure_fps(0.1, self.show_fps)

        splitter1 = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter1.addWidget(self.canvas.native)

        self.setCentralWidget(splitter1)

        self.createColorsDock()
        self.createPearsonsPatternsDock()
        self.createLightingDock()

        self.initializeGui()
        self.connectSignals()

        # FPS message in statusbar:
        self.status = self.statusBar()
        self.status_label = QtWidgets.QLabel('...')
        self.status.addWidget(self.status_label)

    def createLightingDock(self):
        self.lightingDock = QtWidgets.QDockWidget('Lighting', self)

        lightingSwitches = ('ambient', 'diffuse', 'specular', 'lightBox')
        rL = []
        self.lightingSwitches = []
        groupBox = QtWidgets.QGroupBox()
        vbox = QtWidgets.QVBoxLayout()
        for name in lightingSwitches:
            lightSwith = QtWidgets.QCheckBox(name)
            self.lightingSwitches.append(lightSwith)
            vbox.addWidget(lightSwith)
        vbox.addStretch(1)
        groupBox.setLayout(vbox)

        self.lightingDock.setWidget(groupBox)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.lightingDock)

        # TODO add Doc to menu for on/off visibility...
        # LEBONMENU.addAction(self.lightingDock.toggleViewAction())

    def createColorsDock(self):
        self.colorDock = QtWidgets.QDockWidget('Colors', self)
        self.colorsComboBox = QtWidgets.QComboBox(self.colorDock)

        colors = []
        for key in MainRenderer.colormapDictionnary.keys():
            colors.append(MainRenderer.colormapDictionnary[key])
        for key in MainRenderer.colormapDictionnaryShifted.keys():
            colors.append(MainRenderer.colormapDictionnaryShifted[key])
        colors.sort()
        self.colorsComboBox.addItems(colors)

        self.colorDock.setWidget(self.colorsComboBox)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.colorDock)

        # TODO add Doc to menu for on/off visibility...
        # LEBONMENU.addAction(self.colorDock.toggleViewAction())

    def createPearsonsPatternsDock(self):
        self.pearsonsPatternsDock = QtWidgets.QDockWidget('Pearson\'s patterns', self)
        self.pearsonsPatternsComboBox = QtWidgets.QComboBox(self.pearsonsPatternsDock)

        patterns = []
        for key in GrayScottModel.speciesDictionnary.keys():
            patterns.append(GrayScottModel.speciesDictionnary[key])
        for key in GrayScottModel.speciesDictionnaryShifted.keys():
            patterns.append(GrayScottModel.speciesDictionnaryShifted[key])
        patterns.sort()
        self.pearsonsPatternsComboBox.addItems(patterns)

        self.pearsonsPatternsDock.setWidget(self.pearsonsPatternsComboBox)
        self.addDockWidget(QtCore.Qt.TopDockWidgetArea, self.pearsonsPatternsDock)

        # TODO add Doc to menu for on/off visibility...
        # LEBONMENU.addAction(self.colorDock.toggleViewAction())

    def initializeGui(self):
        self.colorsComboBox.setCurrentText(self.canvas.mainRenderer.cmapName)
        self.pearsonsPatternsComboBox.setCurrentText(self.canvas.grayScottModel.specie)

    def connectSignals(self):
        self.colorsComboBox.textActivated[str].connect(self.canvas.mainRenderer.setColorMap)
        self.colorsComboBox.textActivated[str].emit(self.colorsComboBox.currentText())

        self.pearsonsPatternsComboBox.textActivated[str].connect(self.canvas.grayScottModel.setSpecie)
        self.pearsonsPatternsComboBox.textActivated[str].emit(self.pearsonsPatternsComboBox.currentText())

    def show_fps(self, fps):
        msg = "FPS - %0.2f" % float(fps)
        # NOTE: We can't use showMessage in PyQt5 because it causes
        #       a draw event loop (show_fps for every drawing event,
        #       showMessage causes a drawing event, and so on).
        self.status_label.setText(msg)

        self.canvas.visible = True


################################################################################
# def fun(x):
#     c.title = c.title2 +' - FPS: %0.1f' % x


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=textwrap.dedent(Canvas.getCommandsDocs()),
                                     epilog= textwrap.dedent("""Examples:
    python3 gs3D.py
    python3 gs3D.py -c osmort
    python3 gs3D.py -s 512 -p kappa_left -c oslo
    python3 gs3D.py -s 512 -w 800 -p alpha_left -c detroit"""),
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

    args = parser.parse_args()

    appQt = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    appQt.exec()



    # c = Canvas(modelSize=(args.Size, args.Size),
    #            size=(args.Window, args.Window),
    #            specie=args.Pattern,
    #            cmap=args.Colormap)
    # c.measure_fps2(callback=fun)
    # app.run()
