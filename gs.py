# -*- coding: utf-8 -*-
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
# -----------------------------------------------------------------------------
"""
    Gray-Scott reaction-diffusion model
    -----------------------------------

    Pearson's pattern can be switched with keys:
            'a': 'alpha_left',
            'A': 'alpha_right',
            'b': 'beta_left',
            'B': 'beta_right',
            'd': 'delta_left',
            'D': 'delta_right',
            'e': 'epsilon_left',
            'E': 'epsilon_right',
            'g': 'gamma_left',
            'G': 'gamma_right',
            'h': 'eta',
            'i': 'iota',
            'k': 'kappa_left',
            'K': 'kappa_right',
            'l': 'lambda_left',
            'L': 'lambda_right',
            'm': 'mu_left',
            'M': 'mu_right',
            'n': 'nu_left',
            'p': 'pi_left',
            't': 'theta_left',
            'T': 'theta_right',
            'x': 'xi_left',
            'z': 'zeta_left',
            'Z': 'zeta_right'
    Several colormaps are available via keys 1 - 0, shifted for reversed version.
    Mouse left click in the grid refills reagent v at 0.5.
    Mouse right click in the grid put reagent v at 0.
    Ctrl + Mouse click and drag to modify globaly feed and kill rates.
    Alt + Mouse click and drag to vary sinusoidally feed and kill rates.
    Shift + Mouse click and drag modifies the hillshading parameters.
    key = reset the spatial variation of feed and kill to their Pattern values
    key + toggle showing guidelines when modifying/modulatin feed and kill rates.
    key / switch presentation between u and v.
    key * toggles hillshading on or off.
    key $ toggles interpolation on or off.
    key Up/Down arrows to multiply computation cycles/frame
    key Left/right decreases/increases globally dD of the model
    key Shift + Left/right decreases/increases gaussian variation of dD
    Spacebar reseeds the grid.
"""

import math
import argparse
import textwrap

import numpy as np

from vispy import app, gloo
from vispy.gloo import gl

from shaders import (vertex_shader, compute_fragment, render_hs_fragment,
                     lines_vertex, lines_fragment)

from systems import (pearson, PearsonGrid, PearsonCurve, PearsonCurve2)

from gs_lib import (get_colormap, createColormaps,
                    plot_color_gradients, import_pearsons_types, setup_grid)

################################################################################


class Canvas(app.Canvas):
    # Gray-Scott model related variables
    (fMin, fMax)            = (0.0, 0.08)
    (kMin, kMax)            = (0.03, 0.07)
    (ddMin, ddMax)          = (0.2, 1.3)
    fModAmount, kModAmount  = 0, 0
    ddMod                   = 0.0

    # Pearson's patterns related variables
    species = {}
    speciesDictionnary = {
        'a': 'alpha_left',
        'A': 'alpha_right',
        'b': 'beta_left',
        'B': 'beta_right',
        'd': 'delta_left',
        'D': 'delta_right',
        'e': 'epsilon_left',
        'E': 'epsilon_right',
        'g': 'gamma_left',
        'G': 'gamma_right',
        'h': 'eta',
        'i': 'iota',
        'k': 'kappa_left',
        'K': 'kappa_right',
        'l': 'lambda_left',
        'L': 'lambda_right',
        'm': 'mu_left',
        'M': 'mu_right',
        'n': 'nu_left',
        'p': 'pi_left',
        't': 'theta_left',
        'T': 'theta_right',
        'x': 'xi_left',
        'z': 'zeta_left',
        'Z': 'zeta_right'
    }

    # Colormaps related variables
    colormapDictionnary = {
        '1': 'boston_r',
        '&': 'boston',
        '2': 'malmo',
        'é': 'malmo_r',
        '3': 'uppsala',
        '"': 'uppsala_r',
        '4': 'oslo_r',
        '\'': 'oslo',
        '5': 'Lochinver',
        '(': 'Lochinver_r',
        '6': 'rejkjavik',
        '§': 'rejkjavik_r',
        '7': 'detroit',
        'è': 'antidetroit',
        '8': 'tromso',
        '!': 'osmort',
        '9': 'irkoutsk',
        'ç': 'irkoutsk_r',
        '0': 'krasnoiarsk',
        'à': 'krasnoiarsk_r'
    }
    createColormaps()
    cmapName = ''

    # brush to add reagent v
    brush                   = np.zeros((1, 1, 2), dtype=np.float32)
    brushType               = 0

    mouseDown               = False
    mousePressControlPos    = [0.0, 0.0]
    mousePressAltPos        = [0.0, 0.0]

    # Hillshading lightning parameters
    hsdirection, hsaltitude = 0, 0
    hsdir, hsalt            = 0, 0

    # interpolation
    interpolationMode       = 'linear'

    # put all instance attribute names here
    __slots__ = ('w','h','cwidth','cheight','dt','dd','specie','compute','render',
                 'pearsonsGrid','pearsonsCurve','pearsonsCurve2','fkLines','ddLines','hsLines',
                 'hsz','lastHsz','guidelines','cycle','P','P2','UV','params','params2','texture',
                 'pingpong','framebuffer','keyactionDictionnary')

    def __init__(self,
                 size=(1024, 1024),
                 scale=0.5,
                 specie='lambda_left',
                 cmap='irkoutsk',
                 hsz=2.0,
                 dt=1.0,
                 dd=1.0,
                 guidelines=False):
        super().__init__(size=size,
                         title='Gray-Scott Reaction-Diffusion: - GregVDS',
                         keys='interactive',
                         resizable=False,
                         dpi=221)

        self.cwidth, self.cheight   = int(size[0]*scale), int(size[1]*scale)
        self.dt                     = dt
        self.dd                     = dd
        self.specie                 = specie
        self.cmapName               = cmap
        self.hsz                    = hsz
        self.lastHsz                = 2.0
        self.guidelines             = guidelines
        # cycle of computation per frame
        self.cycle                  = 0
        # pipeline toggling parameter
        self.pingpong               = 1

        # ? better computation for diffusion and concentration ?
        gl.GL_FRAGMENT_PRECISION_HIGH = 1

        # Dictionnary to map key commands to function
        # All these functions will receive the calling event.
        self.keyactionDictionnary = {
            ' '    : self.initializeGrid,
            '/'    : self.switchReagent,
            '$'    : self.toggleHillshading,
            '*'    : self.toggleInterpolation,
            'Up'   : self.increaseCycle,
            'Down' : self.decreaseCycle,
            'Right': self.modifyDd,
            'Left' : self.modifyDd,
            '+'    : self.toggleGuidelines,
            '='    : self.resetFK
        }
        # complete the dictionnary with keys of colormaps and species and give them
        # their proper function
        for key in Canvas.colormapDictionnary.keys():
            self.keyactionDictionnary[key] = self.pickColorMap
        for key in Canvas.speciesDictionnary.keys():
            self.keyactionDictionnary[key] = self.pickSpecie

        self.initialize()
        self.framebuffer = gloo.FrameBuffer(color=self.compute["texture"],
                                            depth=gloo.RenderBuffer((self.w, self.h), format='depth'))
        # gloo.set_state(depth_test=True)
        self.show()

    def initialize(self):
        self.h, self.w      = self.cwidth, self.cheight
        # program for computation of Gray-Scott reaction-diffusion sim
        self.compute        = gloo.Program(vertex_shader, compute_fragment, count=4)
        # program for rendering u or v reagent concentration
        self.render         = gloo.Program(vertex_shader, render_hs_fragment, count=4)
        # Programs to render grids, curves, lines and others
        self.pearsonsGrid   = gloo.Program(lines_vertex, lines_fragment)
        self.pearsonsCurve  = gloo.Program(lines_vertex, lines_fragment)
        self.pearsonsCurve2 = gloo.Program(lines_vertex, lines_fragment)
        self.fkLines        = gloo.Program(lines_vertex, lines_fragment)
        self.ddLines        = gloo.Program(lines_vertex, lines_fragment)
        self.hsLines        = gloo.Program(lines_vertex, lines_fragment)

        self.species              = import_pearsons_types()[0]
        # definition of parameters for du, dv, f, k
        self.setSpecie(self.specie)
        # grid initialization, with central random patch
        self.initializeGrid()
        self.initializeParams2()
        self.compute["position"]  = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        self.render["position"]   = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        self.compute["texcoord"]  = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.render["texcoord"]   = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.render['dx']         = 1.0 / self.w
        self.render['dy']         = 1.0 / self.h
        self.compute['ddmin']     = self.ddMin
        self.compute['ddmax']     = self.ddMax
        self.compute['pingpong']  = self.pingpong
        self.render['pingpong']   = self.pingpong
        self.compute['brush']     = self.brush
        self.compute['brushtype'] = self.brushType
        self.hsdir, self.hsalt    = self.getHsDirAlt()
        self.render["hsdir"]      = self.hsdir
        self.render["hsalt"]      = self.hsalt
        self.render["hsz"]        = self.hsz
        self.render["hscmap"]     = get_colormap('hs').map(np.linspace(0.0, 1.0, 1024)).astype('float32')
        self.setColormap(self.cmapName)
        self.render["reagent"]    = 1
        self.initializePlots()

    def setSpecie(self, specie):
        self.specie = specie
        self.printPearsonPatternDescription()
        self.P = np.zeros((self.h, self.w, 4), dtype=np.float32)
        self.P[:, :] = self.species[self.specie][0:4]
        self.modulateFK()
        self.updateComputeParams()

    def initializeGrid(self, event=None):
        print('Initialization of the grid.')
        self.UV = np.zeros((self.h, self.w, 4), dtype=np.float32)
        self.UV[:, :, 0:2] = setup_grid(self.h, self.w)
        self.UV += np.random.uniform(-0.02, 0.1, (self.h, self.w, 4))
        self.UV[:, :, 2] = self.UV[:, :, 0]
        self.UV[:, :, 3] = self.UV[:, :, 1]
        if not hasattr(self, 'texture'):
            self.texture = gloo.texture.Texture2D(data=self.UV, format=gl.GL_RGBA, internalformat='rgba32f')
        else:
            self.texture.set_data(self.UV)
        self.compute["texture"] = self.texture
        self.compute["texture"].interpolation = gl.GL_NEAREST
        self.compute["texture"].wrapping = gl.GL_REPEAT
        self.render["texture"] = self.compute["texture"]
        self.render["texture"].interpolation = gl.GL_LINEAR
        self.render["texture"].wrapping = gl.GL_REPEAT

    def initializeParams2(self):
        print('Initialization of model parameters')
        self.P2 = np.ones((self.h, self.w, 4), dtype=np.float32)
        self.P2[:, :, 0] = 1.0 / self.w
        self.P2[:, :, 1] = 1.0 / self.h
        self.P2[:, :, 2] = (self.dd - self.ddMin) / (self.ddMax - self.ddMin)
        self.P2[:, :, 3] = self.dt
        self.updateComputeParams2()

    def initializePlots(self):
        self.plotFKLines(display=False)
        self.plotDdLines(display=False)
        self.plotHsLines(display=False)

        self.pearsonsGrid["color"] = np.ones((len(PearsonGrid), 4), np.float32) * .25
        self.pearsonsCurve["color"] = np.ones((len(PearsonCurve), 4), np.float32)
        self.pearsonsCurve2["color"] = np.ones((len(PearsonCurve2), 4), np.float32)
        color = np.ones((7+self.cwidth + self.cheight, 4), np.float32)
        color[0] *= 0                                          # starting point
        color[1:3] *= .5                                       # x baseline
        color[3:self.cwidth+3] *= .75                          # x profile
        color[self.cwidth+3] = color[0]                        # hidden point
        color[self.cwidth+4:self.cwidth+6] *= .5               # y baseline
        color[self.cwidth+6:self.cwidth+self.cheight+6] *= .75 # y profile
        color[-1] = color[0]                                   # endpoint
        self.fkLines["color"] = color
        self.ddLines["color"] = np.ones((2 + self.cwidth, 4), np.float32)
        self.hsLines["color"] = np.ones((2, 4), np.float32)

    def setColormap(self, name):
        print('Using colormap %s.' % name)
        self.cmapName = name
        self.render["cmap"] = get_colormap(self.cmapName).map(np.linspace(0.0, 1.0, 1024)).astype('float32')

    def on_draw(self, event):
        with self.framebuffer:
            gloo.set_viewport(0, 0, self.cwidth, self.cheight)
            self.compute.draw('triangle_strip')

            for cycle in range(self.cycle):
                self.pingpong = 1 - self.pingpong
                self.compute["pingpong"] = self.pingpong
                self.compute.draw('triangle_strip')
                self.pingpong = 1 - self.pingpong
                self.compute["pingpong"] = self.pingpong
                self.compute.draw('triangle_strip')

        # rendering the state of the model
        gloo.clear(color=True, depth=True)
        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
        # gloo.set_state(depth_test=True)
        self.render.draw('triangle_strip')
        self.pingpong = 1 - self.pingpong
        self.compute["pingpong"] = self.pingpong
        self.render["pingpong"] = self.pingpong

        self.pearsonsGrid.draw('line_strip')
        self.pearsonsCurve.draw('line_strip')
        self.pearsonsCurve2.draw('line_strip')
        self.fkLines.draw('line_strip')
        self.ddLines.draw('line_strip')
        self.hsLines.draw('line_strip')

        self.update()

    def on_mouse_press(self, event):
        self.mouseDown = True
        (x, y) = event.pos
        (sx, sy) = self.size
        xpos = x/sx
        ypos = 1 - y/sy
        if len(event.modifiers) == 0:
            self.compute['brush'] = [xpos, ypos]
            self.compute['brushtype'] = event.button
        elif len(event.modifiers) == 1 and 'control' in event.modifiers:
            self.mousePressControlPos = [xpos, ypos]
            if self.guidelines:
                self.plotFKLines(display=True)
            self.printFK(event)
        elif len(event.modifiers) == 1 and 'Alt' in event.modifiers:
            self.mousePressAltPos = [xpos, ypos]
            if self.guidelines:
                self.plotFKLines(display=True)
            self.printDfDk(event)
        elif len(event.modifiers) == 1 and 'shift' in event.modifiers:
            self.updateHillshading((xpos, ypos))
            self.plotHsLines((xpos, ypos), display=True)
            self.printHS(event)
        else:
            print(event.modifiers)

    def on_mouse_move(self, event):
        if(self.mouseDown):
            (x, y) = event.pos
            (sx, sy) = self.size
            xpos = x/sx
            ypos = 1 - y/sy
            if len(event.modifiers) == 0:
                # update brush coords here
                self.compute['brush'] = [xpos, ypos]
                self.compute['brushtype'] = event.button
            elif len(event.modifiers) == 1 and 'control' in event.modifiers:
                self.updateFK((xpos, ypos))
                if self.guidelines:
                    self.plotFKLines(display=True)
                self.mousePressControlPos = [xpos, ypos]
                self.printFK(event)
            elif len(event.modifiers) == 1 and 'Alt' in event.modifiers:
                self.modulateFK(pos=(xpos, ypos))
                if self.guidelines:
                    self.plotFKLines(display=True)
                self.mousePressAltPos = [xpos, ypos]
                self.printDfDk(event)
            elif len(event.modifiers) == 1 and 'shift' in event.modifiers:
                self.updateHillshading((xpos, ypos))
                self.plotHsLines((xpos, ypos), display=True)
                self.printHS(event)

    def on_mouse_release(self, event):
        self.mouseDown = False
        self.compute['brush'] = [0, 0]
        self.compute['brushtype'] = 0
        self.plotHsLines(display=False)
        self.plotFKLines(display=False)
        if len(event.modifiers) == 1 and 'control' in event.modifiers:
            self.printFK(event)
        elif len(event.modifiers) == 1 and 'Alt' in event.modifiers:
            self.printDfDk(event)
        elif len(event.modifiers) == 1 and 'shift' in event.modifiers:
            self.printHS(event)

    def on_mouse_wheel(self, event):
        if self.hsz > 0.0:
            self.hsz = np.clip(self.hsz - event.delta[1]/(13/3), 0.5, 2.5)
            print(' Hillshade Z: %1.2f                       ' % self.hsz, end='\r')
            self.render["hsz"] = self.hsz

    def on_key_press(self, event):
        if len(event.text) > 0:
            key = event.text
        else:
            key = event.key.name
        action = self.keyactionDictionnary.get(key)
        if action is not None:
            action(event)

    def switchReagent(self, event=None):
        self.render["reagent"] = 1 - self.render["reagent"]
        reagents = ('U', 'V')
        print('Displaying %s reagent concentration.' % reagents[int(self.render["reagent"])])

    def pickColorMap(self, event):
        colorMapName = self.colormapDictionnary.get(event.text)
        if colorMapName is not None:
            self.setColormap(colorMapName)

    def pickSpecie(self, event):
        specieName = self.speciesDictionnary.get(event.text)
        if specieName is not None:
            self.setSpecie(specieName)

    def updateFK(self, pos):
        fModAmount = pos[1] - self.mousePressControlPos[1]
        kModAmount = pos[0] - self.mousePressControlPos[0]
        f = self.P[0, 0, 2]
        k = self.P[0, 0, 3]
        self.P[:, :, 2] -= f
        self.P[:, :, 3] -= k
        self.P[:, :, 2] = np.clip(self.P[:, :, 2] + (f + 0.006 * fModAmount), self.fMin, self.fMax)
        self.P[:, :, 3] = np.clip(self.P[:, :, 3] + (k + 0.003 * kModAmount), self.kMin, self.kMax)
        self.updateComputeParams()

    def modulateFK(self, pos=None):
        f = self.P[0, 0, 2]
        k = self.P[0, 0, 3]
        if pos:
            self.fModAmount += 0.002 * (pos[1] - self.mousePressAltPos[1])
            self.kModAmount += 0.001 * (pos[0] - self.mousePressAltPos[0])
        rows, cols = self.h, self.w
        sinsF = np.sin(np.linspace(0.0, 2*np.pi, cols))
        sinsK = np.sin(np.linspace(0.0, 2*np.pi, rows))
        for i in range(rows):
            self.P[i, :, 2] = np.clip(f + self.fModAmount*sinsF, self.fMin, self.fMax)
        for i in range(cols):
            self.P[:, i, 3] = np.clip(k + self.kModAmount*sinsK, self.kMin, self.kMax)
        self.updateComputeParams()

    def resetFK(self, event=None):
        self.fModAmount = 0.0
        self.kModAmount = 0.0
        self.modulateFK(pos=None)
        print('f and k constants.')

    def updatedd(self, amount):
        self.dd = np.clip(self.dd + amount, self.ddMin, self.ddMax)
        self.modulateDd(self.ddMod)

    def modulateDd(self, amount):
        ddPivot = (self.dd - self.ddMin) / (self.ddMax - self.ddMin)
        ddUpperProportion = (self.ddMax - self.dd) / (self.ddMax - self.ddMin)
        ddLowerProportion = ddPivot
        gaussGrid = self.gauss(sigma = 0.33)
        if amount > 0.0:
            self.P2[:, :, 2] = (1 - amount) * ddPivot \
                                 + (amount) * (ddPivot \
                                             + (gaussGrid * ddUpperProportion) \
                                             - ((1-gaussGrid) * ddLowerProportion)
                                              )
        elif amount < 0.0:
            self.P2[:, :, 2] = (1 - amount) * ddPivot \
                                 + (amount) * (ddPivot \
                                             - ((1-gaussGrid) * ddUpperProportion) \
                                             + (gaussGrid * ddLowerProportion)
                                              )
        else:
            self.P2[:, :, 2] = ddPivot
        self.updateComputeParams2()

    def modifyDd(self, event):
        modDd = 0
        if event.key.name == 'Right':
            modDd = 0.01
        elif event.key.name == 'Left':
            modDd = -0.01
        if len(event.modifiers) == 0:
            self.updatedd(modDd)
        elif event.modifiers[0] == 'Shift':
            self.ddMod = np.clip(self.ddMod - 2*modDd, -1, 1)
            self.modulateDd(self.ddMod)
        self.printDd()
        self.plotDdLines(self.guidelines)

    def updatedt(self, amount):
        self.dt += amount
        self.P2[:, :, 3] = self.dt
        self.updateComputeParams2()

    def updateComputeParams(self):
        self.params = gloo.texture.Texture2D(data=self.P, format=gl.GL_RGBA, internalformat='rgba32f')
        self.compute["params"] = self.params

    def updateComputeParams2(self):
        self.params2 = gloo.texture.Texture2D(data=self.P2, format=gl.GL_RGBA, internalformat='rgba32f')
        self.compute["params2"] = self.params2

    def increaseCycle(self, event=None):
        if not self.cycle:
            self.cycle = 1
        else:
            self.cycle *= 2
        if self.cycle > 64:
            self.cycle = 64
        print('Number of cycles: %2.0f' % (1 + 2 * self.cycle), end='\r')

    def decreaseCycle(self, event=None):
        self.cycle = int(self.cycle/2)
        if self.cycle < 1:
            self.cycle = 0
        print('Number of cycles: %2.0f' % (1 + 2 * self.cycle), end='\r')

    def toggleHillshading(self, event=None):
        if self.hsz > 0.0:
            self.lastHsz = self.hsz
            self.hsz = 0.0
            print('Hillshading off.')
        else:
            self.hsz = self.lastHsz
            print('Hillshading on.')
        self.render['hsz'] = self.hsz

    def updateHillshading(self, pos):
        hsdir = 180.0/math.pi*math.atan2(
            (pos[0] - 0.5)*2,
            (pos[1] - 0.5)*2
        )
        self.hsdirection = 360 + hsdir if hsdir < 0.0 else hsdir
        self.hsaltitude = 90 - (180.0/math.pi*math.atan(
            math.sqrt(
                ((pos[0] - 0.5)*2) *
                ((pos[0] - 0.5)*2) +
                ((pos[1] - 0.5)*2) *
                ((pos[1] - 0.5)*2)
                )))
        self.hsaltitude *= .5
        self.hsdir, self.hsalt = self.getHsDirAlt(lightDirection=self.hsdirection,
                                                  lightAltitude=self.hsaltitude)
        self.render["hsdir"] = self.hsdir
        self.render["hsalt"] = self.hsalt

    def toggleInterpolation(self, event=None):
        if self.interpolationMode == 'linear':
            self.interpolationMode = 'nearest'
            self.render["texture"].interpolation = gl.GL_NEAREST
        else:
            self.interpolationMode = 'linear'
            self.render["texture"].interpolation = gl.GL_LINEAR
        print('Interpolation mode: %s.' % self.interpolationMode)

    def toggleGuidelines(self, event=None):
        self.guidelines = not self.guidelines
        print('Guidelines: %s' % self.guidelines)

    def printPearsonPatternDescription(self):
        self.title = 'Gray-Scott Reaction-Diffusion: Pattern %s - GregVDS' % self.specie
        print('Pearson\'s Pattern %s' % self.specie)
        print(self.species[self.specie][4])
        print('        dU  dV  f     k \n        %s %s %s %s' % (self.species[self.specie][0],
                                                 self.species[self.specie][1],
                                                 self.species[self.specie][2],
                                                 self.species[self.specie][3]))

    def printFK(self, event):
        begin = ' Current'
        end = "\r"
        if event.type == "mouse_release":
            begin = '     New'
            end="\n"
        print('%s f and k: %1.4f, %1.4f' % (begin, self.P[0, 0, 2], self.P[0, 0, 3]), end=end)

    def printHS(self, event):
        begin = ' Current'
        end = "\r"
        if event.type == "mouse_release":
            begin = '     New'
            end="\n"
        print('%s Hillshading Direction and altitude: %3.0f, %3.0f' % (begin, self.hsdirection, self.hsaltitude), end=end)

    def printDfDk(self, event):
        begin = ' Current'
        end = "\r"
        df = (self.P[int(self.h/4), int(self.w/4), 2] - self.P[int(self.h*3/4), int(self.w*3/4), 2])/2
        dk = (self.P[int(self.h/4), int(self.w/4), 3] - self.P[int(self.h*3/4), int(self.w*3/4), 3])/2
        if event.type == "mouse_release":
            begin = '     New'
            end="\n"
        print('%s df and dk: ±%1.4f, ±%1.4f' % (begin, df, dk), end=end)

    def printDd(self):
        dDVals = [self.dd]
        dDCorner = (self.P2[0,             0,            2])*(self.ddMax - self.ddMin) + self.ddMin
        dDCenter = (self.P2[int(self.h/2), int(self.w/2),2])*(self.ddMax - self.ddMin) + self.ddMin
        dDVals += sorted([dDCorner, dDCenter])
        print("dD: %1.2f, dDMin: %1.2f, dDMax: %1.2f" % (dDVals[0], dDVals[1], dDVals[2]), end="\r")

    def plotFKLines(self, display=True):
        if display is True:
            # get f and k base values and scale them between -1 and 1
            xf = ((self.P[0, 0, 2] - self.fMin)/(self.fMax - self.fMin) - 0.5) * 2
            yf = ((self.P[0, 0, 3] - self.kMin)/(self.kMax - self.kMin) - 0.5) * 2

            P = np.zeros((7+self.cwidth + self.cheight, 2), np.float32)

            startingPoint = P[0]
            x_baseline = P[1:3]
            x_profile = P[3:self.cwidth+3]
            hiddenPoint = P[self.cwidth+3]
            y_baseline = P[self.cwidth+4:self.cwidth+4+2]
            y_profile = P[self.cwidth+4+2:self.cwidth+self.cheight+4+2]
            endingPoint = P[-1]
            #
            startingPoint[...] = (-1, xf)
            x_baseline[...] = (-1, xf), (1, xf)
            hiddenPoint[...] = (1.01, -1.01)
            y_baseline[...] = (yf, -1), (yf, 1)
            endingPoint[...] = (yf, 1)

            #
            x_profile[1:-1, 0] = np.linspace(-1, 1, self.cwidth-2)
            x_profile[1:-1, 1] = (((self.P[0, :, 2] - self.fMin)/(self.fMax - self.fMin) - 0.5) * 2)[1:-1]
            x_profile[0] = (-1, xf)
            x_profile[-1] = (1.01, xf)
            #
            y_profile[1:-1, 0] = (((self.P[:, 0, 3] - self.kMin)/(self.kMax - self.kMin) - 0.5) * 2)[1:-1]
            y_profile[1:-1, 1] = np.linspace(-1, 1, self.cheight-2)
            y_profile[0] = (yf, -1.01)
            y_profile[-1] = (yf, 1)

            self.pearsonsGrid["position"] = PearsonGrid
            self.pearsonsCurve["position"] = PearsonCurve
            self.pearsonsCurve2["position"] = PearsonCurve2
            self.fkLines["position"] = P
        else:
            self.pearsonsGrid["position"] = np.zeros((len(PearsonGrid), 2), np.float32)
            self.pearsonsCurve["position"] = np.zeros((len(PearsonCurve), 2), np.float32)
            self.pearsonsCurve2["position"] = np.zeros((len(PearsonCurve2), 2), np.float32)
            self.fkLines["position"] = np.zeros((7+self.cwidth + self.cheight, 2), np.float32)

    def plotDdLines(self, display=True):
        if display is True:
            ddPivot = (((self.dd - self.ddMin) / (self.ddMax - self.ddMin))-0.5) *2
            P = np.zeros((2 + self.cwidth, 2), np.float32)
            baseline = P[0:2]
            profile  = P[2:2 + self.cwidth]

            baseline[...] = (1.01, ddPivot), (-1.01, ddPivot)
            profile[1:-1, 0] = np.linspace(-1, 1, self.cwidth-2)
            profile[1:-1, 1] = (((self.P2[int(self.cwidth/2), :, 2]) - 0.5) * 2)[1:-1]
            profile[0] = profile[1]
            profile[-1] = profile[-2]
            self.ddLines["position"] = P
        else:
            self.ddLines["position"] = np.zeros((2 + self.cwidth, 2), np.float32)

    def plotHsLines(self, pos=None, display=True):
        if display is True and pos is not None:
            hsLine = np.zeros((2, 2), np.float32)
            hsLine[0] = ((pos[0]-0.5)*2.0, (pos[1]-0.5)*2.0)
            self.hsLines["position"] = hsLine
        else:
            self.hsLines["position"] = np.zeros((2, 2), np.float32)

    def getHsDirAlt(self, lightDirection=315, lightAltitude=20):
        hsdir = 360.0 - lightDirection + 90.0
        if hsdir >= 360.0:
            hsdir -= 360.0
        hsdir *= math.pi / 180.0
        hsalt = (90 - lightAltitude) * math.pi / 180.0
        return (hsdir, hsalt)

    def gauss(self, sigma=1, muu=0.000):
        x, y = np.meshgrid(np.linspace(-1, 1, self.h), np.linspace(-1, 1, self.w))
        dst = np.sqrt(x*x+y*y)
        return np.exp(-((dst - muu)**2 / (2.0 * sigma**2)))

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


################################################################################

class checkDt(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, np.clip(values, .5, 2.0))


class checkDd(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, np.clip(values, .2, 1.3))

################################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=textwrap.dedent(__doc__),
                                     epilog= textwrap.dedent("""Examples:
    python3 gs.py
    python3 gs.py -c osmort -l 0
    python3 gs.py -s 512 -p kappa_left -c oslo -l 1.5 -d .5 -t 2.0
    python3 gs.py -s 512 -p alpha_left -c detroit -l 0.8 -d .3 -g"""),
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-s",
                        "--Size",
                        type=int,
                        default=1024,
                        help="Size of model")
    parser.add_argument("-p",
                        "--Pattern",
                        choices=Canvas.speciesDictionnary.values(),
                        default="lambda_left",
                        help="Pearson\' pattern")
    parser.add_argument("-c",
                        "--Colormap",
                        choices=Canvas.colormapDictionnary.values(),
                        default="irkoutsk",
                        help="Colormap used")
    parser.add_argument("-l",
                        "--Hillshade",
                        type=float,
                        default=2.0,
                        help="Hillshading presentation, 0 for none, 0.0 - 2.5")
    parser.add_argument("-d",
                        "--dD",
                        type=float,
                        action=checkDd,
                        default=1.0,
                        help="Space step dD of the Gray-Scott Model, 0.2 - 1.3")
    parser.add_argument("-t",
                        "--dT",
                        type=float,
                        action=checkDt,
                        default=1.0,
                        help="Time step dT of the Gray-Scott Model, 0.2 - 2.0")
    parser.add_argument("-g",
                        "--guidelines",
                        action='store_true',
                        help="plot guidelines when modifing/modulating f and k values")
    args = parser.parse_args()

    # 'debug' to show the colormaps
    # plot_color_gradients("Custom colormaps", [Canvas.colormapDictionnary[each] for each in Canvas.colormapDictionnary.keys()])
    # plt.show()
    # just close the window to run the model

    c = Canvas(size=(args.Size, args.Size),
               specie=args.Pattern,
               cmap=args.Colormap,
               hsz=args.Hillshade,
               dt=args.dT,
               dd=args.dD,
               guidelines=args.guidelines)
    c.measure_fps2(window=2)
    app.run()
