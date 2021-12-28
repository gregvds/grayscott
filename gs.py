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
from numpy.random import rand

import vispy.color as color
from vispy import app, gloo
from vispy.gloo import gl

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from shaders import vertex_shader, compute_fragment, render_hs_fragment
from shaders import lines_vertex, lines_fragment
from systems import pearson, PearsonGrid, PearsonCurve


def get_colormap(name, size=256):
    vispy_cmaps = color.get_colormaps()
    if name in vispy_cmaps:
        cmap = color.get_colormap(name)
    else:
        try:
            mpl_cmap = plt.get_cmap(name)
        except AttributeError:
            raise KeyError(
                f'Colormap "{name}" not found in either vispy '
                'or matplotlib.'
            )
        mpl_colors = mpl_cmap(np.linspace(0, 1, size))
        cmap = color.Colormap(mpl_colors)
    return cmap


def invertCmapName(cmap):
    # a very crude reversing name method for Matplotlib colormap inversion
    if '_r' in cmap:
        return(cmap.split('_')[0])
    else:
        return(cmap+'_r')


def createAndRegisterCmap(name, cmapNameList, proportions=None):
    # routine to create custom Matplotlib colormap and its reversed version
    newColors = []
    newColors_r = []
    if proportions and len(proportions) != len(cmapNameList):
        raise KeyError(
            f'cmapNameList and proportions for Colormap {name}'
            'have not the same length.')
    if proportions is None:
        proportions = 32 * np.ones(1, len(cmapNameList))

    for cmapName, proportion in zip(cmapNameList, proportions):
        newColors.append(cm.get_cmap(cmapName, proportion)
                         (np.linspace(0, 1, proportion)))
    newcmp = ListedColormap(np.vstack(newColors), name=name)
    cmapNameList.reverse()
    proportions.reverse()
    for cmapName, proportion in zip(cmapNameList, proportions):
        newColors_r.append(cm.get_cmap(invertCmapName(
            cmapName), proportion)(np.linspace(0, 1, proportion)))
    newcmp_r = ListedColormap(np.vstack(newColors_r),
                              name=invertCmapName(name))
    cm.register_cmap(name=name, cmap=newcmp)
    cm.register_cmap(name=invertCmapName(name), cmap=newcmp_r)


def createAndRegisterLinearSegmentedCmap(name, colors, nodes=None):
    # routine to create custom Matplotlib colormap and its reversed version
    if nodes and len(nodes) != len(colors):
        raise KeyError(
            f'colors and nodes for Colormap {name}'
            'have not the same length.')
    if nodes is None:
        nodes = np.linspace(0, 1, len(colors))

    newcmp = LinearSegmentedColormap.from_list(name, list(zip(nodes, colors)))
    colors.reverse()
    nodes.reverse()
    nodes = [1-x for x in nodes]
    newcmp_r = LinearSegmentedColormap.from_list(invertCmapName(name), list(zip(nodes, colors)))
    cm.register_cmap(name=name, cmap=newcmp)
    cm.register_cmap(name=invertCmapName(name), cmap=newcmp_r)


def createColormaps():
    # definition of some custom colormaps
    createAndRegisterCmap(
        'malmo', [
            'bone_r',
            'Blues_r',
            'GnBu_r',
            'Greens'],
        proportions=[512, 128, 512, 128])
    createAndRegisterCmap(
        'Boston', [
            'bone_r',
            'gray',
            'pink_r',
            'YlOrBr_r'],
        proportions=[960, 160, 160, 960])
    # createAndRegisterCmap(
    #     'seattle', [
    #         'bone_r',
    #         'Blues_r',
    #         'bone_r',
    #         'gray',
    #         'pink_r',
    #         'YlOrBr_r'],
    #     proportions=[160, 480, 3200, 640, 640, 3840])
    createAndRegisterLinearSegmentedCmap(
        'detroit', [
            'white',
            'lightblue',
            'cadetblue',
            'black',
            'cadetblue',
            'lightblue',
            'white'
        ],
        nodes=[0.0, 0.4, 0.47, 0.5, 0.53, 0.6, 1.0])
    createAndRegisterLinearSegmentedCmap(
        'antidetroit', [
            'black',
            'cadetblue',
            'lightblue',
            'white',
            'lightblue',
            'cadetblue',
            'black'
        ],
        nodes=[0.0, 0.4, 0.47, 0.5, 0.53, 0.6, 1.0])
    createAndRegisterLinearSegmentedCmap(
        'uppsala', [
            'slategrey',
            'goldenrod',
            'antiquewhite',
            'olivedrab',
            'lawngreen',
            'lightseagreen',
            'paleturquoise'
        ],
        nodes=[0.0, 0.4, 0.46, 0.5, 0.58, 0.75, 1.0])
    createAndRegisterLinearSegmentedCmap(
        'oslo', [
            'black',
            'saddlebrown',
            'sienna',
            'white',
            'aliceblue',
            'skyblue'
        ],
        nodes=[0.0, 0.48, 0.5, 0.51, 0.57, 1.0])
    createAndRegisterLinearSegmentedCmap(
        'Lochinver', [
            'floralwhite',
            'antiquewhite',
            'wheat',
            'cornflowerblue',
            'dodgerblue',
            'powderblue'
        ],
        nodes=[0.0, 0.4, 0.45, 0.5, 0.65, 1.0])
    createAndRegisterLinearSegmentedCmap(
        'tromso', [
            'black',
            'black',
            'chocolate',
            'ivory',
            'sandybrown',
            'black',
            'black'
        ],
        nodes=[0.0, 0.3, 0.47, 0.5, 0.53, 0.7, 1.0])
    createAndRegisterLinearSegmentedCmap(
        'osmort', [
            'linen',
            'linen',
            'chocolate',
            'black',
            'sandybrown',
            'linen',
            'linen'
        ],
        nodes=[0.0, 0.4, 0.47, 0.5, 0.53, 0.6, 1.0])
    createAndRegisterLinearSegmentedCmap(
        'irkoutsk', [
            'darkkhaki',
            'olive',
            'dimgray',
            'seashell',
            'yellowgreen',
            'darkolivegreen',
            'olivedrab'
        ],
        nodes=[0.0, 0.4, 0.47, 0.5, 0.53, 0.6, 1.0])
    createAndRegisterLinearSegmentedCmap(
        'krasnoiarsk', [
            'darkgoldenrod',
            'goldenrod',
            'dimgray',
            'cornsilk',
            'gold',
            'tan',
            'burlywood'
        ],
        nodes=[0.0, 0.4, 0.47, 0.5, 0.53, 0.6, 1.0])
    createAndRegisterLinearSegmentedCmap(
        'Rejkjavik', [
            'rebeccapurple',
            'indigo',
            'darkorchid',
            'gold',
            'darkorange',
            'orange',
            'gold'
        ],
        nodes=[0.0, 0.3, 0.47, 0.5, 0.53, 0.7, 1.0])
    createAndRegisterLinearSegmentedCmap(
        'hs', [
            'black',
            '#100926',
            'grey',
            'gold',
            'lemonchiffon',
            'azure'
        ],
        nodes=[0.0, 0.2, 0.5, 0.6, 0.7, 1.0])


def plot_color_gradients(cmap_category, cmap_list):
    # Create figure and adjust figure height to number of colormaps
    nrows = len(cmap_list)
    figh = 0.35 + 0.15 + (nrows + (nrows-1)*0.1)*0.22
    fig, axs = plt.subplots(nrows=nrows, figsize=(6.4, figh))
    fig.subplots_adjust(top=1-.35/figh, bottom=.15/figh, left=0.2, right=0.99)

    axs[0].set_title(cmap_category + ' colormaps', fontsize=14)

    for ax, cmap_name in zip(axs, cmap_list):
        ax.imshow(gradient, aspect='auto', cmap=cmap_name)
        ax.text(-.01, .5, cmap_name, va='center', ha='right', fontsize=10,
                transform=ax.transAxes)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axs:
        ax.set_axis_off()


def import_pearsons_types():
    species = {}
    type = None
    for each in pearson.keys():
        type = pearson[each]
        species[each] = [type['factors']['Da'],
                         type['factors']['Db'],
                         type['factors']['f'],
                         type['factors']['k'],
                         type['description']]
    return species


def setup_grid(rows, cols, blob_scale=0.1):
    # 20x20 mesh point area located symmetrically around center to U(a) = 1/2,  V(b)=1/4.
    # These conditions perturbed +/- 1% random noise to break the square symmetery
    # grid is and two layers rows by cols numpy array filled first with 1
    grid = np.ones((rows, cols, 2))
    # the second layer is filled with 0
    grid[:, :, 1] = 0.0

    # here we seed the central part of the grid with randomized .5 on first layer and .25 on second layer values
    DEV = 0.09
    from_row = int((rows/2) - rows*blob_scale/2)
    to_row = int((rows/2) + rows*blob_scale/2)
    from_col = int((cols / 2) - cols * blob_scale / 2)
    to_col = int((cols / 2) + cols * blob_scale / 2)

    for i in range(from_row, to_row):
        for j in range(int(np.random.uniform(1-DEV, 1+DEV)*from_col),
                       int(np.random.uniform(1-DEV, 1+DEV)*to_col)
                       ):
            grid[i, j, 0] = 0.5
            grid[i, j, 1] = 0.25

    grid[from_row:to_row, from_col:to_col, :] = (
                            (1+rand(to_row-from_row, to_col-from_col, 2) / 100)

                            * grid[from_row:to_row, from_col:to_col, :]
    )

    return grid


################################################################################


class Canvas(app.Canvas):
    # Pearson's patterns related variables
    fMin = 0.0
    fMax = 0.08
    kMin = 0.03
    kMax = 0.07
    ddMin = 0.2
    ddMax = 1.3
    ddMod = 0.0
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
        '1': 'Boston_r',
        '&': 'Boston',
        '2': 'malmo',
        'é': 'malmo_r',
        '3': 'uppsala',
        '"': 'uppsala_r',
        '4': 'oslo_r',
        '\'': 'oslo',
        '5': 'Lochinver',
        '(': 'Lochinver_r',
        '6': 'Rejkjavik',
        '§': 'Rejkjavik_r',
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

    # brush to add reagent v
    brush                   = np.zeros((1, 1, 2), dtype=np.float32)
    brushType               = 0

    mouseDown = False
    mousePressControlPos    = [0.0, 0.0]
    mousePressAltPos        = [0.0, 0.0]
    fModAmount              = 0
    kModAmount              = 0

    # pipeline toggling parameter
    pingpong                = 1

    # Hillshading lightning parameters
    hsdirection             = 0
    hsaltitude              = 0
    hsdir                   = 0
    hsalt                   = 0

    # interpolation
    interpolationMode       = 'linear'

    # cycle of computation per frame
    cycle                   = 0

    # ? better computation for diffusion and concentration ?
    gl.GL_FRAGMENT_PRECISION_HIGH = 1

    # program for computation of Gray-Scott reaction-diffusion sim
    compute = gloo.Program(vertex_shader, compute_fragment, count=4)

    # program for rendering u or v reagent concentration
    render = gloo.Program(vertex_shader, render_hs_fragment, count=4)

    # Programs to render grids, curves, lines and others
    pearsonsGrid = gloo.Program(lines_vertex, lines_fragment)
    pearsonsCurve = gloo.Program(lines_vertex, lines_fragment)
    fkLines = gloo.Program(lines_vertex, lines_fragment)
    ddLines = gloo.Program(lines_vertex, lines_fragment)
    hsLines = gloo.Program(lines_vertex, lines_fragment)

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
        self.guidelines            = guidelines
        self.initialize()
        self.framebuffer = gloo.FrameBuffer(color=self.compute["texture"],
                                            depth=gloo.RenderBuffer((self.w, self.h)))
        self.show()

    def initialize(self):
        self.h, self.w            = self.cwidth, self.cheight
        self.species              = import_pearsons_types()
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
        self.P[:, :] = self.species[specie][0:4]
        self.modulateFK()
        self.updateComputeParams()

    def initializeGrid(self):
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
        self.pearsonsGrid["position"] = np.zeros((len(PearsonGrid), 2), np.float32)
        self.pearsonsGrid["color"] = np.ones((len(PearsonGrid), 4), np.float32) * .25

        self.pearsonsCurve["position"] = np.zeros((len(PearsonCurve), 2), np.float32)
        self.pearsonsCurve["color"] = np.ones((len(PearsonCurve), 4), np.float32)

        self.fkLines["position"] = np.zeros((7+self.cwidth + self.cheight, 2), np.float32)
        color = np.ones((7+self.cwidth + self.cheight, 4), np.float32)
        color[0] *= 0                                          # starting point
        color[1:3] *= .5                                       # x baseline
        color[3:self.cwidth+3] *= .75                          # x profile
        color[self.cwidth+3] = color[0]                        # hidden point
        color[self.cwidth+4:self.cwidth+6] *= .5               # y baseline
        color[self.cwidth+6:self.cwidth+self.cheight+6] *= .75 # y profile
        color[-1] = color[0]                                   # endpoint
        self.fkLines["color"] = color

        self.ddLines["position"] = np.zeros((2 + self.cwidth, 2), np.float32)
        self.ddLines["color"] = np.ones((2 + self.cwidth, 4), np.float32)

        self.hsLines["position"] = np.zeros((2, 2), np.float32)
        self.hsLines["color"] = np.ones((2, 4), np.float32)

    def setColormap(self, name):
        print('Using colormap %s.' % name)
        self.cmapName = name
        self.cm = get_colormap(self.cmapName)
        self.render["cmap"] = self.cm.map(np.linspace(0.0, 1.0, 1024)).astype('float32')

    def on_draw(self, event):
        # holding rendering output
        self.framebuffer.activate()
        gloo.set_viewport(0, 0, self.cwidth, self.cheight)
        self.compute.draw('triangle_strip')

        for cycle in range(self.cycle):
            self.pingpong = 1 - self.pingpong
            self.compute["pingpong"] = self.pingpong
            self.compute.draw(gl.GL_TRIANGLE_STRIP)
            self.pingpong = 1 - self.pingpong
            self.compute["pingpong"] = self.pingpong
            self.compute.draw(gl.GL_TRIANGLE_STRIP)

        # releasing rendering output
        self.framebuffer.deactivate()

        # rendering the state of the model
        gloo.clear(color=True, depth=True)
        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
        self.render.draw('triangle_strip')
        self.pingpong = 1 - self.pingpong
        self.compute["pingpong"] = self.pingpong
        self.render["pingpong"] = self.pingpong

        self.pearsonsGrid.draw('line_strip')
        self.pearsonsCurve.draw('line_strip')
        self.fkLines.draw('line_strip')
        self.ddLines.draw('line_strip')
        self.hsLines.draw('line_strip')

        self.update()

    def on_mouse_press(self, event):
        self.mouseDown = True
        xpos = event.pos[0]/self.size[0]
        ypos = 1 - event.pos[1]/self.size[1]
        if len(event.modifiers) == 0:
            self.compute['brush'] = [xpos, ypos]
            self.compute['brushtype'] = event.button
        elif len(event.modifiers) == 1 and 'control' in event.modifiers:
            self.mousePressControlPos = [xpos, ypos]
            if self.guidelines:
                self.plotFKLines()
            self.printFK(event)
        elif len(event.modifiers) == 1 and 'shift' in event.modifiers:
            self.updateHillshading((xpos, ypos))
            self.plotHsLines((xpos, ypos))
            self.printHS(event)
        elif len(event.modifiers) == 1 and 'Alt' in event.modifiers:
            self.mousePressAltPos = [xpos, ypos]
            if self.guidelines:
                self.plotFKLines()
            self.printDfDk(event)
        else:
            print(event.modifiers)

    def on_mouse_move(self, event):
        if(self.mouseDown):
            xpos = event.pos[0]/self.size[0]
            ypos = 1 - event.pos[1]/self.size[1]
            if len(event.modifiers) == 0:
                # update brush coords here
                self.compute['brush'] = [xpos, ypos]
                self.compute['brushtype'] = event.button
            elif len(event.modifiers) == 1 and 'control' in event.modifiers:
                # update f and k values according to the x and y movements
                self.updateFK((xpos, ypos))
                if self.guidelines:
                    self.plotFKLines()
                self.mousePressControlPos = [xpos, ypos]
                self.printFK(event)
            elif len(event.modifiers) == 1 and 'shift' in event.modifiers:
                self.updateHillshading((xpos, ypos))
                self.plotHsLines((xpos, ypos))
                self.printHS(event)
            elif len(event.modifiers) == 1 and 'Alt' in event.modifiers:
                self.modulateFK(pos=(xpos, ypos))
                if self.guidelines:
                    self.plotFKLines()
                self.mousePressAltPos = [xpos, ypos]
                self.printDfDk(event)

    def on_mouse_release(self, event):
        self.mouseDown = False
        if len(event.modifiers) == 0:
            self.compute['brush'] = [0, 0]
            self.compute['brushtype'] = 0
        elif len(event.modifiers) == 1 and 'control' in event.modifiers:
            self.printFK(event)
        elif len(event.modifiers) == 1 and 'shift' in event.modifiers:
            self.printHS(event)
            self.hideHsLines()
        elif len(event.modifiers) == 1 and 'Alt' in event.modifiers:
            self.printDfDk(event)
        self.hideFKLines()

    def on_mouse_wheel(self, event):
        if self.hsz > 0.0:
            delta = event.delta[1]
            hsz = self.hsz
            if delta < 0:
                hsz += 0.1
            elif delta > 0:
                hsz -= 0.1
            self.hsz = np.clip(hsz, 0.5, 2.5)
            print(' Hillshade Z: %1.2f                       ' % self.hsz, end='\r')
            self.render["hsz"] = self.hsz

    def on_key_press(self, event):
        if event.text == ' ':
            self.initializeGrid()
        elif event.text in self.speciesDictionnary.keys():
            self.setSpecie(self.speciesDictionnary[event.text])
        elif event.text == '/':
            self.switchReagent()
        elif event.text in self.colormapDictionnary.keys():
            self.setColormap(self.colormapDictionnary[event.text])
        elif event.text == '$':
            self.toggleHillshading()
        elif event.text == '*':
            self.toggleInterpolation()
        elif event.text == '+':
            self.guidelines = not self.guidelines
            print('Guidelines: %s' % self.guidelines)
        elif event.text == '=':
            self.fModAmount = 0.0
            self.kModAmount = 0.0
            self.modulateFK(pos=None)
            print('f and k constants.')
        elif event.key and event.key.name == 'Up':
            self.increaseCycle()
            print('Number of cycles: %2.0f' % (1 + 2* self.cycle), end='\r')
        elif event.key and event.key.name == 'Down':
            self.decreaseCycle()
            print('Number of cycles: %2.0f' % (1 + 2* self.cycle), end='\r')
        elif event.key and event.key.name == 'Right':
            if len(event.modifiers) == 0:
                self.updatedd(0.01)
            elif event.modifiers[0] == 'Shift':
                self.ddMod = np.clip(self.ddMod + 0.02, -1, 1)
                self.modulateDd(self.ddMod)
            self.printDd()
            self.plotDdLines(self.guidelines)
        elif event.key and event.key.name == 'Left':
            if len(event.modifiers) == 0:
                self.updatedd(-0.01)
            elif event.modifiers[0] == 'Shift':
                self.ddMod = np.clip(self.ddMod - 0.02, -1, 1)
                self.modulateDd(self.ddMod)
            self.printDd()
            self.plotDdLines(self.guidelines)

    def switchReagent(self):
        if self.render["reagent"] == 1:
            self.render["reagent"] = 0
            print('Displaying V.')
        else:
            self.render["reagent"] = 1
            print('Displaying U.')

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
        rows = self.h
        cols = self.w
        sinsF = np.sin(np.linspace(0.0, 2*np.pi, cols))
        sinsK = np.sin(np.linspace(0.0, 2*np.pi, rows))
        for i in range(rows):
            self.P[i, :, 2] = np.clip(f + self.fModAmount*sinsF, self.fMin, self.fMax)
        for i in range(cols):
            self.P[:, i, 3] = np.clip(k + self.kModAmount*sinsK, self.kMin, self.kMax)
        self.updateComputeParams()

    def updatedd(self, amount):
        self.dd = np.clip(self.dd + amount, self.ddMin, self.ddMax)
        self.modulateDd(self.ddMod)

    def modulateDd(self, amount):
        ddPivot = (self.dd - self.ddMin) / (self.ddMax - self.ddMin)
        if amount > 0.0:
            ddUpperProportion = (self.ddMax - self.dd) / (self.ddMax - self.ddMin)
            ddLowerProportion = (self.dd - self.ddMin) / (self.ddMax - self.ddMin)
            gaussGrid = self.gauss(sigma = 0.33)
            self.P2[:, :, 2] = (1 - amount) * ddPivot \
                                 + (amount) * (ddPivot \
                                             + (gaussGrid * ddUpperProportion) \
                                             - ((1-gaussGrid) * ddLowerProportion)
                                              )
        elif amount < 0.0:
            ddUpperProportion = (self.ddMax - self.dd) / (self.ddMax - self.ddMin)
            ddLowerProportion = (self.dd - self.ddMin) / (self.ddMax - self.ddMin)
            gaussGrid = self.gauss(sigma = 0.33)
            self.P2[:, :, 2] = (1 - amount) * ddPivot \
                                 + (amount) * (ddPivot \
                                             - ((1-gaussGrid) * ddUpperProportion) \
                                             + (gaussGrid * ddLowerProportion)
                                              )
        else:
            self.P2[:, :, 2] = ddPivot
        self.updateComputeParams2()

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

    def increaseCycle(self):
        if not self.cycle:
            self.cycle = 1
        else:
            self.cycle *= 2
        if self.cycle > 64:
            self.cycle = 64

    def decreaseCycle(self):
        self.cycle = int(self.cycle/2)
        if self.cycle < 1:
            self.cycle = 0

    def toggleHillshading(self):
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
        # self.hsdirection -= 180
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

    def toggleInterpolation(self):
        if self.interpolationMode == 'linear':
            self.interpolationMode = 'nearest'
            self.render["texture"].interpolation = gl.GL_NEAREST
        else:
            self.interpolationMode = 'linear'
            self.render["texture"].interpolation = gl.GL_LINEAR
        print('Interpolation mode: %s.' % self.interpolationMode)

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

    def plotFKLines(self):
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
        self.fkLines["position"] = P

    def hideFKLines(self):
        self.pearsonsGrid["position"] = np.zeros((len(PearsonGrid), 2), np.float32)
        self.pearsonsCurve["position"] = np.zeros((len(PearsonCurve), 2), np.float32)
        self.fkLines["position"] = np.zeros((7+self.cwidth + self.cheight, 2), np.float32)

    def plotDdLines(self, display=True):
        if display:
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

    def plotHsLines(self, pos):
        hsLine = np.zeros((2,2), np.float32)
        hsLine[0] = ((pos[0]-0.5)*2.0, (pos[1]-0.5)*2.0)
        self.hsLines["position"] = hsLine

    def hideHsLines(self):
        self.hsLines["position"] = np.zeros((2, 2), np.float32)

    def getHsDirAlt(self, lightDirection=315, lightAltitude=20):
        hsdir = 360.0 - lightDirection + 90.0
        if hsdir >= 360.0:
            hsdir -= 360.0
        hsdir *= math.pi / 180.0
        hsalt = (90 - lightAltitude) * math.pi / 180.0
        return (hsdir, hsalt)

    def gauss(self, sigma = 1, muu = 0.000):
        x, y = np.meshgrid(np.linspace(-1, 1, self.h), np.linspace(-1, 1, self.w))
        dst = np.sqrt(x*x+y*y)
        return np.exp(-( (dst-muu)**2 / ( 2.0 * sigma**2 ) ) )

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

    # 'debug' just to show the colormaps
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
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
