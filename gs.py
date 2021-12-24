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
    Several colormaps are available via 1 - 8, shifted for reversed version.
    Mouse left click in the grid refills reagent v at 0.5.
    Mouse right click in the grid put reagent v at 0.
    Ctrl + Mouse click and drag to modify globaly feed and kill rates.
    Alt + Mouse click and drag to vary sinusoidally feed and kill rates.
    Shift + Mouse click and drag modifies the hillshading parameters.
    key / switch presentation between u and v.
    key * toggles hillshading on or off.
    key $ toggles interpolation on or off.
    key Up/Down arrows to multiply computation cycles/frame
    Spacebar reseeds the grid.
"""

import math

import numpy as np
from numpy.random import rand

import vispy.color as color
from vispy import app, gloo
from vispy.gloo import gl

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from shaders import vertex_shader, compute_fragment, render_hs_fragment
from systems import pearson


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
        nodes=[0.0, 0.48, 0.5, 0.53, 0.6, 1.0])
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
    # fps = 60

    cwidth, cheight = 512, 512
    w, h = cwidth, cheight
    dt = 1.0
    dd = 1.0

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
    cm = get_colormap('malmo_r')

    specie = 'lambda_left'

    # Array for du, dv, f, k
    P = np.zeros((h, w, 4), dtype=np.float32)

    # Arrays for reagents u, v
    UV = np.zeros((h, w, 4), dtype=np.float32)

    # brush to add reagent v
    brush = np.zeros((1, 1, 2), dtype=np.float32)
    brushType = 0

    mouseDown = False
    mousePressControlPos = [0.0, 0.0]
    mousePressAltPos = [0.0, 0.0]
    fModAmount = 0
    kModAmount = 0

    # pipeline toggling parameter
    pingpong = 1

    # Hillshading lightning parameters
    hsdirection = 0
    hsaltitude = 0
    hsdir = 0
    hsalt = 0
    hsz = 2.0
    lastHsz = 2.0

    # interpolation
    interpolationMode = 'linear'

    # cycle of computation per frame
    cycle = 0

    # ? better computation for diffusion and concentration ?
    gl.GL_FRAGMENT_PRECISION_HIGH = 1

    # program for computation of Gray-Scott reaction-diffusion sim
    compute = gloo.Program(vertex_shader, compute_fragment, count=4)

    # program for rendering u or v reagent concentration
    render = gloo.Program(vertex_shader, render_hs_fragment, count=4)

    def __init__(self, size=(1024, 1024), scale=0.5):
        super().__init__(size=size,
                         title='Gray-Scott Reaction-Diffusion - GregVDS',
                         keys='interactive',
                         resizable=False,
                         dpi=221)
        self.cwidth, self.cheight = int(size[0]*scale), int(size[1]*scale)
        self.initialize()
        self.framebuffer = gloo.FrameBuffer(color=self.compute["texture"],
                                            depth=gloo.RenderBuffer((self.w, self.h)))
        self.show()

    def initialize(self):
        self.h, self.w = self.cwidth, self.cheight
        self.species = import_pearsons_types()
        print('Pearson\'s Pattern %s' % self.specie)
        print(self.species[self.specie][4])
        print('        dU  dV  f     k \n        %s %s %s %s' % (self.species[self.specie][0],
                                                 self.species[self.specie][1],
                                                 self.species[self.specie][2],
                                                 self.species[self.specie][3]))
        # definition of parameters for du, dv, f, k
        self.P = np.zeros((self.h, self.w, 4), dtype=np.float32)
        self.P[:, :] = self.species[self.specie][0:4]
        self.params = gloo.texture.Texture2D(data=self.P, format=gl.GL_RGBA, internalformat='rgba32f')
        # grid initialization, with central random patch
        self.UV = np.zeros((self.h, self.w, 4), dtype=np.float32)
        self.UV[:, :, 0:2] = setup_grid(self.h, self.w)
        self.UV += np.random.uniform(0.0, 0.01, (self.h, self.w, 4))
        self.UV[:, :, 2] = self.UV[:, :, 0]
        self.UV[:, :, 3] = self.UV[:, :, 1]
        # hillshading parameters computation
        self.hsdir, self.hsalt = self.getHsDirAlt()
        # setting the programs
        self.compute["params"] = self.params
        self.texture = gloo.texture.Texture2D(data=self.UV, format=gl.GL_RGBA, internalformat='rgba32f')
        self.compute["texture"] = self.texture
        self.compute["texture"].interpolation = gl.GL_NEAREST
        self.compute["texture"].wrapping = gl.GL_REPEAT
        self.compute["position"] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        self.compute["texcoord"] = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.compute['dt'] = self.dt
        self.compute['dx'] = 1.0 / self.w
        self.compute['dy'] = 1.0 / self.h
        self.compute['dd'] = self.dd
        self.compute['pingpong'] = self.pingpong
        self.compute['brush'] = self.brush
        self.compute['brushtype'] = self.brushType
        self.render["hsdir"] = self.hsdir
        self.render["hsalt"] = self.hsalt
        self.render["hsz"] = self.hsz
        self.render['dx'] = 1.0 / self.w
        self.render['dy'] = 1.0 / self.h
        self.render["texture"] = self.compute["texture"]
        self.render["texture"].interpolation = gl.GL_LINEAR
        self.render["texture"].wrapping = gl.GL_REPEAT
        self.render["position"] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        self.render["texcoord"] = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.render["cmap"] = self.cm.map(np.linspace(0.0, 1.0, 1024)).astype('float32')
        self.render["reagent"] = 1
        self.render['pingpong'] = self.pingpong

    def on_draw(self, event):
        # holding rendering output
        self.framebuffer.activate()
        gl.glViewport(0, 0, self.cwidth, self.cheight)
        self.compute.draw(gl.GL_TRIANGLE_STRIP)

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
        gloo.clear(color=True)
        gl.glViewport(0, 0, self.physical_size[0], self.physical_size[1])
        self.render.draw(gl.GL_TRIANGLE_STRIP)
        self.pingpong = 1 - self.pingpong
        self.compute["pingpong"] = self.pingpong
        self.render["pingpong"] = self.pingpong

        self.update()

    def on_mouse_press(self, event):
        self.mouseDown = True
        if len(event.modifiers) == 0:
            self.compute['brush'] = [event.pos[0]/self.size[0],
                                     1 - event.pos[1]/self.size[1]]
            self.compute['brushtype'] = event.button
        elif len(event.modifiers) == 1 and 'control' in event.modifiers:
            self.mousePressControlPos = [event.pos[0]/self.size[0],
                                         1 - event.pos[1]/self.size[1]]
            print(' Current f and k: %1.4f, %1.4f' % (self.P[0, 0, 2],
                                               self.P[0, 0, 3]), end="\r")
        elif len(event.modifiers) == 1 and 'shift' in event.modifiers:
            self.updateHillshading(event)
            print(' Current Hillshading Direction and altitude: %3.0f, %3.0f' % (self.hsdir, self.hsalt), end="\r")
        elif len(event.modifiers) == 1 and 'Alt' in event.modifiers:
            self.mousePressAltPos = [event.pos[0]/self.size[0],
                                         1 - event.pos[1]/self.size[1]]
            print(' Current df and dk: %1.4f, %1.4f' % (self.P[int(self.h/4), int(self.w/4), 2],
                  self.P[int(self.h/4), int(self.w/4), 3]), end="\r")
        else:
            print(event.modifiers)

    def on_mouse_move(self, event):
        if(self.mouseDown):
            if len(event.modifiers) == 0:
                # update brush coords here
                self.compute['brush'] = [event.pos[0]/self.size[0],
                                         1 - event.pos[1]/self.size[1]]
                self.compute['brushtype'] = event.button
            elif len(event.modifiers) == 1 and 'control' in event.modifiers:
                # update f and k values according to the x and y movements
                self.updateFK(event)
                print(' Current f and k: %1.4f, %1.4f' % (self.P[0, 0, 2],
                                                   self.P[0, 0, 3]), end="\r")
            elif len(event.modifiers) == 1 and 'shift' in event.modifiers:
                self.updateHillshading(event)
                print(' Current Hillshading Direction and altitude: %3.0f, %3.0f' % (self.hsdirection, self.hsaltitude), end="\r")
            elif len(event.modifiers) == 1 and 'Alt' in event.modifiers:
                self.modulateF(event)
                self.modulateK(event)
                print(' Current df and dk: %1.4f, %1.4f' % (self.P[int(self.h/4), int(self.w/4), 2],
                                                   self.P[int(self.h/4), int(self.w/4), 3]), end="\r")

    def on_mouse_release(self, event):
        self.mouseDown = False
        if len(event.modifiers) == 0:
            self.compute['brush'] = [0, 0]
            self.compute['brushtype'] = 0
        elif len(event.modifiers) == 1 and 'control' in event.modifiers:
            print('     New f and k: %1.4f, %1.4f' % (self.P[0, 0, 2], self.P[0, 0, 3]), end="\n")
        elif len(event.modifiers) == 1 and 'shift' in event.modifiers:
            print('     New Hillshading Direction and altitude: %3.0f, %3.0f' % (self.hsdirection, self.hsaltitude), end="\n")
        elif len(event.modifiers) == 1 and 'Alt' in event.modifiers:
            print('     New df and dk: %1.4f, %1.4f' % (self.P[int(self.h/4), int(self.w/4), 2],
                                               self.P[int(self.h/4), int(self.w/4), 3]), end="\n")

    def on_mouse_wheel(self, event):
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
            self.reinitializeGrid()
        elif event.text in self.speciesDictionnary.keys():
            self.switchSpecie(self.speciesDictionnary[event.text])
        elif event.text == '/':
            self.switchReagent()
        elif event.text in self.colormapDictionnary.keys():
            self.switchColormap(self.colormapDictionnary[event.text])
        elif event.text == '$':
            self.toggleHillshading()
        elif event.text == '*':
            self.toggleInterpolation()
        elif event.key and event.key.name == 'Up':
            self.increaseCycle()
            print('Number of cycles: %2.0f' % (1 + 2* self.cycle), end='\r')
        elif event.key and event.key.name == 'Down':
            self.decreaseCycle()
            print('Number of cycles: %2.0f' % (1 + 2* self.cycle), end='\r')
        elif event.key and event.key.name == 'Right':
            self.updatedd(0.01)
            print(' dd: %1.2f' % self.dd, end='\r')
        elif event.key and event.key.name == 'Left':
            self.updatedd(-0.01)
            print(' dd: %1.2f' % self.dd, end='\r')
        # print(event.key.name)

    def reinitializeGrid(self):
        print('Reinitialization of the grid.')
        self.UV = np.zeros((self.h, self.w, 4), dtype=np.float32)
        self.UV[:, :, 0:2] = setup_grid(self.h, self.w)
        self.UV += np.random.uniform(-0.02, 0.1, (self.h, self.w, 4))
        self.UV[:, :, 2] = self.UV[:, :, 0]
        self.UV[:, :, 3] = self.UV[:, :, 1]
        self.texture.set_data(self.UV)
        self.compute["texture"] = self.texture
        self.compute["texture"].interpolation = gl.GL_NEAREST
        self.compute["texture"].wrapping = gl.GL_REPEAT
        self.render["texture"] = self.compute["texture"]
        self.render["texture"].interpolation = gl.GL_LINEAR
        self.render["texture"].wrapping = gl.GL_REPEAT

    def switchSpecie(self, specie):
        self.specie = specie
        self.title = 'Gray-Scott Reaction-Diffusion: Pattern %s - GregVDS' % self.specie
        print('Pearson\'s Pattern %s' % self.specie)
        print(self.species[self.specie][4])
        print('        dU  dV  f     k \n        %s %s %s %s' % (self.species[self.specie][0],
                                                 self.species[self.specie][1],
                                                 self.species[self.specie][2],
                                                 self.species[self.specie][3]))
        self.P[:, :] = self.species[specie][0:4]
        self.params = gloo.texture.Texture2D(data=self.P, format=gl.GL_RGBA, internalformat='rgba32f')
        self.compute["params"] = self.params

    def switchReagent(self):
        if self.render["reagent"] == 1:
            self.render["reagent"] = 0
            print('Displaying V.')
        else:
            self.render["reagent"] = 1
            print('Displaying U.')

    def updateFK(self, event):
        fModAmount = event.pos[0]/self.size[0] - self.mousePressControlPos[0]
        kModAmount = 1 - event.pos[1]/self.size[1] - self.mousePressControlPos[1]
        f = self.P[0, 0, 2]
        k = self.P[0, 0, 3]
        self.P[:, :, 2] -= f
        self.P[:, :, 3] -= k
        self.P[:, :, 2] = np.clip(self.P[:, :, 2] + (f + 0.00075 * fModAmount), 0.0, 0.08)
        self.P[:, :, 3] = np.clip(self.P[:, :, 3] + (k + 0.001 * kModAmount), 0.03, 0.07)
        self.compute["params"] = self.P

    def modulateF(self, event):
        f = self.P[0, 0, 2]
        self.fModAmount += 0.00375 * (event.pos[0]/self.size[0] - self.mousePressAltPos[0])
        rows = self.h
        cols = self.w
        sins = np.sin(np.linspace(0.0, 2*np.pi, cols))
        # for each column
        for i in range(rows):
            self.P[i, :, 2] = np.clip(f + self.fModAmount*sins, 0.0, 0.08)
        self.params = gloo.texture.Texture2D(data=self.P, format=gl.GL_RGBA, internalformat='rgba32f')
        self.compute["params"] = self.params

    def modulateK(self, event):
        k = self.P[0, 0, 3]
        self.kModAmount += 0.005 * (1 - event.pos[1]/self.size[1] - self.mousePressAltPos[1])
        rows = self.h
        cols = self.w
        sins = np.sin(np.linspace(0.0, 2*np.pi, rows))
        # for each row
        for i in range(cols):
            self.P[:, i, 3] = np.clip(k + self.kModAmount*sins, 0.03, 0.07)
        self.params = gloo.texture.Texture2D(data=self.P, format=gl.GL_RGBA, internalformat='rgba32f')
        self.compute["params"] = self.params

    def updatedd(self, amount):
        self.dd = np.clip(self.dd + amount, .2, 1.3)
        self.compute['dd'] = np.clip(self.dd, .2, 1.3)

    def updatedt(self, amount):
        self.dt += amount
        self.compute['dt'] = self.dt

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

    def switchColormap(self, name):
        print('Using colormap %s.' % name)
        self.cm = get_colormap(name)
        self.render["cmap"] = self.cm.map(np.linspace(0.0, 1.0, 1024)).astype('float32')

    def toggleHillshading(self):
        if self.hsz > 0.0:
            self.lastHsz = self.hsz
            self.hsz = 0.0
            print('Hillshading off.')
        else:
            self.hsz = self.lastHsz
            print('Hillshading on.')
        self.render['hsz'] = self.hsz

    def updateHillshading(self, event):
        hsdir = 180.0/math.pi*math.atan2(
            (event.pos[0]/self.size[0] - 0.5)*2,
            (1-event.pos[1]/self.size[1] - 0.5)*2
        )
        self.hsdirection = 360 + hsdir if hsdir < 0.0 else hsdir
        # self.hsdirection -= 180
        self.hsaltitude = 90 - (180.0/math.pi*math.atan(
            math.sqrt(
                ((event.pos[0]/self.size[0] - 0.5)*2) *
                ((event.pos[0]/self.size[0] - 0.5)*2) +
                ((1-event.pos[1]/self.size[1] - 0.5)*2) *
                ((1-event.pos[1]/self.size[1] - 0.5)*2)
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

    def getHsDirAlt(self, lightDirection=315, lightAltitude=20):
        hsdir = 360.0 - lightDirection + 90.0
        if hsdir >= 360.0:
            hsdir -= 360.0
        hsdir *= math.pi / 180.0
        hsalt = (90 - lightAltitude) * math.pi / 180.0
        return (hsdir, hsalt)

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


if __name__ == '__main__':
    print(__doc__)

    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    plot_color_gradients("Custom colormaps", [Canvas.colormapDictionnary[each] for each in Canvas.colormapDictionnary.keys()])
    plt.show()

    c = Canvas(size=(1024, 1024), scale=0.5)
    c.measure_fps2(window=2)
    app.run()
