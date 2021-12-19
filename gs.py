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
import numpy as np
from numpy.random import rand

import vispy.color as color
from vispy import app, gloo
from vispy.gloo import gl

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

from shaders import vertex_shader, compute_fragment, render_fragment
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
    cm.register_cmap(name=name, cmap=newcmp)

    cmapNameList.reverse()
    proportions.reverse()

    for cmapName, proportion in zip(cmapNameList, proportions):
        newColors_r.append(cm.get_cmap(invertCmapName(
            cmapName), proportion)(np.linspace(0, 1, proportion)))
    newcmp_r = ListedColormap(np.vstack(newColors_r),
                              name=invertCmapName(name))
    cm.register_cmap(name=invertCmapName(name), cmap=newcmp_r)
    return newcmp


def import_pearsons_types(scale = 1.0):
    species = {}
    type = None
    # scale = 4.0 + 4.0/np.sqrt(2)
    for each in pearson.keys():
        type = pearson[each]
        species[each] = [type['factors']['Da']/scale,
                         type['factors']['Db']/scale,
                         type['factors']['f'],
                         type['factors']['k'],
                         type['scaling']['a']['icl'],
                         type['scaling']['a']['im'],
                         type['scaling']['a']['icu'],
                         0,0,
                         type['scaling']['b']['icl'],
                         type['scaling']['b']['im'],
                         type['scaling']['b']['icu'],
                         0,0]
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


class Canvas(app.Canvas): # originally app.Canvas
    cwidth, cheight = 512, 512
    dt = 1.0              # 3.5 timestep; original value = 1.5                        lambda_left: 5
    dd = 1.0              # 0.7072 Distance param for diffusion; original value = 1.5    lambda_left: 0.765
    w, h = cwidth, cheight
    fps = 60

    # get all the Pearsons types
    diffusionScale = 4.0 + 4.0/np.sqrt(2)
    species = import_pearsons_types(scale = 1.0)
    speciesDictionnary = {
        'a': 'alpha_right',
        'b': 'beta_left',
        'd': 'delta_left',
        'e': 'eta',
        'g': 'gamma_right',
        'i': 'iota',
        'k': 'kappa_left',
        'l': 'lambda_left',
        'p': 'pi_left',
        'x': '*xi_left'
    }

    # definition of some custom colormaps
    bostonCmap = createAndRegisterCmap(
        'Boston', [
            'bone_r',
            'gray',
            'pink_r',
            'YlOrBr_r'],
        proportions=[960, 160, 160, 960])
    seattleCmap = createAndRegisterCmap(
        'seattle', [
            'bone_r',
            'Blues_r',
            'bone_r',
            'gray',
            'pink_r',
            'YlOrBr_r'],
        proportions=[160, 480, 3200, 640, 640, 3840])
    colormapDictionnary = {
        '1': 'Boston_r',
        '&': 'Boston',
        '2': 'seattle',
        'é': 'seattle_r',
        '3': 'twilight',
        '"': 'twilight_r',
        '4': 'magma',
        '\'': 'magma_r',
        '5': 'bone',
        '(': 'bone_r',
        '6': 'YlOrBr',
        '§': 'YlOrBr_r'
    }

    cm = get_colormap('Boston_r')
    specie = 'alpha_right'

    # parameters for du, dv, f, k
    P = np.zeros((h, w, 4), dtype=np.float32)
    P[:, :] = species[specie][0:4]

    # parameters for scaling u and v rendering
    uScales = np.zeros((h, w, 4), dtype=np.float32)
    vScales = np.zeros((h, w, 4), dtype=np.float32)
    uScales[:, :] = species[specie][4:8]
    vScales[:, :] = species[specie][8:12]

    # u, v grid initialization, with central random patch
    UV = np.zeros((h, w, 4), dtype=np.float32)
    UV[:, :, 0:2] = setup_grid(h, w)
    UV += np.random.uniform(0.0, 0.01, (h, w, 4))    # Matbe no more useful since setup_grid add some randomness already
    UV[:, :, 2] = UV[:, :, 0]
    UV[:, :, 3] = UV[:, :, 1]

    # brush to add reagent v
    brush = np.zeros((1, 1, 2), dtype=np.float32)
    mouseDown = False

    mousePressControlPos = [0.0, 0.0]

    pingpong = 1

    # program for computation of Gray-Scott reaction-diffusion sim
    compute = gloo.Program(vertex_shader, compute_fragment, count=4)
    compute["params"] = P
    compute["texture"] = UV
    compute["texture"].interpolation = gl.GL_NEAREST
    compute["texture"].wrapping = gl.GL_REPEAT
    compute["position"] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
    compute["texcoord"] = [(0, 0), (0, 1), (1, 0), (1, 1)]
    compute['dt'] = dt
    compute['dx'] = 1.0 / w
    compute['dy'] = 1.0 / h
    compute['dd'] = dd
    compute['pingpong'] = pingpong
    compute['brush'] = brush

    # program for rendering u or v reagent concentration
    render = gloo.Program(vertex_shader, render_fragment, count=4)
    render["texture"] = compute["texture"]
    render["texture"].interpolation = gl.GL_LINEAR
    render["texture"].wrapping = gl.GL_REPEAT
    render["position"] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
    render["texcoord"] = [(0, 0), (0, 1), (1, 0), (1, 1)]
    render["cmap"] = cm.map(np.linspace(0.0, 1.0, 1024)).astype('float32')
    render["uscales"] = uScales
    render["vscales"] = vScales
    render["reagent"] = 1
    render['pingpong'] = pingpong

    # framebuffer = gloo.FrameBuffer(color=compute["texture"],
    #                                depth=gloo.RenderBuffer((w, h), format='depth'))
    framebuffer = gloo.FrameBuffer(color=compute["texture"],
                                   depth=gloo.RenderBuffer((w, h)))

    def __init__(self):
        super().__init__(size=(1024, 1024),
                         title='Gray-Scott Reaction-Diffusion',
                         keys='interactive',
                         # show=True,
                         resizable=False,
                         dpi=221)
        self.show()

    def on_draw(self, event):

        gl.glDisable(gl.GL_BLEND)

        # holding rendering output ?
        self.framebuffer.activate()
        gl.glViewport(0, 0, self.cwidth, self.cheight)
        self.compute.draw(gl.GL_TRIANGLE_STRIP)

        # releasing rendering output
        self.framebuffer.deactivate()
        gloo.clear(color=True, depth=True)
        gl.glViewport(0, 0, self.physical_size[0], self.physical_size[1])
        self.render.draw(gl.GL_TRIANGLE_STRIP)

        # toggling between r,g and b,a of texture
        self.pingpong = abs(1 - self.pingpong)
        self.compute["pingpong"] = self.pingpong
        self.render["pingpong"] = self.pingpong

        self.update()

    def on_mouse_press(self, event):
        self.mouseDown = True
        if len(event.modifiers) == 0:
            self.compute['brush'] = [event.pos[0]/self.size[0],
                                     1 - event.pos[1]/self.size[1]]
        elif 'control' in event.modifiers:
            self.mousePressControlPos = [event.pos[0]/self.size[0],
                                         1 - event.pos[1]/self.size[1]]
            print('Current f and k: %s, %s' % (self.P[0, 0, 2], self.P[0, 0, 3]))

    def on_mouse_move(self, event):
        if(self.mouseDown):
            if len(event.modifiers) == 0:
                # update brush coords here
                self.compute['brush'] = [event.pos[0]/self.size[0],
                                         1 - event.pos[1]/self.size[1]]
            elif 'control' in event.modifiers:
                # update f and k values according to the x and y movements
                fModAmount = event.pos[0]/self.size[0] - self.mousePressControlPos[0]
                kModAmount = 1 - event.pos[1]/self.size[1] - self.mousePressControlPos[1]
                f = self.P[0, 0, 2]
                k = self.P[0, 0, 3]
                self.P[:, :, 2] = np.clip(f + 0.0015 * fModAmount, 0.0, 0.08)
                self.P[:, :, 3] = np.clip(k + 0.002 * kModAmount, 0.03, 0.07)
                self.compute["params"] = self.P

    def on_mouse_release(self, event):
        self.mouseDown = False
        if len(event.modifiers) == 0:
            self.compute['brush'] = [0, 0]
        elif 'control' in event.modifiers:
            print('New f and k: %s, %s' % (self.P[0, 0, 2], self.P[0, 0, 3]))

    def on_key_press(self, event):
        if event.text == ' ':
            self.reinitializeGrid()
        if event.text in self.speciesDictionnary.keys():
            self.switchSpecie(self.speciesDictionnary[event.text])
        if event.text == '*':
            self.switchReagent()
        if event.text in self.colormapDictionnary.keys():
            self.switchColormap(self.colormapDictionnary[event.text])

    def reinitializeGrid(self):
        print('Reinitialization of the grid with')
        print('Pearson\'s Pattern %s' % self.specie)
        print('dU, dV, f, k: %s, %s, %s, %s.' % (self.species[self.specie][0],
                                                 self.species[self.specie][1],
                                                 self.species[self.specie][2],
                                                 self.species[self.specie][3]))
        self.P = np.zeros((self.h, self.w, 4), dtype=np.float32)
        self.P[:, :] = self.species[self.specie][0:4]
        self.UV = np.zeros((self.h, self.w, 4), dtype=np.float32)
        self.UV[:, :, 0:2] = setup_grid(self.h, self.w)
        self.UV += np.random.uniform(-0.02, 0.1, (self.h, self.w, 4))
        self.UV[:, :, 2] = self.UV[:, :, 0]
        self.UV[:, :, 3] = self.UV[:, :, 1]
        self.compute["params"] = self.P
        self.compute["texture"] = self.UV
        self.compute["texture"].interpolation = gl.GL_NEAREST
        self.compute["texture"].wrapping = gl.GL_REPEAT
        self.render["texture"] = self.compute["texture"]
        self.render["texture"].interpolation = gl.GL_LINEAR
        self.render["texture"].wrapping = gl.GL_REPEAT

    def switchSpecie(self, specie):
        self.specie = specie
        print('Pearson\'s Pattern %s' % self.specie)
        print('dU, dV, f, k: %s, %s, %s, %s.' % (self.species[self.specie][0],
                                                 self.species[self.specie][1],
                                                 self.species[self.specie][2],
                                                 self.species[self.specie][3]))
        self.P[:, :] = self.species[specie][0:4]
        self.uScales[:, :] = self.species[specie][4:8]
        self.vScales[:, :] = self.species[specie][8:12]
        self.compute["params"] = self.P
        self.render["uscales"] = self.uScales
        self.render["vscales"] = self.vScales

    def switchReagent(self):
        if self.render["reagent"] == 1:
            self.render["reagent"] = 0
            print('Displaying V.')
        else:
            self.render["reagent"] = 1
            print('Displaying U.')

    def switchColormap(self, name):
        print('Using colormap %s.' % name)
        self.cm = get_colormap(name)
        self.render["cmap"] = self.cm.map(np.linspace(0.0, 1.0, 1024)).astype('float32')

################################################################################


if __name__ == '__main__':
    print('Gray-Scott reaction-diffusion model')
    print('-----------------------------------')
    print('\nPearson\'s pattern can be switched with keys:')
    print('such as a, b, d, g, k, i, e, p, x, replacing greek letters.')
    print('Several colormaps are available via 1 - 6, shifted for reversed version.')
    print('Mouse left click in the grid refills reagent v at 0.5.')
    print('Ctrl + Mouse click and drag to modify feed and kill rates')
    print('* switch presentation between u and v.')
    print('Spacebar reseeds the grid.')
    c = Canvas()
    # c.measure_fps(window=1, callback='%1.1f FPS')
    app.run()
