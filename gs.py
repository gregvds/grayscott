# -----------------------------------------------------------------------------
# Copyright (c) 2021 GREGVDS. All rights reserved.
#
# -----------------------------------------------------------------------------
import numpy as np
from numpy.random import rand
from vispy import app, gloo
from vispy.gloo import gl
import vispy.color as color
from shaders import vertex_shader, compute_fragment, render_fragment
from systems import pearson


def import_pearsons_types():
    species = {}
    type = None
    scale = 4.0 + 4.0/np.sqrt(2)
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
    grid[:, :, 1] = 0

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
                            (1+rand(to_row-from_row, to_col-from_col, 2) / 10)

                            * grid[from_row:to_row, from_col:to_col, :]
    )

    return grid


class Canvas(app.Canvas):
    cwidth, cheight = 300, 300
    dt = 3.5    # timestep; original value = 1.5                        lambda_left: 5
    dd = 0.7072    # Distance param for diffusion; original value = 1.5  lambda_left: 0.765
    w, h = cwidth, cheight

    # get all the Pearsons types
    species = import_pearsons_types()

    # definition of colormap
    colorList = ['#f5f3f1', '#f6b26b', '#c48229', '#080704', '#0c3b66', '#089ff1', '#f2f9fd']
    colorIndexes = [0.0, 0.45, 0.475, 0.5, 0.55, 0.6, 1.0]
    cm = color.Colormap(colorList, controls=colorIndexes)
    # cm = color.get_colormap('grays')

    # choice of Pearson type
    # working so far:
    # *xi_left
    # lambda_left
    # kappa_left ?
    # iota ?
    # theta_right ?
    # theta_left
    # eta
    # epsilon_right
    # epsilon_left
    # delta_right
    # delta_left
    # beta_left ?
    # alpha_right
    # alpha_left
    # gamma_left ?
    # gamma_right
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
    UV += np.random.uniform(-0.02, 0.1, (h, w, 4))
    UV[:, :, 2] = UV[:, :, 0]
    UV[:, :, 3] = UV[:, :, 1]

    # brush to add reagent v
    brush = np.zeros((1, 1, 2), dtype=np.float32)
    mouseDown = False

    pingpong = 1

    # program of computation of Gray-Scott reaction-diffusion sim
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

    # program of rendering u chemical concentration
    render = gloo.Program(vertex_shader, render_fragment, count=4)
    render["texture"] = compute["texture"]
    render["texture"].interpolation = gl.GL_LINEAR
    render["texture"].wrapping = gl.GL_REPEAT
    render["position"] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
    render["texcoord"] = [(0, 0), (0, 1), (1, 0), (1, 1)]
    render["cmap"] = cm.map(np.linspace(0, 1, 256)).astype('float32')
    render["uscales"] = uScales
    render["vscales"] = vScales
    render["reagent"] = 1
    render['pingpong'] = pingpong

    framebuffer = gloo.FrameBuffer(color=compute["texture"],
                                   depth=gloo.RenderBuffer((w, h), format='depth'))

    def __init__(self):
        super().__init__(size=(1024, 1024), title='Gray-Scott Reaction-Diffusion',
                         keys='interactive',
                         show=True)
        print(self.size)

    def on_draw(self, event):
        # toggling between r,g and b,a of texture
        self.pingpong = abs(1 - self.pingpong)
        self.compute["pingpong"] = self.pingpong
        self.render["pingpong"] = self.pingpong

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

        self.update()

    def on_mouse_press(self, event):
        # print('mouse press')
        print('Mouse position: %s, %s' % (event.pos[0], event.pos[1]))
        self.mouseDown = True
        self.compute['brush'] = [event.pos[0]/self.size[0], 1- event.pos[1]/self.size[1]]

    def on_mouse_release(self, event):
        pass
        # print('mouse release')
        self.mouseDown = False
        self.compute['brush'] = [0, 0]

    def on_mouse_move(self, event):
        pass
        if(self.mouseDown):
            # update brush coords here
            self.compute['brush'] = [event.pos[0]/self.size[0], 1- event.pos[1]/self.size[1]]


if __name__ == '__main__':
    for each in color.get_colormaps().keys():
        print(each)
    c = Canvas()
    app.run()
