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
    For more use help, call python3 gs3D.py -h
"""

################################################################################
from math import tan, sin, cos, pi, atan, asin, sqrt

import argparse
import textwrap

import numpy as np

from vispy import app, gloo
from vispy.geometry import create_plane
from vispy.gloo import gl, Program, VertexBuffer, IndexBuffer, FrameBuffer
from vispy.io import read_png, load_data_file
from vispy.util.transforms import perspective, translate, rotate

from gs_lib import (get_colormap, createColormaps, import_pearsons_types, setup_grid, createLightBox)

from shaders import compute_vertex
from shaders import compute_fragment_2 as compute_fragment
from shaders import render_3D_vertex
from shaders import render_3D_fragment
from shaders import shadow_vertex
from shaders import shadow_fragment


# ? Use of this ?
# gl.use_gl('gl+')

################################################################################


################################################################################


class Camera():
    """
    Simple class that represents a camera, and holds all its characteristics.
    """
    def __init__(self,
               eye=[0,0,1],
               target=[0,0,0],
               up=[0,0,1],
               fov=60.0,
               aspect=1.0,
               near=1.0,
               far=100.0):
        self.eye = eye
        self.target = target
        self.up = up
        self.view = self.lookAt(self.eye, self.target, up=self.up)
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far
        self.projection = perspective(fov, aspect, near, far)

        self.azimuth = 0.0
        self.elevation = 0.0
        self.distance = np.linalg.norm(np.subtract(self.target, self.eye))
        self.distanceMin = 0.7
        self.distanceMax = 5.0
        self.sensitivity = 5.0

    def setEye(self, eye):
        """Sets the coordinates of the camera, computes its view matrix and return it"""
        self.eye = eye
        self.view = self.lookAt(self.eye, self.target, up=self.up)
        return self.view

    def setTarget(self, target):
        """Sets the target at which the camera is pointing, compute its view matrix and returns it"""
        self.target = target
        self.view = self.lookAt(self.eye, self.target, up=self.up)
        return self.view

    def setProjection(self, fov=None, aspect=None, near=None, far=None):
        """Compute the projection matrix of the camera and returns it"""
        self.fov = fov or self.fov
        self.aspect = aspect or self.aspect
        self.near = near or self.near
        self.far = far or self.far
        self.projection = perspective(self.fov, self.aspect, self.near, self.far)
        return self.projection

    def move(self, azimuth=None, elevation=None, distance=None, target=[0, 0, 0]):
        """Moves the camera according to the azimuth, elevation, distance and
           target received, compute its view matrix and returns it
        """
        self.azimuth = azimuth or self.azimuth
        self.elevation = elevation or self.elevation
        self.distance = distance or self.distance
        self.distance = min(max(self.distance, self.distanceMin), self.distanceMax)
        x = self.distance * sin(self.elevation) * sin(self.azimuth)
        y = self.distance * sin(-self.elevation) * cos(self.azimuth)
        z = self.distance * cos(self.elevation)
        self.eye = [x, y, z]
        self.target = target
        self.view = self.lookAt(self.eye, self.target, up=self.up)
        return self.view

    def zoomOn(self, objectWidth=sqrt(2.0), margin=0.02):
        """Compute a projection matrix which frustum just covers the width of
           the object given. This is used in all directions. This width should
           be the diameter of the sphere containing the object, centered on the
           center of the object. it returns the projection matrix.
        """
        self.distance = np.linalg.norm(np.subtract(self.target, self.eye))
        radius = objectWidth/2.0*(1.0+margin)
        fov = 2 * asin(radius / self.distance) * 180. / pi
        near = self.distance - radius
        far = self.distance + radius
        return self.setProjection(fov, self.aspect, near, far)

    def lookAt(self, eye, target, up=[0, 0, 1]):
        """Computes matrix to put eye looking at target point."""
        eye = np.asarray(eye).astype(np.float32)
        target = np.asarray(target).astype(np.float32)
        up = np.asarray(up).astype(np.float32)

        vforward = eye - target
        vforward /= np.linalg.norm(vforward)
        vright = np.cross(up, vforward)
        vright /= np.linalg.norm(vright)
        vup = np.cross(vforward, vright)
        view = np.r_[vright, -np.dot(vright, eye),
                     vup, -np.dot(vup, eye),
                     vforward, -np.dot(vforward, eye),
                     [0, 0, 0, 1]].reshape(4, 4, order='F')
        return view


class Canvas(app.Canvas):

    colormapDictionnary = {
        '&': 'Boston',
        'é': 'malmo_r',
        '"': 'uppsala_r',
        '\'': 'oslo',
        '(': 'Lochinver_r',
        '§': 'Rejkjavik_r',
        'è': 'antidetroit',
        '!': 'osmort',
        'ç': 'irkoutsk_r',
        'à': 'krasnoiarsk_r'
    }

    colormapDictionnaryShifted = {
        '1': 'Boston_r',
        '2': 'malmo',
        '3': 'uppsala',
        '4': 'oslo_r',
        '5': 'Lochinver',
        '6': 'Rejkjavik',
        '7': 'detroit',
        '8': 'tromso',
        '9': 'irkoutsk',
        '0': 'krasnoiarsk'
    }

    speciesDictionnary = {
        'a': 'alpha_left',
        'b': 'beta_left',
        'd': 'delta_left',
        'e': 'epsilon_left',
        'g': 'gamma_left',
        'h': 'eta',
        'i': 'iota',
        'k': 'kappa_left',
        'l': 'lambda_left',
        'm': 'mu_left',
        'n': 'nu_left',
        'p': 'pi_left',
        't': 'theta_left',
        'x': 'xi_left',
        'z': 'zeta_left'
    }

    speciesDictionnaryShifted = {
        'A': 'alpha_right',
        'B': 'beta_right',
        'D': 'delta_right',
        'E': 'epsilon_right',
        'G': 'gamma_right',
        'K': 'kappa_right',
        'L': 'lambda_right',
        'M': 'mu_right',
        'T': 'theta_right',
        'Z': 'zeta_right'
    }

    species = import_pearsons_types()

    createColormaps()


    def __init__(self,
                 size=(1024, 1024),
                 modelSize=(512,512),
                 specie='alpha_left',
                 cmap='irkoutsk'):
        app.Canvas.__init__(self,
                            size=size,
                            title='3D Gray-Scott Reaction-Diffusion: - GregVDS',
                            keys='interactive')

        # General constants definition
        # --------------------------------------
        (self.w, self.h)                   = modelSize
        (self.fMin, self.fMax)             = (0.0, 0.08)
        (self.kMin, self.kMax)             = (0.03, 0.07)
        (self.fModAmount, self.kModAmount) = (0, 0)
        # (self.ddMin, self.ddMax)           = (0.2, 1.3)

        # Build plane data
        # --------------------------------------
        # Vertices contains
        # Position being vec3
        # texcoord being vec2
        # normal being vec3
        # color being vec4
        V, F, outline = create_plane(width_segments=self.w, height_segments=self.h, direction='+z')
        vertices = VertexBuffer(V)
        self.faces = IndexBuffer(F)

        # Build texture data
        # --------------------------------------
        # the texture contains 4 layers r, g, b, a
        # containing U and V concentrations twice
        # and are used through pingpong alternatively
        # each GPU computation/rendering cycle
        self.initializeGrid()
        self.pingpong = 1
        # cycle of computation per frame
        self.cycle = 0

        # Build view, model, projection
        # --------------------------------------
        self.model = np.eye(4, dtype=np.float32)

        self.viewCoordinates = [0, 0, -2.5]

        self.camera = Camera(eye=[0, 0, 2.5],
                             target=[0,0,0],
                             up=[0,1,0],
                             fov=24.0,
                             aspect=self.size[0] / float(self.size[1]),
                             near=0.1,
                             far=20.0)
        self.view = self.camera.view

        # light Parameters: direction, shininess exponant, attenuation parameters,
        # ambientLight intensity, ambientColor, diffuseColor and specularColor
        # --------------------------------------
        self.lightCoordinates = np.array([-.4, .4, -2.1])
        self.lightIntensity = (1., 1., 1.)
        self.c1 = .8
        self.c2 = .8
        self.c3 = 0.02
        self.ambientIntensity = 0.5
        self.ambientColor = np.array((1., 1., 1., 1))
        self.diffuseColor = np.array((1., 1., .9, 1.))
        self.specularColor = np.array((1., 1., .95, 1.))
        self.shininess = 91.0

        # Build view, projection for shadowCam
        # --------------------------------------
        self.shadowCamAt = [0, 0, 0]
        self.updateLightCam()

        # Currently, the shadowmap is a simple image for I cannot set properly
        # the FrameBuffer depth part (RenderBuffer)...
        # Finally, this is useful to pass moments1 and 2 for VSF mapping :-)
        self.shadowMapSize = 2048
        self.shadowGrid = np.ones((self.shadowMapSize, self.shadowMapSize, 4), dtype=np.float32) * .1
        self.shadowTexture = gloo.texture.Texture2D(data=self.shadowGrid, format=gl.GL_RGBA, internalformat='rgba32f')
        # self.shadowGrid = np.ones((self.shadowMapSize, self.shadowMapSize), dtype=np.float32) * .1
        # self.shadowTexture = gloo.texture.Texture2D(data=self.shadowGrid, format=gl.GL_LUMINANCE, internalformat='r32f')

        self.displaySwitch = 0

        # Build a lightbox for specular Environment
        # --------------------------------------
        self.lightBoxTexture = np.zeros((6, 1024, 1024, 3), dtype=np.float32)
        # self.lightBoxTexture[2] = read_png(load_data_file("skybox/sky-left.png"))/255. #DOWN
        # self.lightBoxTexture[3] = read_png(load_data_file("skybox/sky-right.png"))/255. #UP
        # self.lightBoxTexture[0] = read_png(load_data_file("skybox/sky-front.png"))/255. #LEFT
        # self.lightBoxTexture[1] = read_png(load_data_file("skybox/sky-back.png"))/255. #RIGHT
        # self.lightBoxTexture[4] = read_png(load_data_file("skybox/sky-up.png"))/255. #BACK
        # self.lightBoxTexture[5] = read_png(load_data_file("skybox/sky-down.png"))/255. #FRONT

        self.lightBoxTexture[2] = read_png(load_data_file("skybox/sky-down.png"))/255. #DOWN
        self.lightBoxTexture[3] = np.rot90(read_png(load_data_file("skybox/sky-up.png"))/255., 3) #UP
        self.lightBoxTexture[0] = read_png(load_data_file("skybox/sky-left.png"))/255. #LEFT
        self.lightBoxTexture[1] = np.rot90(read_png(load_data_file("skybox/sky-right.png"))/255., 2) #RIGHT
        self.lightBoxTexture[4] = np.rot90(read_png(load_data_file("skybox/sky-back.png"))/255., 3) #BACK
        self.lightBoxTexture[5] = np.rot90(read_png(load_data_file("skybox/sky-front.png"))/255., 1) #FRONT

        # DEBUG, toggles to switch on and off different parts of lighting
        # --------------------------------------
        self.ambient     = True
        self.attenuation = True
        self.diffuse     = True
        self.specular    = True
        self.shadow      = 3
        self.lightBox    = True

        # Colormaps related variables
        # --------------------------------------
        self.cmapName = cmap

        # Pearson's patterns related variables
        # definition of parameters for du, dv, f, k
        # --------------------------------------
        self.specie = specie
        self.setSpecie(specie=self.specie)

        # Mouse interactions parameters
        # --------------------------------------
        self.pressed = False
        self.brush = [0, 0]
        self.brushType = 0

        # ? better computation ?
        # --------------------------------------
        gl.GL_FRAGMENT_PRECISION_HIGH = 1

        # Build compute program
        # --------------------------------------
        self.computeProgram = Program(compute_vertex, compute_fragment, count=4)
        self.computeProgram["position"] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        self.computeProgram["texcoord"] = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.computeProgram["texture"] = self.texture
        self.computeProgram["texture"].interpolation = gl.GL_NEAREST
        self.computeProgram["texture"].wrapping = gl.GL_REPEAT
        self.computeProgram["params"] = self.params
        self.computeProgram["dx"] = 1./self.w
        self.computeProgram["dy"] = 1./self.h
        self.computeProgram['pingpong'] = self.pingpong
        self.computeProgram['brush'] = self.brush
        self.computeProgram['brushtype'] = self.brushType

        # Build render program
        # --------------------------------------
        self.renderProgram = Program(render_3D_vertex, render_3D_fragment)
        self.renderProgram.bind(vertices)
        self.renderProgram["texture"] = self.computeProgram["texture"]
        self.renderProgram["texture"].interpolation = gl.GL_LINEAR
        self.renderProgram["texture"].wrapping = gl.GL_REPEAT
        self.renderProgram["scalingFactor"] = 30. * (self.w/512)
        self.renderProgram["dx"] = 1./self.w
        self.renderProgram["dy"] = 1./self.h
        self.renderProgram['pingpong'] = self.pingpong
        self.renderProgram["reagent"] = 1
        self.renderProgram["u_view"] = self.camera.view
        self.renderProgram["u_model"] = self.model
        self.renderProgram["shadowMap"] = self.shadowTexture
        self.renderProgram["shadowMap"].interpolation = gl.GL_LINEAR
        self.renderProgram["shadowMap"].wrapping = gl.GL_CLAMP_TO_EDGE
        self.renderProgram["u_Shadowmap_projection"] = self.shadowProjection
        self.renderProgram["u_Shadowmap_view"] = self.shadowView
        self.renderProgram["u_Tolerance_constant"] = 5e-03
        self.renderProgram["vsf_gate"] = 2.5e-05
        self.renderProgram["ambient"] = self.ambient
        self.renderProgram["attenuation"] = self.attenuation
        self.renderProgram["diffuse"] = self.diffuse
        self.renderProgram["specular"] = self.specular
        self.renderProgram["shadow"] = self.shadow
        self.renderProgram["lightBox"] = self.lightBox
        self.renderProgram["cubeMap"] = gloo.TextureCube(self.lightBoxTexture, interpolation='linear')
        self.renderProgram["u_light_position"] = self.lightCoordinates
        self.renderProgram["u_light_intensity"] = self.lightIntensity
        self.renderProgram["u_Ambient_color"] = self.ambientColor
        self.renderProgram["u_ambient_intensity"] = self.ambientIntensity
        self.renderProgram["u_diffuse_color"] = self.diffuseColor
        self.renderProgram["u_specular_color"] = self.specularColor
        self.renderProgram['u_Shininess'] = self.shininess
        self.renderProgram['c1'] = self.c1
        self.renderProgram['c2'] = self.c2
        self.renderProgram['c3'] = self.c3
        self.setColorMap(name=self.cmapName)

        # Build shadowmap render program
        # --------------------------------------
        self.shadowProgram = Program(shadow_vertex, shadow_fragment, count=self.w*self.h)
        self.shadowProgram.bind(vertices)
        self.shadowProgram["texture"] = self.computeProgram["texture"]
        self.shadowProgram["texture"].interpolation = gl.GL_LINEAR
        self.shadowProgram["texture"].wrapping = gl.GL_REPEAT
        self.shadowProgram['pingpong'] = self.pingpong
        self.shadowProgram["reagent"] = 1
        self.shadowProgram["scalingFactor"] = 30. * (self.w/512)
        self.shadowProgram["u_model"] = self.model
        self.shadowProgram["u_view"] = self.shadowView
        self.shadowProgram['u_projection'] = self.shadowProjection

        # Define a FrameBuffer to update model state in texture
        # --------------------------------------
        self.modelbuffer = FrameBuffer(color=self.computeProgram["texture"],
                                       depth=gloo.RenderBuffer((self.h, self.w), format='depth'))

        # Define a 'DepthBuffer' to render the shadowmap from the light
        # --------------------------------------
        self.shadowBuffer = FrameBuffer(color=self.renderProgram["shadowMap"],
                                       depth=gloo.RenderBuffer((self.shadowMapSize, self.shadowMapSize), format='depth'))

        # OpenGL initialization
        # --------------------------------------
        gloo.set_state(clear_color=(0.30, 0.30, 0.35, 1.00), depth_test=True,
                       line_width=0.75)

        self.activate_zoom()
        self.show()

    ############################################################################
    #

    def on_draw(self, event):
        # Render next model state into buffer
        with self.modelbuffer:
            gloo.set_viewport(0, 0, self.h, self.w)
            gloo.set_state(depth_test=False,
                           clear_color='black',
                           polygon_offset=(0, 0))
            self.computeProgram.draw('triangle_strip')
            # repeat model state computation several time to speed up slow patterns
            for cycle in range(self.cycle):
                self.pingpong = 1 - self.pingpong
                self.computeProgram["pingpong"] = self.pingpong
                self.computeProgram.draw('triangle_strip')
                self.pingpong = 1 - self.pingpong
                self.computeProgram["pingpong"] = self.pingpong
                self.computeProgram.draw('triangle_strip')

        # Render the shadowmap into buffer
        if self.shadow > 0:
            with self.shadowBuffer:
                gloo.set_viewport(0, 0, self.shadowMapSize, self.shadowMapSize)
                gloo.set_state(depth_test=True,
                               polygon_offset=(1, 1),
                               polygon_offset_fill=True)
                gloo.clear(color=True, depth=True)
                self.shadowProgram.draw('triangles', self.faces)
        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
        gloo.set_state(blend=False, depth_test=True,
                       clear_color=(0.30, 0.30, 0.35, 1.00),
                       blend_func=('src_alpha', 'one_minus_src_alpha'),
                       polygon_offset=(1, 1),
                       polygon_offset_fill=True)
        gloo.clear(color=True, depth=True)
        self.renderProgram.draw('triangles', self.faces)

        # exchange between rg and ba sets in texture
        self.pingpong = 1 - self.pingpong
        self.computeProgram["pingpong"] = self.pingpong
        self.renderProgram["pingpong"] = self.pingpong

        self.update()

    def on_resize(self, event):
        self.activate_zoom()

    def activate_zoom(self):
        gloo.set_viewport(0, 0, *self.physical_size)
        projection = self.camera.setProjection(aspect=self.size[0] / float(self.size[1]))
        if hasattr(self, 'renderProgram'):
            self.renderProgram['u_projection'] = projection
        if hasattr(self, 'coordinatesProgram'):
            self.coordinatesProgram['u_projection'] = projection

    ############################################################################
    # Mouse and keys interactions

    def on_mouse_wheel(self, event):
        # no shift modifier key
        # Move the camera AND light in z
        self.view = self.camera.move(distance=self.camera.distance - event.delta[1])
        if hasattr(self, 'renderProgram'):
            self.renderProgram["u_view"] = self.view

        self.lightCoordinates[2] += event.delta[1]/2
        self.lightCoordinates[2] = np.clip(self.lightCoordinates[2], self.viewCoordinates[2]+.1, -0.8)
        # shift modifier key
        # Move only the light in z
        self.lightCoordinates[2] += event.delta[0]/2
        self.lightCoordinates[2] = np.clip(self.lightCoordinates[2], self.viewCoordinates[2]+.1, -0.8)
        self.updateLight()

    def on_mouse_press(self, event):
        self.pressed = True
        (x, y) = event.pos
        (sx, sy) = self.size
        xpos = x/sx
        ypos = 1 - y/sy
        if len(event.modifiers) == 0:
            # self.mousePos = [xpos, ypos]
            self.mousePos = event.pos
        elif len(event.modifiers) == 1 and event.modifiers[0] == 'Shift':
            self.computeProgram['brush'] = [xpos, ypos]
            self.computeProgram['brushtype'] = event.button

    def on_mouse_release(self, event):
        self.pressed = False
        self.computeProgram['brush'] = [0, 0]
        self.computeProgram['brushtype'] = 0

    def on_mouse_move(self, event):
        if(self.pressed):
            (x, y) = event.pos
            (sx, sy) = self.size
            xpos = x/sx
            ypos = 1 - y/sy
            if len(event.modifiers) == 0:
                dazimuth = (event.pos[0] - self.mousePos[0]) * (2*pi) / self.size[0]
                delevation = (event.pos[1] - self.mousePos[1]) * (2*pi) / self.size[1]
                self.mousePos = event.pos
                self.view = self.camera.move(azimuth = self.camera.azimuth - dazimuth/self.camera.sensitivity,
                                             elevation=self.camera.elevation - delevation/self.camera.sensitivity)
                if hasattr(self, 'renderProgram'):
                    self.renderProgram["u_view"] = self.view
            elif len(event.modifiers) == 1 and event.modifiers[0] == 'Shift':
                # update brush coords here
                self.computeProgram['brush'] = [xpos, ypos]
                self.computeProgram['brushtype'] = event.button

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
            func(self, event, *args)

    ############################################################################
    # functions related to the Gray-Scott model parameters

    def initializeGrid(self, event=None):
        """
        Initialize the concentrations of U and V of the model accross a grid
        with a seed patch in its center.
        """
        print('Initialization of the grid.', end="\r")
        self.UV = np.zeros((self.h, self.w, 4), dtype=np.float32)
        self.UV[:, :, 0:2] = setup_grid(self.h, self.w)
        self.UV += np.random.uniform(-0.02, 0.1, (self.h, self.w, 4))
        self.UV[:, :, 2] = self.UV[:, :, 0]
        self.UV[:, :, 3] = self.UV[:, :, 1]
        if not hasattr(self, 'texture'):
            self.texture = gloo.texture.Texture2D(data=self.UV, format=gl.GL_RGBA, internalformat='rgba32f')
        else:
            self.texture.set_data(self.UV)
        if hasattr(self, 'computeProgram'):
            self.computeProgram["texture"] = self.texture
            self.computeProgram["texture"].interpolation = gl.GL_NEAREST
            self.computeProgram["texture"].wrapping = gl.GL_REPEAT
        if hasattr(self, 'renderProgram'):
            self.renderProgram["texture"] = self.computeProgram["texture"]
            self.renderProgram["texture"].interpolation = gl.GL_LINEAR
            self.renderProgram["texture"].wrapping = gl.GL_REPEAT

    def setSpecie(self, event=None, specie=''):
        """Set the feed, kill and diffusion rates for the choosen pattern"""
        if specie != '':
            self.specie = specie
            self.printPearsonPatternDescription()
            self.P = np.zeros((self.h, self.w, 4), dtype=np.float32)
            self.P[:, :] = Canvas.species[self.specie][0:4]
            self.updateComputeParams()

    def updateComputeParams(self):
        self.params = gloo.texture.Texture2D(data=self.P, format=gl.GL_RGBA, internalformat='rgba32f')
        if hasattr(self, 'computeProgram'):
            self.computeProgram["params"] = self.params

    def increaseCycle(self, event=None):
        """
        Increases number of cycles processed per frame shown.
        Each increase multiplies by two the number of added cycles
        """
        if self.cycle == 0:
            self.cycle = 1
        else:
            self.cycle *= 2
        if self.cycle > 64:
            self.cycle = 64
        print(' Number of cycles: %3.0f' % (1 + 2 * self.cycle), end='\r')

    def decreaseCycle(self, event=None):
        """
        Decreases number of cycles processed per frame shown.
        Each decrease divides by two the number of added cycles
        """
        self.cycle = int(self.cycle/2)
        if self.cycle < 1:
            self.cycle = 0
        print(' Number of cycles: %3.0f' % (1 + 2 * self.cycle), end='\r')

    ############################################################################
    # functions related to the Gray-Scott model appearances/representation

    def switchReagent(self, event=None):
        """Toggles between representing U or V concentrations."""
        self.renderProgram["reagent"] = 1 - self.renderProgram["reagent"]
        self.shadowProgram["reagent"] = 1 - self.shadowProgram["reagent"]
        reagents = ('U', 'V')
        print(' Displaying %s reagent concentration.' % reagents[int(self.renderProgram["reagent"])], end="\r")

    def setColorMap(self, event=None, name=''):
        """Set the colormap used to render the concentration"""
        if name != '':
            self.cmapName = name
            self.renderProgram["cmap"] = get_colormap(self.cmapName).map(np.linspace(0.0, 1.0, 1024)).astype('float32')
            print(' Using colormap %s.' % name, end="\r")

    SHADOW_TYPE = ("None                     ",
                   "Simple                   ",
                   "Percent 4 samplings      ",
                   "Variance shadow mapping  ")

    def modifyLightCharacteristic(self, event, lightType=None, modification=None):
        """
        Modifies one of the light parameters.
        Some like ambient, diffuse, specular are simply toggles on/off,
        Others increase/decrease a value, such as the shininess exponant
        """
        if lightType == 'ambient':
            self.ambient = not self.ambient
            self.renderProgram[lightType] = self.ambient
            print(' Ambient light: %s        ' % self.ambient, end="\r")
        elif lightType == 'diffuse':
            self.diffuse = not self.diffuse
            self.renderProgram[lightType] = self.diffuse
            print(' Diffuse light: %s        ' % self.diffuse, end="\r")
        elif lightType == 'specular':
            self.specular = not self.specular
            self.renderProgram[lightType] = self.specular
            print(' Specular light: %s       ' % self.specular, end="\r")
        elif lightType == 'shadow':
            self.shadow = (self.shadow + 1) % 4
            self.renderProgram[lightType] = self.shadow
            print(' Shadows: %s              ' % self.SHADOW_TYPE[self.shadow], end="\r")
        elif lightType == 'attenuation':
            self.attenuation = not self.attenuation
            self.renderProgram[lightType] = self.attenuation
            print(' Attenuation: %s          ' % self.attenuation, end="\r")
        elif lightType == 'shininess' and modification == '-':
            self.shininess *= sqrt(2)
            self.shininess = np.clip(self.shininess, 0.1, 8192)
            self.renderProgram['u_Shininess'] = self.shininess
            print(' Shininess exponant: %3.0f' % self.shininess, end="\r")
        elif lightType == 'shininess' and modification == '+':
            self.shininess /= sqrt(2)
            self.shininess = np.clip(self.shininess, 0.1, 8192)
            self.renderProgram['u_Shininess'] = self.shininess
            print(' Shininess exponant: %3.0f' % self.shininess, end="\r")
        elif lightType == "vsf_gate" and modification == '+':
            self.renderProgram['vsf_gate'] = self.renderProgram['vsf_gate'] * 2.0
            print("vsf_gate: %s" % self.renderProgram['vsf_gate'])
        elif lightType == "vsf_gate" and modification == '-':
            self.renderProgram['vsf_gate'] = self.renderProgram['vsf_gate'] / 2.0
            print("vsf_gate: %s" % self.renderProgram['vsf_gate'])
        elif lightType == "lightbox":
            self.lightBox = not self.lightBox
            self.renderProgram["lightBox"] = self.lightBox
            print("LightBox %s" % self.lightBox)

    ############################################################################
    # functions to manipulate orientations and positions of model, view, light

    def resetCamera(self, event=None):
        """
        Replaces the camera in its original position.
        """
        self.view = self.camera.setEye([0,0,2.5])
        if hasattr(self, 'renderProgram'):
            self.renderProgram["u_view"] = self.view

    def updateLight(self, event=None, coordinatesDelta=(0,0,0)):
        """
        Moves the light in x and y.
        """
        # light is able to move along x,y,z axis
        self.lightCoordinates[0] += coordinatesDelta[0]
        self.lightCoordinates[1] += coordinatesDelta[1]
        self.lightCoordinates[2] += coordinatesDelta[2]
        self.lightCoordinates[0] = np.clip(self.lightCoordinates[0], -1, 1)
        self.lightCoordinates[1] = np.clip(self.lightCoordinates[1], -1, 1)
        if hasattr(self, 'renderProgram'):
            self.renderProgram["u_light_position"] = self.lightCoordinates
        self.updateLightCam()

    def updateLightCam(self):
        # lightcam is placed at the light coordinates
        self.shadowViewCoordinates = self.lightCoordinates
        self.shadowView = translate((-self.shadowViewCoordinates[0],
                                     -self.shadowViewCoordinates[1],
                                     self.shadowViewCoordinates[2]))

        # point camera at center of the model, and compute projection parameters
        self.shadowView, (self.shadowCamFoV, self.shadowCamNear, self.shadowCamFar) = self.rotateShadowView(self.shadowView,
                                                    self.shadowViewCoordinates,
                                                    self.shadowCamAt,
                                                    self.viewCoordinates)
        self.shadowProjection = perspective(self.shadowCamFoV,
                                            self.size[0] / float(self.size[1]),
                                            self.shadowCamNear,
                                            self.shadowCamFar)
        """
        # This would be much more clean, but currently does not work properly...
        self.shadowView = lookAt(self.shadowViewCoordinates, self.shadowCamAt)
        self.shadowProjection = zoomOn(self.shadowViewCoordinates, self.shadowCamAt)
        """

        if hasattr(self, 'shadowProgram'):
            self.shadowProgram["u_view"] = self.shadowView
            self.shadowProgram["u_projection"] = self.shadowProjection
        if hasattr(self, 'renderProgram'):
            self.renderProgram["u_Shadowmap_view"] = self.shadowView
            self.renderProgram["u_Shadowmap_projection"] = self.shadowProjection

    def rotateShadowView(self, shadowView, eye, at, cam):
        azimuth = 180. / pi * atan((eye[0]-at[0])/(eye[2]-at[2]))
        declination = -180. / pi * atan((eye[1]-at[1])/(eye[2]-at[2]))
        azRotationMatrix = rotate(azimuth, (0, 1, 0))
        deRotationMatrix = rotate(declination, (1, 0, 0))
        rotatedView = np.matmul(np.matmul(shadowView, deRotationMatrix), azRotationMatrix)
        # Attempt at computing a fov adequate to encompass the model
        # The object being a simple square plane of 1 x 1, but this one being
        # orientable, let's just consider a sphere with radius = half of the diagonal
        # of the square, with 2% more
        # one could also modulate the radius following the closeness of the main camera?
        length = np.linalg.norm(np.subtract(at, eye))
        radius = 1.0 / sqrt(2) * 1.02
        radiusForFOV = radius * np.clip((cam[2] / -1.25), 0.0, 1.0)
        radiusForNearFar = radius * np.clip((cam[2] / -0.9), 0.0, 1.0)
        fov = 2 * asin(radius / length) * 180. / pi
        near = length - radius
        far = length + radius
        return rotatedView, (fov, near, far)

    ############################################################################
    # Debug/utilities functions

    def switchDisplay(self, event=None):
        self.displaySwitch = (self.displaySwitch + 1) % 3

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
        (' ', ()): (initializeGrid, ()),
        ('/', ('Shift',)): (switchReagent, ()),
        ('P', ('Control',)): (increaseCycle, ()),
        ('O', ('Control',)): (decreaseCycle, ()),
        ('Up', ()): (updateLight, ((0, 0.02, 0),)),
        ('Down', ()): (updateLight, ((0, -0.02, 0),)),
        ('Left', ()): (updateLight, ((-0.02, 0, 0),)),
        ('Right', ()): (updateLight, ((0.02, 0, 0),)),
        ('@', ()): (resetCamera, ()),
        (',', ('Control',)): (modifyLightCharacteristic, ('ambient',)),
        (';', ('Control',)): (modifyLightCharacteristic, ('diffuse',)),
        (':', ('Control',)): (modifyLightCharacteristic, ('specular',)),
        ('=', ('Control',)): (modifyLightCharacteristic, ('shadow',)),
        ('N', ('Control',)): (modifyLightCharacteristic, ('attenuation',)),
        ('L', ('Control',)): (modifyLightCharacteristic, ('shininess', '-')),
        ('M', ('Control',)): (modifyLightCharacteristic, ('shininess', '+')),
        ('J', ('Control',)): (modifyLightCharacteristic, ('vsf_gate', '-')),
        ('K', ('Control',)): (modifyLightCharacteristic, ('vsf_gate', '+')),
        ('I', ('Control',)): (modifyLightCharacteristic, ('lightbox',))
    }
    for key in colormapDictionnary.keys():
        keyactionDictionnary[(key, ())] = (setColorMap, (colormapDictionnary[key],))
    for key in colormapDictionnaryShifted.keys():
        keyactionDictionnary[(key, ('Shift',))] = (setColorMap, (colormapDictionnaryShifted[key],))
    for key in speciesDictionnary.keys():
        keyactionDictionnary[(key, ())] = (setSpecie, (speciesDictionnary[key],))
    for key in speciesDictionnaryShifted.keys():
        keyactionDictionnary[(key, ('Shift',))] = (setSpecie, (speciesDictionnaryShifted[key],))

    ############################################################################
    # Output functions

    def printPearsonPatternDescription(self):
        self.title2 = '3D Gray-Scott Reaction-Diffusion: Pattern %s - GregVDS' % self.specie
        specie = Canvas.species[self.specie]
        print('Pearson\'s Pattern %s' % self.specie)
        print(specie[4])
        print('        dU  dV  f     k \n        %s %s %s %s' % (specie[0],
                                                                 specie[1],
                                                                 specie[2],
                                                                 specie[3]))

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
def fun(x):
    c.title = c.title2 +' - FPS: %0.1f' % x


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
                        choices=Canvas.speciesDictionnary.values(),
                        default="lambda_left",
                        help="Pearson\' pattern")
    parser.add_argument("-c",
                        "--Colormap",
                        choices=Canvas.colormapDictionnary.values(),
                        default="irkoutsk",
                        help="Colormap used")

    args = parser.parse_args()

    c = Canvas(modelSize=(args.Size, args.Size),
               size=(args.Window, args.Window),
               specie=args.Pattern,
               cmap=args.Colormap)
    c.measure_fps2(callback=fun)
    app.run()
