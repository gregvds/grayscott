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
from math import pi, sin, cos, tan, asin, sqrt, atan2
import argparse
import textwrap

import numpy as np

from vispy import app, gloo
from vispy.geometry import create_plane
from vispy.gloo import gl, Program, VertexBuffer, IndexBuffer, FrameBuffer
from vispy.io import read_png, load_data_file
from vispy.util.transforms import perspective

from gs_lib import (get_colormap, createColormaps, import_pearsons_types, setup_grid)

from shaders import compute_vertex
from shaders import compute_fragment_2 as compute_fragment
from shaders import render_3D_vertex
from shaders import render_3D_fragment
from shaders import shadow_vertex
from shaders import shadow_fragment

# ? Use of this ?
# gl.use_gl('gl+')

################################################################################


class Camera():
    """
    Simple class that represents a camera, and holds all its characteristics.
    """

    def __init__(self,
               model,
               eye=[1,1,-1],
               target=[0,0,0],
               up=[0,1,0],
               fov=60.0,
               aspect=1.0,
               near=1.0,
               far=100.0):
        self.eye = eye
        self.target = target
        self.up = up
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far

        self.fovMin = 5.0
        self.fovMax = 120.0
        self.fovRange = self.fovMax - self.fovMin
        self.distanceMin = 0.5
        self.distanceMax = 10.0
        self.elevationMin = pi/2.0*.01
        self.elevationMax = pi/2.0*.99
        self.elevationRange = self.elevationMax - self.elevationMin


        (self.azimuth, self.elevation, self.distance) = self.InitializeAzElDi()
        # these parameters modulate the target y to better center the grid in
        # the view, according to fov and elevation. They are first computed by
        # the two calls at move and setProjection. See methods for details.
        self.centerModFromElev = 0
        self.centerModFromFov = 0
        self.model = model
        self.view = np.eye(4, dtype=np.float32)
        self.projection = np.eye(4, dtype=np.float32)
        self.vm = np.eye(4, dtype=np.float32)
        self.pvm = np.eye(4, dtype=np.float32)

        self.move()
        self.setProjection()

        self.defaultEye = eye
        self.defaultTarget = target
        self.defaultFov = fov
        self.defaultAzimuth = self.azimuth
        self.defaultElevation = self.elevation
        self.defaultDistance = self.distance

        self.sensitivity = 5.0

    def InitializeAzElDi(self):
        """
        Computes azimuth, elevation and distance from eye and target received.
        """
        (dx, dy, dz) = np.subtract(self.eye, self.target)
        distance = np.linalg.norm((dx, dy, dz))
        elevation = atan2(dy, sqrt(dx**2 + dz**2))
        azimuth = atan2(dz, dx)
        return (azimuth, elevation, distance)

    def setEye(self, eye):
        """
        Sets the coordinates of the camera, computes its view matrix and return it.
        """
        self.eye = eye
        self.view = self.lookAt(self.eye, self.target, up=self.up)
        return self.view

    def setTarget(self, target):
        """
        Sets the target at which the camera is pointing, compute its view matrix and returns it.
        """
        self.target = target
        self.view = self.lookAt(self.eye, self.target, up=self.up)
        return self.view

    def move(self, azimuth=None, elevation=None, distance=None, target=None):
        """
        Moves the camera according to the azimuth, elevation, distance and
        target received, compute its view matrix and returns it.
        It also compute the modulation of target y according to elevation.
        """
        self.azimuth = azimuth or self.azimuth
        # clamp elevation to avoid awkward flip turn when camera is placed exactly
        # facing down
        self.elevation = elevation or self.elevation
        self.elevation = min(max(self.elevation, self.elevationMin), self.elevationMax)
        self.distance = distance or self.distance
        self.distance = min(max(self.distance, self.distanceMin), self.distanceMax)
        z = self.distance * sin(self.elevation) * sin(self.azimuth)
        x = self.distance * sin(self.elevation) * cos(self.azimuth)
        y = self.distance * cos(self.elevation)
        eye = [x, y, z]
        self.centerModFromElev = (1.0 - (2.0 * abs(((self.elevation - self.elevationMin) / self.elevationRange) - .5))**1)
        target = target or [0, self.centerModFromFov*self.centerModFromElev, 0]
        self.view = self.lookAt(eye, target, up=self.up)
        return self.view

    def setProjection(self, fov=None, aspect=None, near=None, far=None):
        """
        Compute the projection matrix of the camera and returns it.
        it also computes the modulation of target y according to fov.
        """
        self.fov = fov or self.fov
        self.fov = min(max(self.fov, self.fovMin), self.fovMax)
        self.centerModFromFov = -0.6 * (self.fov - self.fovMin) / self.fovRange
        self.aspect = aspect or self.aspect
        self.near = near or self.near
        self.far = far or self.far
        self.projection = perspective(self.fov, self.aspect, self.near, self.far)
        self.view = self.lookAt(self.eye, target=[0,self.centerModFromFov*self.centerModFromElev, 0])
        return self.projection

    def zoomOn(self, objectWidth=sqrt(2.0), margin=0.02):
        """
        Compute a projection matrix which frustum just covers the width of
        the object given. This is used in all directions. This width should
        be the diameter of the sphere containing the object, centered on the
        center of the object. it returns the projection matrix.
        """
        self.distance = np.linalg.norm(np.subtract(self.target, self.eye))
        radius = min(objectWidth/2.0*(1.0+margin), sqrt(2.0)*0.51)
        fov = 2 * asin(radius / self.distance) * 180. / pi
        ratioNearFar = 2.0
        near = self.distance - radius * 1.0 / ratioNearFar
        far = self.distance + radius * ratioNearFar
        return self.setProjection(fov, self.aspect, near, far)

    def lookAt(self, eye, target, up=[0, 1, 0]):
        """
        Computes matrix to put eye looking at target point.
        """
        self.eye = eye
        self.target = target
        self.up = up
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
        self.view = view
        self.InitializeAzElDi()
        self.updateMatrices()
        return view

    def updateMatrices(self):
        """
        Recompute view*model and projection*view*model
        """
        self.vm = self.model @ self.view
        self.pvm = self.vm @ self.projection


class GrayScottModel():
    """
    Simple class to encapsulate all things related to the
    Gray-Scott reaction-diffusion and its management
    """
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

    def __init__(self,
                 canvas=None,
                 gridSize=[512, 512],
                 specie='lambda_left'):
        # Base parameters
        self.canvas = canvas
        self.pingpong = 1
        self.specie = specie
        (self.h, self.w) = gridSize

        # Mouse interactions parameters
        # --------------------------------------
        self.brush = [0, 0]
        self.brushType = 0

        # Build plane data
        # --------------------------------------
        # Vertices contains
        # Position being vec3
        # texcoord being vec2
        # normal being vec3
        # color being vec4
        V, F, outline = create_plane(width_segments=self.w, height_segments=self.h, direction='+y')
        self.vertices = VertexBuffer(V)
        self.faces = IndexBuffer(F)

        # Build program to compute Gray-Scott Model step
        # --------------------------------------
        self.program = Program(compute_vertex, compute_fragment, count=4)
        self.program["position"] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        self.program["texcoord"] = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.program["dx"] = 1./self.w
        self.program["dy"] = 1./self.h
        self.program['pingpong'] = self.pingpong
        self.program['brush'] = self.brush
        self.program['brushtype'] = self.brushType

        # Build texture data and set it to program
        # --------------------------------------
        # the texture contains 4 layers r, g, b, a
        # containing U and V concentrations twice
        # and are used through pingpong alternatively
        # each GPU computation/rendering cycle
        self.gridReinitialized = False
        self.initializeGrid()

        # Pearson's patterns related variables
        # --------------------------------------
        # defines parameters for du, dv, f, k
        # and passes them to program
        self.setSpecie(specie=self.specie)

        # Define a FrameBuffer to update model state in texture
        # --------------------------------------
        self.buffer = FrameBuffer(color=self.program["texture"],
                                  depth=gloo.RenderBuffer((self.h, self.w), format='depth'))

        # cycle of computation per frame
        # --------------------------------------
        self.cycle = 0

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
        self.UV = np.rot90(self.UV, k=1)
        if not hasattr(self, 'texture'):
            self.texture = gloo.Texture2D(data=self.UV, format=gl.GL_RGBA, internalformat='rgba32f', interpolation='linear')
        else:
            self.texture.set_data(self.UV)

        self.program["texture"] = self.texture
        self.program["texture"].interpolation = gl.GL_NEAREST
        self.program["texture"].wrapping = gl.GL_REPEAT
        self.gridReinitialized = True

    def printPearsonPatternDescription(self):
        if self.canvas is not None:
            self.canvas.title2 = '3D Gray-Scott Reaction-Diffusion: Pattern %s - GregVDS' % self.specie
        specie = GrayScottModel.species[self.specie]
        print('Pearson\'s Pattern %s' % self.specie)
        print(specie[4])
        print('        dU  dV  f     k \n        %s %s %s %s' % (specie[0],
                                                                 specie[1],
                                                                 specie[2],
                                                                 specie[3]))

    def setSpecie(self, event=None, specie=''):
        """
        Set the feed, kill and diffusion rates for the choosen pattern.
        """
        if specie != '':
            self.specie = specie
            self.printPearsonPatternDescription()
            self.program["params"] = GrayScottModel.species[self.specie][0:4]

    def interact(self, brushCoords, brushType):
        """
        Modify local V concentrations.
        """
        self.program['brush'] = brushCoords
        self.program['brushtype'] = brushType

    def increaseCycle(self, event=None):
        """
        Increases number of cycles processed per frame shown.
        Each increase multiplies by two the number of added cycles.
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
        Each decrease divides by two the number of added cycles.
        """
        self.cycle = int(self.cycle/2)
        if self.cycle < 1:
            self.cycle = 0
        print(' Number of cycles: %3.0f' % (1 + 2 * self.cycle), end='\r')

    def draw(self, drawingType='triangle_strip'):
        """
        Draws one or several step(s) of the reaction-diffusion.
        """
        self.program.draw(drawingType)
        for cycle in range(self.cycle):
            self.flipPingpong()
            self.program.draw(drawingType)
            self.flipPingpong()
            self.program.draw(drawingType)

    def flipPingpong(self):
        """
        Toggles between rg and ba sets in texture.
        """
        self.pingpong = 1 - self.pingpong
        self.program["pingpong"] = self.pingpong


class Renderer():
    """
    Generic renderer that holds the common functions.
    """

    def __init__(self,
                 grayScottModel,
                 camera,
                 vertex_shader,
                 fragment_shader):
        self.grayScottModel = grayScottModel
        self.camera = camera
        self.reagent = 1

        # Build render program
        # --------------------------------------
        self.program = Program(vertex_shader, fragment_shader)
        self.program.bind(self.grayScottModel.vertices)
        self.program['pingpong']      = self.grayScottModel.program['pingpong']
        self.program["reagent"]       = self.reagent
        self.program["scalingFactor"] = 30. * (self.grayScottModel.w / 512)
        self.program["u_pvm"]         = self.camera.pvm

    def draw(self, drawingType='triangles'):
        self.program.draw(drawingType, self.grayScottModel.faces)


class ShadowRenderer(Renderer):
    """
    A renderer for shadowmap.
    """

    def __init__(self,
                 grayScottModel,
                 camera,
                 shadowMapSize=1024):
        super().__init__(grayScottModel,
                        camera,
                        shadow_vertex,
                        shadow_fragment)

        self.shadowMapSize = shadowMapSize
        self.shadowGrid = np.ones((self.shadowMapSize, self.shadowMapSize, 4), dtype=np.float32) * .1
        self.shadowTexture = gloo.texture.Texture2D(data=self.shadowGrid, format=gl.GL_RGBA, internalformat='rgba32f')

        # Complete shadowmap render program
        # --------------------------------------
        self.camera.zoomOn(objectWidth=sqrt(2.0), margin=0.02)
        self.program['u_pvm'] = self.camera.pvm
        self.program["texture"] = self.grayScottModel.program["texture"]
        self.program["texture"].interpolation = gl.GL_LINEAR
        self.program["texture"].wrapping = gl.GL_REPEAT


class MainRenderer(Renderer):

    colormapDictionnary = {
        '&': 'Boston',
        'é': 'malmo_r',
        '"': 'papetee_r',
        '\'': 'oslo',
        '(': 'honolulu_r',
        '§': 'Rejkjavik_r',
        'è': 'antidetroit',
        '!': 'osmort',
        'ç': 'irkoutsk_r',
        'à': 'vancouver_r'
    }

    colormapDictionnaryShifted = {
        '1': 'Boston_r',
        '2': 'malmo',
        '3': 'papetee',
        '4': 'oslo_r',
        '5': 'honolulu',
        '6': 'Rejkjavik',
        '7': 'detroit',
        '8': 'tromso',
        '9': 'irkoutsk',
        '0': 'vancouver'
    }

    createColormaps()

    SHADOW_TYPE = ("None                     ",
                   "Simple                   ",
                   "Percent 4 samplings      ",
                   "Variance shadow mapping  ")

    REAGENTS = ('U', 'V')

    def __init__(self,
                 grayScottModel,
                 camera,
                 shadowRenderer,
                 cmap='irkoutsk'):
        super().__init__(grayScottModel,
                        camera,
                        render_3D_vertex,
                        render_3D_fragment)

        self.shadowRenderer = shadowRenderer
        self.cmapName = cmap

        # light Parameters: direction, shininess exponant, attenuation parameters,
        # ambientLight intensity, ambientColor, diffuseColor and specularColor
        # --------------------------------------
        self.lightCoordinates = [self.shadowRenderer.camera.eye[0],
                                 self.shadowRenderer.camera.eye[1],
                                 self.shadowRenderer.camera.eye[2]]
        self.lightIntensity = (1., 1., 1.)
        self.c1 = 1.0
        self.c2 = 0.5
        self.c3 = 0.02
        self.ambientIntensity = 0.5
        self.ambientColor = np.array((1., 1., 1., 1))
        self.diffuseColor = np.array((1., 1., .9, 1.))
        self.specularColor = np.array((1., 1., .95, 1.))
        self.shininess = 91.0

        # Build a lightbox for specular Environment
        # --------------------------------------
        self.lightBoxTexture = np.zeros((6, 1024, 1024, 3), dtype=np.float32)
        self.lightBoxTexture[0] = np.rot90(read_png(load_data_file("skybox/sky-right.png"))/255., 1) #RIGHT
        self.lightBoxTexture[1] = np.rot90(read_png(load_data_file("skybox/sky-left.png"))/255., 1) #LEFT
        self.lightBoxTexture[2] = np.rot90(read_png(load_data_file("skybox/sky-front.png"))/255., 1) #DOWN
        self.lightBoxTexture[3] = np.rot90(read_png(load_data_file("skybox/sky-back.png"))/255., 1) #UP
        self.lightBoxTexture[4] = np.rot90(read_png(load_data_file("skybox/sky-up.png"))/255., 1) #BACK
        self.lightBoxTexture[5] = np.rot90(read_png(load_data_file("skybox/sky-down.png"))/255., 1) #FRONT
        # self.lightBoxTexture = createLightBox()

        # Toggles to switch on and off different parts of lighting
        # --------------------------------------
        self.ambient     = True
        self.attenuation = True
        self.diffuse     = True
        self.specular    = True
        self.shadow      = 3
        self.lightBox    = True

        # Complete render program
        # --------------------------------------
        self.program["texture"]                = self.grayScottModel.program["texture"]
        self.program["texture"].interpolation  = gl.GL_LINEAR
        self.program["texture"].wrapping       = gl.GL_REPEAT
        self.program["dx"]                     = 1. / self.grayScottModel.w
        self.program["dy"]                     = 1. / self.grayScottModel.h
        self.program["u_vm"]                   = self.camera.vm

        self.program["shadowMap"]              = self.shadowRenderer.shadowTexture
        self.program["shadowMap"].interpolation = gl.GL_LINEAR
        self.program["shadowMap"].wrapping     = gl.GL_CLAMP_TO_EDGE
        self.program["u_shadowmap_pvm"]        = self.shadowRenderer.camera.pvm

        self.program["u_Tolerance_constant"]   = 5e-03
        self.program["vsf_gate"]               = 2.5e-05
        self.program["ambient"]                = self.ambient
        self.program["attenuation"]            = self.attenuation
        self.program["diffuse"]                = self.diffuse
        self.program["specular"]               = self.specular
        self.program["shadow"]                 = self.shadow
        self.program["lightBox"]               = self.lightBox
        self.program["cubeMap"]                = gloo.TextureCube(self.lightBoxTexture, interpolation='linear')
        self.program["u_light_position"]       = self.lightCoordinates
        self.program["u_light_intensity"]      = self.lightIntensity
        self.program["u_Ambient_color"]        = self.ambientColor
        self.program["u_ambient_intensity"]    = self.ambientIntensity
        self.program["u_diffuse_color"]        = self.diffuseColor
        self.program["u_specular_color"]       = self.specularColor
        self.program['u_Shininess']            = self.shininess
        self.program['c1']                     = self.c1
        self.program['c2']                     = self.c2
        self.program['c3']                     = self.c3

        self.setColorMap(name=self.cmapName)

        # Define a 'DepthBuffer' to render the shadowmap from the light
        # --------------------------------------
        self.buffer = FrameBuffer(color=self.program["shadowMap"],
                                  depth=gloo.RenderBuffer((self.shadowRenderer.shadowMapSize, self.shadowRenderer.shadowMapSize), format='depth'))

    def refreshTextureInterpolation(self):
        """
        As the GrayScottModel renderer uses a texture with interpolation set
        at gl.GL_NEAREST, when it is reinitialized, the interpolation here is
        lost and has to be reset to gl.GL_LINEAR.
        """
        if self.grayScottModel.gridReinitialized is True:
            self.program["texture"].interpolation = gl.GL_LINEAR
            self.grayScottModel.gridReinitialized = False

    def moveCamera(self, dAzimuth=0.0, dElevation=0.0, dDistance=0.0):
        """
        Moves the camera according to inputs. Compute view Matrix.
        """
        azimuth = self.camera.azimuth + dAzimuth/self.camera.sensitivity
        elevation = self.camera.elevation + dElevation/self.camera.sensitivity
        distance = self.camera.distance + dDistance
        self.camera.move(azimuth=azimuth, elevation=elevation, distance=distance)
        self.program["u_vm"]                  = self.camera.vm
        self.program["u_pvm"]                 = self.camera.pvm
        # self.adjustShadowMapFrustum()

    def zoomCamera(self, percentage=0.0):
        """
        Zoom with camera according to inputs. Computes projection Matrix.
        """
        self.camera.setProjection(fov=self.camera.fov*(1+percentage))
        self.program["u_vm"]                  = self.camera.vm
        self.program["u_pvm"]                 = self.camera.pvm
        # self.adjustShadowMapFrustum()

    def adjustShadowMapFrustum(self):
        """
        Attempts at adjusting at its narrowest possible the
        shadowRenderer camera projection to optimize its resolution.
        """
        fieldWidth = self.camera.distance * 2.0 * sin(self.camera.fov/2.0 * pi / 180.0)
        # The lower the elevation the more problematic the shadow frustum can be
        # The wider the fov of the self.camera, the more problematic too...
        securityBuffer = .02 + sin(self.camera.elevation)
        # We limit the extent of the shadowMap projection to the model, useless
        # to go wider than that.
        fieldWidth = min(fieldWidth, sqrt(2.0))
        self.shadowRenderer.camera.zoomOn(fieldWidth, margin=securityBuffer)
        self.shadowRenderer.program['u_pvm'] = self.shadowRenderer.camera.pvm
        self.program["u_shadowmap_pvm"] = self.shadowRenderer.camera.pvm

    def updateAspect(self, aspect):
        """
        Compute projection Matrix for new Wndow aspect.
        """
        self.camera.setProjection(aspect=aspect)
        self.program["u_vm"]                  = self.camera.vm
        self.program["u_pvm"]                 = self.camera.pvm

    def resetCamera(self, event=None):
        """
        Replaces the camera at its original position.
        """
        self.camera.move(self.camera.defaultAzimuth,
                                     self.camera.defaultElevation,
                                     self.camera.defaultDistance)
        self.camera.setProjection(fov=self.camera.defaultFov)
        self.program["u_vm"]                  = self.camera.vm
        self.program["u_pvm"]                 = self.camera.pvm

    def modifyLightCharacteristic(self, event, lightType=None, modification=None):
        """
        Modifies one of the light parameters.
        Some like ambient, diffuse, specular are simply toggles on/off,
        Others increase/decrease a value, such as the shininess exponant
        """
        if lightType == 'ambient':
            self.ambient = not self.ambient
            self.program[lightType] = self.ambient
            print(' Ambient light: %s        ' % self.ambient, end="\r")
        elif lightType == 'diffuse':
            self.diffuse = not self.diffuse
            self.program[lightType] = self.diffuse
            print(' Diffuse light: %s        ' % self.diffuse, end="\r")
        elif lightType == 'specular':
            self.specular = not self.specular
            self.program[lightType] = self.specular
            print(' Specular light: %s       ' % self.specular, end="\r")
        elif lightType == 'shadow':
            self.shadow = (self.shadow + 1) % len(MainRenderer.SHADOW_TYPE)
            self.program[lightType] = self.shadow
            print(' Shadows: %s              ' % self.SHADOW_TYPE[self.shadow], end="\r")
        elif lightType == 'attenuation':
            self.attenuation = not self.attenuation
            self.program[lightType] = self.attenuation
            print(' Attenuation: %s          ' % self.attenuation, end="\r")
        elif lightType == 'shininess' and modification == '-':
            self.shininess *= sqrt(2)
            self.shininess = np.clip(self.shininess, 0.1, 8192)
            self.program['u_Shininess'] = self.shininess
            print(' Shininess exponant: %3.0f' % self.shininess, end="\r")
        elif lightType == 'shininess' and modification == '+':
            self.shininess /= sqrt(2)
            self.shininess = np.clip(self.shininess, 0.1, 8192)
            self.program['u_Shininess'] = self.shininess
            print(' Shininess exponant: %3.0f' % self.shininess, end="\r")
        elif lightType == "vsf_gate" and modification == '+':
            self.program['vsf_gate'] = self.program['vsf_gate'] * 2.0
            print(" vsf_gate: %s             " % self.program['vsf_gate'])
        elif lightType == "vsf_gate" and modification == '-':
            self.program['vsf_gate'] = self.program['vsf_gate'] / 2.0
            print(" vsf_gate: %s             " % self.program['vsf_gate'])
        elif lightType == "lightbox":
            self.lightBox = not self.lightBox
            self.program["lightBox"] = self.lightBox
            print(" LightBox %s              " % self.lightBox)

    def setColorMap(self, event=None, name=''):
        """
        Set the colormap used to render the concentration
        """
        if name != '':
            self.cmapName = name
            self.program["cmap"] = get_colormap(self.cmapName).map(np.linspace(0.0, 1.0, 1024)).astype('float32')
            print(' Using colormap %s.              ' % name, end="\r")

    def switchReagent(self, event=None):
        """
        Toggles between representing U or V concentrations.
        """
        self.reagent = 1 - self.reagent
        self.program["reagent"] = self.reagent
        self.shadowRenderer.program["reagent"] = self.reagent
        print(' Displaying %s reagent concentration.' % MainRenderer.REAGENTS[self.reagent], end="\r")


class Canvas(app.Canvas):

    def __init__(self,
                 size=(1024, 1024),
                 modelSize=(512,512),
                 specie='alpha_left',
                 cmap='irkoutsk'):
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
                             aspect=self.size[0] / float(self.size[1]))

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
        self.show()

    ############################################################################
    #

    def on_draw(self, event):
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
        self.mainRenderer.refreshTextureInterpolation()
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
        self.mainRenderer.moveCamera(dDistance=(-event.delta[1])/3.0)
        # Shift modifier key: zoom in out
        self.mainRenderer.zoomCamera((event.delta[0])/3.0)

    def on_mouse_press(self, event):
        self.pressed = True
        if len(event.modifiers) == 0:
            self.mousePos = event.pos
        elif len(event.modifiers) == 1 and event.modifiers[0] == 'Shift':
            (x, y) = event.pos
            (sx, sy) = self.size
            xpos = 1 - x/sx
            ypos = 1 - y/sy
            self.grayScottModel.interact([ypos, xpos], event.button)

    def on_mouse_release(self, event):
        self.pressed = False
        self.grayScottModel.interact([0, 0], 0)

    def on_mouse_move(self, event):
        if(self.pressed):
            if len(event.modifiers) == 0:
                # no Shift modifier key: moves the camera
                dazimuth = (event.pos[0] - self.mousePos[0]) * (2*pi) / self.size[0]
                delevation = -1.0 * (event.pos[1] - self.mousePos[1]) * (2*pi) / self.size[1]
                self.mousePos = event.pos
                self.mainRenderer.moveCamera(dAzimuth=dazimuth, dElevation=delevation)
            elif len(event.modifiers) == 1 and event.modifiers[0] == 'Shift':
                (x, y) = event.pos
                (sx, sy) = self.size
                xpos = 1 - x/sx
                ypos = 1 - y/sy
                # update brush coords here
                self.grayScottModel.interact([ypos, xpos], event.button)

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
                func(self, event, *args)
            elif hasattr(self.grayScottModel, func.__name__):
                func(self.grayScottModel, event, *args)
            elif hasattr(self.mainRenderer, func.__name__):
                func(self.mainRenderer, event, *args)
            else:
                print("Method %s does not seem to be found..." % str(fun))

    ############################################################################
    # Debug/utilities functions

    def switchDisplay(self, event=None):
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
        (',', ('Control',)): (MainRenderer.modifyLightCharacteristic, ('ambient',)),
        (';', ('Control',)): (MainRenderer.modifyLightCharacteristic, ('diffuse',)),
        (':', ('Control',)): (MainRenderer.modifyLightCharacteristic, ('specular',)),
        ('=', ('Control',)): (MainRenderer.modifyLightCharacteristic, ('shadow',)),
        ('N', ('Control',)): (MainRenderer.modifyLightCharacteristic, ('attenuation',)),
        ('L', ('Control',)): (MainRenderer.modifyLightCharacteristic, ('shininess', '-')),
        ('M', ('Control',)): (MainRenderer.modifyLightCharacteristic, ('shininess', '+')),
        ('J', ('Control',)): (MainRenderer.modifyLightCharacteristic, ('vsf_gate', '-')),
        ('K', ('Control',)): (MainRenderer.modifyLightCharacteristic, ('vsf_gate', '+')),
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
                        choices={**GrayScottModel.speciesDictionnary,
                                 **GrayScottModel.speciesDictionnaryShifted}.values(),
                        default="alpha_left",
                        help="Pearson\' pattern")
    parser.add_argument("-c",
                        "--Colormap",
                        choices={**MainRenderer.colormapDictionnary,
                                 **MainRenderer.colormapDictionnaryShifted}.values(),
                        default="papetee_r",
                        help="Colormap used")

    args = parser.parse_args()

    c = Canvas(modelSize=(args.Size, args.Size),
               size=(args.Window, args.Window),
               specie=args.Pattern,
               cmap=args.Colormap)
    c.measure_fps2(callback=fun)
    app.run()
