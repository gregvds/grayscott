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
from math import pi, sin, cos, asin, sqrt, atan2
import textwrap, sys, os


import numpy as np

from vispy import gloo, app
from vispy.geometry import create_plane
from vispy.gloo import gl, Program, VertexBuffer, IndexBuffer, FrameBuffer
from vispy.io import read_png, load_data_file
from vispy.util.transforms import perspective

from gs_lib import (get_colormap, createColormaps, import_pearsons_types, setup_grid, gauss)

from shaders import compute_vertex
from shaders import compute_fragment_2 as compute_fragment_isotropic
from shaders import compute_fragment_3 as compute_fragment_anisotropic
from shaders import render_3D_vertex
from shaders import render_3D_fragment
from shaders import shadow_vertex
from shaders import shadow_fragment

# ? Use of this ?
# gl.use_gl('gl+')
# app.use_app('pyside6')

################################################################################


class Camera():
    """
    Simple class that represents a camera, and holds all its characteristics.
    """

    def __init__(self,
               model,
               eye,
               target,
               up,
               fov=60.0,
               aspect=1.0,
               near=1.0,
               far=100.0,
               shadowCam=False):
        self.eye = eye
        self.target = target
        self.up = up
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far
        self.shadowCam = shadowCam

        self.fovMin = 5.0
        self.fovMax = 120.0
        self.fovRange = self.fovMax - self.fovMin
        self.distanceMin = 0.5
        self.distanceMax = 10.0
        self.elevationMin = pi/2.0*.001
        self.elevationMax = pi/2.0*.999
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
        if self.shadowCam is True:
            self.zoomOn()
        self.defaultEye = self.eye
        self.defaultTarget = self.target
        self.defaultFov = self.fov
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
        x = self.distance * sin(self.elevation) * sin(self.azimuth)
        z = self.distance * sin(self.elevation) * cos(self.azimuth)
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
        'r': 'rho_left',
        's': 'sigma_left',
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
        'N': 'nu_right',
        'R': 'rho_right',
        'S': 'sigma_right',
        'T': 'theta_right',
        'X': 'xi_right',
        'Z': 'zeta_right'
    }

    (species, fMin, fMax, kMin, kMax) = import_pearsons_types()

    def __init__(self,
                 canvas=None,
                 gridSize=[512, 512],
                 specie='lambda_left',
                 isotropic=True):
        # Base parameters
        self.canvas = canvas
        self.pingpong = 1
        self.specie = specie
        self.isotropic = isotropic
        (self.h, self.w) = gridSize
        # A reference to the instances that uses this GrayScottModel, just to
        # be able to refresh it gl.GL_LINEAR texture parameters when a call to
        # self.initializeGrid() occurs.
        self.owner = []

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

        # Pearson's patterns related variables
        # --------------------------------------
        # defines parameters for du, dv, f, k
        # and passes them to program
        self.dFeed = 0.0
        self.dKill = 0.0
        self.dU = 1.0
        self.dV = 0.5
        self.dDUDV = 0.0
        self.dUMin = 0.2
        self.dUMax = 1.3
        self.dVMin = 0.5*self.dUMin
        self.dVMax = 0.5*self.dUMax

        # Build program to compute Gray-Scott Model according the the isotropic
        # parameter. if True, feed, kill, dU and dV will be the same through all
        # the model, else they will be gridded and can be modulated in the grid.
        # --------------------------------------
        if self.isotropic is True:
            self.program = Program(compute_vertex, compute_fragment_isotropic, count=4)
        else:
            self.program = Program(compute_vertex, compute_fragment_anisotropic, count=4)
            self.program["dUMin"] = self.dUMin
            self.program["dUMax"] = self.dUMax
            self.program["dVMin"] = self.dVMin
            self.program["dVMax"] = self.dVMax
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
        # self.gridReinitialized = False
        self.initializeGrid()
        self.baseParams = []
        self.setSpecie(specie=self.specie)

        # Define a FrameBuffer to update model state in texture
        # --------------------------------------
        self.buffer = FrameBuffer(color=self.program["texture"],
                                  depth=gloo.RenderBuffer((self.h, self.w), format='depth'))

        # cycle of computation per frame
        # --------------------------------------
        self.cycle = 0

    def initializeGrid(self):
        """
        Initialize the concentrations of U and V of the model accross a grid
        with a seed patch in its center.
        """
        self.UV = np.zeros((self.h, self.w, 4), dtype=np.float32)
        self.UV[:, :, 0:2] = setup_grid(self.h, self.w)
        self.UV += np.random.uniform(-0.02, 0.1, (self.h, self.w, 4))
        self.UV[:, :, 2] = self.UV[:, :, 0]
        self.UV[:, :, 3] = self.UV[:, :, 1]
        if not hasattr(self, 'texture'):
            print(' Initialization of the grid.', end="\r")
            self.texture = gloo.Texture2D(data=self.UV,
                                          format=gl.GL_RGBA,
                                          internalformat='rgba32f')
        else:
            print(' Reinitialization of the grid.', end="\r")
            self.texture.set_data(self.UV)

        self.program["texture"] = self.texture
        self.program["texture"].interpolation = gl.GL_NEAREST
        self.program["texture"].wrapping = gl.GL_REPEAT
        # inside, the texture is used with interpolation = gl.GL_NEAREST, but
        # in the Renderer, it should be interpolation = gl.GL_LINEAR
        for owner in self.owner:
            if hasattr(owner, 'program'):
                owner.program["texture"].interpolation = gl.GL_LINEAR

    def printPearsonPatternDescription(self):
        if self.canvas is not None and self.canvas.native.parent() is None:
            self.canvas.title2 = '3D Gray-Scott Reaction-Diffusion: Pattern %s - GregVDS' % self.specie
            print(self.getPearsonPatternDescription())

    def getPearsonPatternDescription(self, specie=None):
        """
        Returns a text describing the current Pearson's pattern and associated
        parameters.
        """
        specie = specie or self.specie
        specieDetail = GrayScottModel.species[specie]
        symbol = specieDetail[5]
        text = "%s - %s\n" % (symbol, specie)
        text += "%s\n\n" % specieDetail[4]
        text += 'dU: %1.3f\n' % specieDetail[0]
        text += 'dV: %1.3f\n' % specieDetail[1]
        text += 'f : %1.3f\n' % specieDetail[2]
        text += 'k : %1.3f\n' % specieDetail[3]
        return text

    def setSpecie(self, specie=''):
        """
        Set the feed, kill and diffusion rates for the choosen pattern.
        """
        if specie != '':
            self.specie = specie
            self.printPearsonPatternDescription()
            self.baseParams = GrayScottModel.species[self.specie][0:4]
            if self.isotropic is True:
                self.program["params"] = self.baseParams
            else:
                self.P = np.zeros((self.h, self.w, 4), dtype=np.float32)
                self.P[:, :] = self.baseParams
                # self.modulateFK()
                self.updateParams(feed=self.baseParams[2],
                                  kill=self.baseParams[3],
                                  dU=self.baseParams[0],
                                  dV=self.baseParams[1])
                self.setProgramParams()

    def setParams(self, feed=None, kill=None, dU=None, dV=None, dFeed=None, dKill=None, dDUDV=None):
        """
        set one or more parameters of the model, feed, kill, dU and/or dV
        """
        if self.isotropic is True:
            vals = self.program["params"]
            vals[0] = dU or vals[0]
            vals[1] = dV or vals[1]
            vals[2] = feed or vals[2]
            vals[3] = kill or vals[3]
            self.program["params"] = vals
        else:
            self.dFeed = dFeed or self.dFeed
            self.dKill = dKill or self.dKill
            self.dDUDV = dDUDV or self.dDUDV
            self.dU    = dU or self.dU
            self.dV    = dV or self.dV
            if feed is not None or kill is not None or dU is not None or dV is not None:
                self.updateParams(feed, kill, dU, dV)
            if dFeed is not None or dKill is not None:
                self.modulateFK()
            if dDUDV is not None:
                self.ModulateDUDV(self.dU, self.dDUDV, self.dUMin, self.dUMax, 0)
                self.ModulateDUDV(self.dV, self.dDUDV, self.dVMin, self.dVMax, 1)
            self.setProgramParams()

    def updateParams(self, feed=None, kill=None, dU=None, dV=None):
        """
        Updates feed, kill, dU and dV params in texture when not in isotropic mode.
        """
        f = self.P[0, 0, 2]
        k = self.P[0, 0, 3]
        feed = feed or f
        kill = kill or k
        self.P[:, :, 2] -= f
        self.P[:, :, 3] -= k
        self.P[:, :, 2] = np.clip(self.P[:, :, 2] + feed, self.fMin, self.fMax)
        self.P[:, :, 3] = np.clip(self.P[:, :, 3] + kill, self.kMin, self.kMax)

        self.dU = dU or self.dU
        self.dV = dV or self.dV
        if dU is not None:
            self.ModulateDUDV(self.dU, self.dDUDV, self.dUMin, self.dUMax, 0)
        if dV is not None:
            self.ModulateDUDV(self.dV, self.dDUDV, self.dVMin, self.dVMax, 1)

    def modulateFK(self):
        """
        Modulates feed and kill with dFeed and dKill when not in isotropic mode.
        """
        f = self.P[0, 0, 2]
        k = self.P[0, 0, 3]
        rows, cols = self.h, self.w
        sinsF = np.sin(np.linspace(0.0, 2*np.pi, cols))
        sinsK = np.sin(np.linspace(0.0, 2*np.pi, rows))
        for i in range(rows):
            self.P[i, :, 2] = np.clip(f + self.dFeed*sinsF, self.fMin, self.fMax)
        for i in range(cols):
            self.P[:, i, 3] = np.clip(k + self.dKill*sinsK, self.kMin, self.kMax)

    def ModulateDUDV(self, dRef, dVar, dMin, dMax, indexInParams):
        ddPivot = (dRef - dMin) / (dMax - dMin)
        ddUpperProportion = (dMax - dRef) / (dMax - dMin)
        ddLowerProportion = ddPivot
        gaussGrid = gauss(size=[self.h, self.w], sigma=0.33)
        if dVar > 0.0:
            self.P[:, :, indexInParams] = (1 - dVar) * ddPivot \
                                 + (dVar) * (ddPivot \
                                             + (gaussGrid * ddUpperProportion) \
                                             - ((1-gaussGrid) * ddLowerProportion)
                                              )
        elif dVar < 0.0:
            self.P[:, :, indexInParams] = (1 - dVar) * ddPivot \
                                 + (dVar) * (ddPivot \
                                             - ((1-gaussGrid) * ddUpperProportion) \
                                             + (gaussGrid * ddLowerProportion)
                                              )
        else:
            self.P[:, :, indexInParams] = ddPivot

    def setProgramParams(self):
        """
        Updates the parameters of the model when not in isotropic mode.
        """
        self.params = gloo.texture.Texture2D(data=self.P, format=gl.GL_RGBA, internalformat='rgba32f')
        self.program["params"] = self.params

    def interact(self, brushCoords, brushType):
        """
        Modify local V concentrations.
        """
        self.program['brush'] = brushCoords
        self.program['brushtype'] = brushType

    def increaseCycle(self):
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

    def decreaseCycle(self):
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
        self.camera.move()
        self.camera.setProjection()
        # Add itself to the list of owners of this grayScottModel
        self.grayScottModel.owner.append(self)

        # Build render program
        # --------------------------------------
        self.program = Program(vertex_shader, fragment_shader)
        self.program.bind(self.grayScottModel.vertices)
        self.program['pingpong']      = self.grayScottModel.program['pingpong']
        self.program["reagent"]       = self.reagent
        self.program["scalingFactor"] = 30. * (self.grayScottModel.w / 512)
        self.program["u_vm"]          = self.camera.vm
        self.program["u_pvm"]         = self.camera.pvm
        self.program["texture"]                = self.grayScottModel.program["texture"]
        self.program["texture"].interpolation  = gl.GL_LINEAR
        self.program["texture"].wrapping       = gl.GL_REPEAT

    def draw(self, drawingType='triangles'):
        self.program.draw(drawingType, self.grayScottModel.faces)

    def moveCamera(self, dAzimuth=0.0, dElevation=0.0, dDistance=0.0, dFov=0.0):
        """
        Moves the camera according to inputs. Compute view Matrix.
        """
        azimuth = self.camera.azimuth - dAzimuth/self.camera.sensitivity
        elevation = self.camera.elevation - dElevation/self.camera.sensitivity
        distance = self.camera.distance - dDistance
        self.camera.move(azimuth=azimuth, elevation=elevation, distance=distance)
        fov = self.camera.fov - dFov / self.camera.sensitivity
        self.camera.setProjection(fov=fov)
        self.program["u_vm"]  = self.camera.vm
        self.program["u_pvm"] = self.camera.pvm

    def resetCamera(self):
        """
        Replaces the camera at its original position.
        """
        self.camera.move(self.camera.defaultAzimuth,
                                     self.camera.defaultElevation,
                                     self.camera.defaultDistance)
        self.camera.setProjection(fov=self.camera.defaultFov)
        if self.camera.shadowCam is True:
            self.camera.zoomOn()
        self.program["u_vm"]  = self.camera.vm
        self.program["u_pvm"] = self.camera.pvm

    def switchReagent(self):
        """
        Toggles between representing U or V concentrations.
        """
        self.reagent = 1 - self.reagent
        self.program["reagent"] = self.reagent


class ShadowRenderer(Renderer):
    """
    Renderer for shadowmap.
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
        self.shadowGrid = np.ones((self.shadowMapSize, self.shadowMapSize, 4),
                                  dtype=np.float32) * .1
        self.shadowTexture = gloo.texture.Texture2D(data=self.shadowGrid,
                                                    format=gl.GL_RGBA,
                                                    internalformat='rgba32f')

        # Complete shadowmap render program
        # --------------------------------------
        self.camera.zoomOn(objectWidth=sqrt(2.0), margin=0.02)
        self.program["u_vm"] = self.camera.vm
        self.program['u_pvm'] = self.camera.pvm


class MainRenderer(Renderer):
    """
    Renderer responsible for the main display of the scene on the canvas.
    """

    colormapDictionnary = {
        '&': 'boston',
        'é': 'malmo_r',
        '"': 'papetee_r',
        '\'': 'oslo',
        '(': 'honolulu_r',
        '§': 'rejkjavik_r',
        'è': 'antidetroit',
        '!': 'osmort',
        'ç': 'irkoutsk_r',
        'à': 'vancouver_r',
        'q': 'helsinki'
    }

    colormapDictionnaryShifted = {
        '1': 'boston_r',
        '2': 'malmo',
        '3': 'papetee',
        '4': 'oslo_r',
        '5': 'honolulu',
        '6': 'rejkjavik',
        '7': 'detroit',
        '8': 'tromso',
        '9': 'irkoutsk',
        '0': 'vancouver',
        'Q': 'helsinki_r'
    }

    # This dictionnary holds all the lighting parameters. They are automatically
    # stuffed into the program uniform named u_firstKey_secondKey.
    # the value passed is the first of the list. Second member of the list is
    # the type of the variable; this for Qt to be able to build the correct
    # QWidgets for it. Further parameters are minimum and maximum when relevant
    # WIP one could add a kind of modification function (+int, +float, *float) to
    # use when modifying the value...
    lightingDictionnary = {
        "ambient": {
            "on": [True, "bool"],
            "color": [[1., 1., 1., 1], "color"],
            "intensity": [0.3, "float", 0.0, 1.0]
        },
        "diffuse": {
            "on": [True, "bool"],
            "color": [[1., 1., .9, 1.], "color"],
            "intensity": [1.0, "float", 0.0, 1.0]
        },
        "specular": {
            "on": [True, "bool"],
            "color": [[1., 1., .95, 1.], "color"],
            "shininess": [182.0, "float", 10.0, 8192.0]
        },
        "attenuation": {
            "on": [True, "bool"],
            "c1": [1.0, "float", 1.0, 2.0],
            "c2": [0.5, "float", 0.0, 1.0],
            "c3": [0.02, "float", 0.0, 1.0],
        },
        "shadow": {
            "on": [True, "bool"],
            "type": [2, "int", 0, 2],
            "hardtolerance": [5e-03, "float", 5e-04, 5e-2],
            "pcftolerance": [5e-03, "float", 5e-04, 5e-2],
            "pcfspreading": [1000.0, "float", 500.0, 3000.0],
            "vsfgate": [2.5e-05, "float", 1e-05, 5e-01],
        },
        "lightbox": {
            "on": [True, "bool"],
            "intensity": [0.9, "float", 0.0, 1.0],
            "fresnelexponant": [2.5, "float", 0.1, 5.0]
        }
    }

    createColormaps()

    SHADOW_TYPE = ("Simple                   ",
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

        # light Parameters:
        # --------------------------------------
        self.lightCoordinates = [self.shadowRenderer.camera.eye[0],
                                 self.shadowRenderer.camera.eye[1],
                                 self.shadowRenderer.camera.eye[2]]
        self.lightIntensity = (1., 1., 1.)

        # Build a lightbox for specular Environment
        # --------------------------------------
        # These png come from a vispy example and were used with z axis up,
        # so I had to fiddle around which png to what index and how many
        # rot90 to apply to each to have them stitching together...
        self.lightBoxTexture = np.zeros((6, 1024, 1024, 3), dtype=np.float32)
        self.lightBoxTexture[0] = np.rot90(read_png(load_data_file("skybox/sky-right.png"))/255., 1) #RIGHT
        self.lightBoxTexture[1] = np.rot90(read_png(load_data_file("skybox/sky-left.png"))/255., 1) #LEFT
        self.lightBoxTexture[2] = np.rot90(read_png(load_data_file("skybox/sky-front.png"))/255., 1) #DOWN
        self.lightBoxTexture[3] = np.rot90(read_png(load_data_file("skybox/sky-back.png"))/255., 1) #UP
        self.lightBoxTexture[4] = np.rot90(read_png(load_data_file("skybox/sky-up.png"))/255., 1) #BACK
        self.lightBoxTexture[5] = np.rot90(read_png(load_data_file("skybox/sky-down.png"))/255., 1) #FRONT

        # Complete render program
        # --------------------------------------
        self.program["dx"]                     = 1. / self.grayScottModel.w
        self.program["dy"]                     = 1. / self.grayScottModel.h
        self.program["shadowMap"]              = self.shadowRenderer.shadowTexture
        self.program["shadowMap"].interpolation = gl.GL_LINEAR
        self.program["shadowMap"].wrapping     = gl.GL_CLAMP_TO_EDGE
        self.program["u_shadowmap_pvm"]        = self.shadowRenderer.camera.pvm
        self.program["cubeMap"]                = gloo.TextureCube(self.lightBoxTexture, interpolation='linear')
        self.program["u_light_position"]       = [self.shadowRenderer.camera.eye[0],
                                                  self.shadowRenderer.camera.eye[1],
                                                  self.shadowRenderer.camera.eye[2]]
        self.program["u_light_intensity"]      = self.lightIntensity
        self.setLighting()
        self.setColorMap(name=self.cmapName)

        # Define a 'DepthBuffer' to render the shadowmap from the light
        # --------------------------------------
        self.buffer = FrameBuffer(color=self.program["shadowMap"],
                                  depth=gloo.RenderBuffer((self.shadowRenderer.shadowMapSize, self.shadowRenderer.shadowMapSize), format='depth'))

    def setLighting(self, lightType=None, param=None, val=None):
        """
        Modifies lighting dictionnary and updates program Attributes
        """
        if lightType is not None and param is not None and val is not None:
            # Check if val is of proper type by comparison with second arg in dic
            self.lightingDictionnary[lightType][param][0] = val
        elif lightType is not None and param == 'on' and val is None:
            self.lightingDictionnary[lightType][param][0] = not \
                self.lightingDictionnary[lightType][param][0]
        for first in self.lightingDictionnary.keys():
            for second in self.lightingDictionnary[first].keys():
                self.program["u_%s_%s"%(first, second)] = self.lightingDictionnary[first][second][0]

    def moveCamera(self, dAzimuth=0.0, dElevation=0.0, dDistance=0.0, dFov=0.0):
        """
        Moves the camera according to inputs.
        """
        super().moveCamera(dAzimuth, dElevation, dDistance, dFov)
        self.adjustShadowMapFrustum()

    def resetCamera(self):
        """
        Replaces the camera at its original position.
        """
        super().resetCamera()
        self.adjustShadowMapFrustum()

    def zoomCamera(self, percentage=0.0):
        """
        Zoom with camera according to inputs. Computes projection Matrix.
        """
        self.camera.setProjection(fov=self.camera.fov*(1+percentage))
        self.program["u_vm"]                  = self.camera.vm
        self.program["u_pvm"]                 = self.camera.pvm
        self.adjustShadowMapFrustum()

    def moveLight(self, dAzimuth=0.0, dElevation=0.0, dDistance=0.0):
        """
        Moves the light, and the camera attached for shadowmap rendering
        """
        self.shadowRenderer.moveCamera(dAzimuth, dElevation, dDistance)
        self.program["u_shadowmap_pvm"] = self.shadowRenderer.camera.pvm
        self.program["u_light_position"] = [self.shadowRenderer.camera.eye[0],
                                            self.shadowRenderer.camera.eye[1],
                                            self.shadowRenderer.camera.eye[2]]

    def resetLight(self, dAzimuth=0.0, dElevation=0.0, dDistance=0.0):
        """
        Moves the light, and the camera attached for shadowmap rendering
        """
        self.shadowRenderer.resetCamera()
        self.shadowRenderer.camera.zoomOn(objectWidth=sqrt(2.0), margin=0.02)
        self.program["u_shadowmap_pvm"] = self.shadowRenderer.camera.pvm
        self.program["u_light_position"] = [self.shadowRenderer.camera.eye[0],
                                            self.shadowRenderer.camera.eye[1],
                                            self.shadowRenderer.camera.eye[2]]

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
        Compute projection Matrix for new Window aspect.
        """
        self.camera.setProjection(aspect=aspect)
        self.program["u_vm"]                  = self.camera.vm
        self.program["u_pvm"]                 = self.camera.pvm

    def modifyLightCharacteristic(self, lightType=None, modification=None):
        """
        Modifies one of the light parameters.
        Some like ambient, diffuse, specular are simply toggles on/off,
        Others increase/decrease a value, such as the shininess exponant
        WIP One day could be entirely replaced by setLighting...
        """
        if lightType == 'shadow':
            secondKey = 'type'
            vals = self.lightingDictionnary[lightType][secondKey]
            vals[0] = (vals[0] + 1) % 3
            self.setLighting(lightType, 'type', vals[0])
            # print("shadowType = %s" % vals[0])
            # self.program["u_%s_%s" % (lightType, secondKey)] = vals[0]
            # print(' Shadows: %s              ' % self.SHADOW_TYPE[vals[0]], end="\r")

        elif lightType == 'shininess':
            mod = sqrt(2)
            if modification == '+':
                mod = 1.0/mod
            first = 'specular'
            vals = self.lightingDictionnary[first][lightType]
            vals[0] *= mod
            vals[0] = np.clip(vals[0], vals[2], vals[3])
            self.setLighting(lightType, first, vals[0])
            # self.program["u_%s_%s" % (first, lightType)] = vals[0]
            # print(' Shininess exponant: %3.0f' % vals[0], end="\r")

        elif lightType == 'fresnelexponant':
            mod = 1.0/sqrt(2)
            if modification == '+':
                mod = 1.0/mod
            first = 'lightbox'
            vals = self.lightingDictionnary[first][lightType]
            vals[0] *= mod
            vals[0] = np.clip(vals[0], vals[2], vals[3])
            self.setLighting(lightType, first, vals[0])
            # self.program["u_%s_%s" % (first, lightType)] = vals[0]
            # print(' Fresnel exponant: %3.0f' % vals[0], end="\r")

    def setColorMap(self, name=''):
        """
        Set the colormap used to render the concentration
        """
        if name != '':
            self.cmapName = name
            self.program["cmap"] = get_colormap(self.cmapName).map(np.linspace(0.0, 1.0, 4096)).astype('float32')
            print(' Using colormap %s.              ' % name, end="\r")

    def switchReagent(self):
        """
        Toggles between representing U or V concentrations.
        """
        super().switchReagent()
        self.shadowRenderer.switchReagent()
        print(' Displaying %s reagent concentration.' % MainRenderer.REAGENTS[self.reagent], end="\r")


class Canvas(app.Canvas):

    def __init__(self,
                 size=(1024, 1024),
                 modelSize=(512,512),
                 specie='alpha_left',
                 cmap='honolulu_r',
                 verbose=False,
                 isotropic=True,
                 mainAppAccess=False):
        app.Canvas.__init__(self,
                            size=size,
                            title='3D Gray-Scott Reaction-Diffusion - GregVDS',
                            keys='interactive')

        if not verbose:
            sys.stdout = open(os.devnull, 'w')

        # Link towards QMainWindow app if provided
        # --------------------------------------
        self.mainAppAccess = mainAppAccess

        # Create the Gray-Scott model
        # --------------------------------------
        # this contains the 3D grid model, texture and program
        self.grayScottModel = GrayScottModel(canvas=self,
                                             gridSize=modelSize,
                                             specie=specie,
                                             isotropic=isotropic)

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
        self.backgroundColor = (0.30, 0.30, 0.35, 1.00)

        # ? better computation ?
        # --------------------------------------
        gl.GL_FRAGMENT_PRECISION_HIGH = 1

        # OpenGL initialization
        # --------------------------------------
        gloo.set_state(clear_color=self.backgroundColor, depth_test=True,
                       line_width=0.75)

        self.activate_zoom()

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
        if self.mainRenderer.lightingDictionnary['shadow']['on'][0] is True:
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
                           clear_color=self.backgroundColor,
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
        # trying to update shown Field of View and Distance in Main App UI
        if self.mainAppAccess:
            try:
                self.mainAppAccess.fovSlider.updateParam()
                self.mainAppAccess.distSlider.updateParam()
            except Exception as e:
                print(e)

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
            # trying to update shown elevation and azimuth in Main App UI
            if self.mainAppAccess:
                try:
                    self.mainAppAccess.elevSlider.updateParam()
                    self.mainAppAccess.aziSlider.updateParam()
                except Exception as e:
                    print(e)

    NO_ACTION = (None, None)

    def on_key_press(self, event):
        """
        treats all key event that are defined in keyactionDictionnary
        WIP: ONe could avoid this method if self.canvas.native.parent() is not
        None, meaning there is a GUI outhere (PySide6/PyQT5) that has its own
        Key mapping...
        """
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
                if self.mainAppAccess:
                    try:
                        self.mainAppAccess.normalRadioButton.toggle()
                    except Exception as e:
                        print(e)
            elif hasattr(self.grayScottModel, func.__name__):
                func(self.grayScottModel, *args)
                if self.mainAppAccess:
                    try:
                        self.mainAppAccess.pearsonsPatternsComboBox.setCurrentText(self.grayScottModel.specie)
                    except Exception as e:
                        print(e)
            elif hasattr(self.mainRenderer, func.__name__):
                func(self.mainRenderer, *args)
                if self.mainAppAccess:
                    try:
                        self.mainAppAccess.colorsComboBox.setCurrentText(self.mainRenderer.cmapName)
                    except Exception as e:
                        print(e)
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

    def setBackgroundColor(self, color=None):
        """
        Set Canvas background color.
        """
        self.backgroundColor = color or self.backgroundColor

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
        ('P', ('Control',)): (GrayScottModel.increaseCycle, ()),
        ('O', ('Control',)): (GrayScottModel.decreaseCycle, ()),
        ('/', ('Shift',)): (MainRenderer.switchReagent, ()),
        ('@', ()): (MainRenderer.resetCamera, ()),
        ('<', ()): (MainRenderer.resetLight, ()),
        (',', ('Control',)): (MainRenderer.setLighting, ('ambient', 'on')),
        (';', ('Control',)): (MainRenderer.setLighting, ('diffuse', 'on')),
        (':', ('Control',)): (MainRenderer.setLighting, ('specular', 'on')),
        ('=', ('Control',)): (MainRenderer.setLighting, ('shadow', 'on')),
        ('N', ('Control',)): (MainRenderer.setLighting, ('attenuation', 'on')),
        ('L', ('Control',)): (MainRenderer.modifyLightCharacteristic, ('shininess', '-')),
        ('M', ('Control',)): (MainRenderer.modifyLightCharacteristic, ('shininess', '+')),
        ('J', ('Control',)): (MainRenderer.modifyLightCharacteristic, ('fresnelexponant', '-')),
        ('K', ('Control',)): (MainRenderer.modifyLightCharacteristic, ('fresnelexponant', '+')),
        ('I', ('Control',)): (MainRenderer.setLighting, ('lightbox', 'on'))
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
        Calls static method to harvest all __doc__
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
