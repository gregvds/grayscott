# -*- coding: utf-8 -*-

################################################################################

import vispy.color as color

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import numpy as np
from numpy.random import rand

from systems import pearson

################################################################################

materials = {
    'Brass': {
        'ambient': (0.329412, 0.223529, 0.027451, 1.0),
        'diffuse': (0.780392, 0.568627, 0.113725, 1.0),
        'specular': (0.992157, 0.941176, 0.807843, 1.0),
        'shininess': 27.8974},
    'Bronze': {
        'ambient': (0.2125, 0.1275, 0.054, 1.0),
        'diffuse': (0.714, 0.4284, 0.18144, 1.0),
        'specular': (0.393548, 0.271906, 0.166721, 1.0),
        'shininess': 25.6},
    'Polished Bronze': {
        'ambient': (0.25, 0.148, 0.06475, 1.0),
        'diffuse': (0.4, 0.2368, 0.1036, 1.0),
        'specular': (0.774597, 0.458561, 0.200621, 1.0),
        'shininess': 76.8},
    'Chrome': {
        'ambient': (0.25, 0.25, 0.25, 1.0),
        'diffuse': (0.4, 0.4, 0.4, 1.0),
        'specular': (0.774597, 0.774597, 0.774597, 1.0),
        'shininess': 76.8},
    'Copper': {
        'ambient': (0.19125, 0.0735, 0.0225, 1.0),
        'diffuse': (0.7038, 0.27048, 0.0828, 1.0),
        'specular': (0.256777, 0.137622, 0.086014, 1.0),
        'shininess': 12.8},
    'Polished Copper': {
        'ambient': (0.2295, 0.08825, 0.0275, 1.0),
        'diffuse': (0.5508, 0.2118, 0.066, 1.0),
        'specular': (0.580594, 0.223257, 0.0695701, 1.0),
        'shininess': 51.2},
    'Gold': {
        'ambient': (0.24725, 0.1995, 0.0745, 1.0),
        'diffuse': (0.75164, 0.60648, 0.22648, 1.0),
        'specular': (0.628281, 0.555802, 0.366065, 1.0),
        'shininess': 51.2},
    'Polished Gold': {
        'ambient': (0.24725, 0.2245, 0.0645, 1.0),
        'diffuse': (0.34615, 0.3143, 0.0903, 1.0),
        'specular': (0.797357, 0.723991, 0.208006, 1.0),
        'shininess': 83.2},
    'Pewter': {
        'ambient': (0.105882, 0.058824, 0.113725, 1.0),
        'diffuse': (0.427451, 0.470588, 0.541176, 1.0),
        'specular': (0.333333, 0.333333, 0.521569, 1.0),
        'shininess': 9.84615},
    'Silver': {
        'ambient': (0.19225, 0.19225, 0.19225, 1.0),
        'diffuse': (0.50754, 0.50754, 0.50754, 1.0),
        'specular': (0.508273, 0.508273, 0.508273, 1.0),
        'shininess': 51.2},
    'Polished Silver': {
        'ambient': (0.23125, 0.23125, 0.23125, 1.0),
        'diffuse': (0.2775, 0.2775, 0.2775, 1.0),
        'specular': (0.773911, 0.773911, 0.773911, 1.0),
        'shininess': 89.6},
    'Emerald': {
        'ambient': (0.0215, 0.1745, 0.0215, 0.55),
        'diffuse': (0.07568, 0.61424, 0.07568, 0.55),
        'specular': (0.633, 0.727811, 0.633, 0.55),
        'shininess': 76.8},
    'Jade': {
        'ambient': (0.135, 0.2225, 0.1575, 0.95),
        'diffuse': (0.54, 0.89, 0.63, 0.95),
        'specular': (0.316228, 0.316228, 0.316228, 0.95),
        'shininess': 12.8},
    'Obsidian': {
        'ambient': (0.05375, 0.05, 0.06625, 0.82),
        'diffuse': (0.18275, 0.17, 0.22525, 0.82),
        'specular': (0.332741, 0.328634, 0.346435, 0.82),
        'shininess': 38.4},
    'Pearl': {
        'ambient': (0.25, 0.20725, 0.20725, 0.922),
        'diffuse': (1.0, 0.829, 0.829, 0.922),
        'specular': (0.296648, 0.296648, 0.296648, 0.922),
        'shininess': 11.264},
    'Ruby': {
        'ambient': (0.1745, 0.01175, 0.01175, 0.55),
        'diffuse': (0.61424, 0.04136, 0.04136, 0.55),
        'specular': (0.727811, 0.626959, 0.626959, 0.55),
        'shininess': 76.8},
    'Turquoise': {
        'ambient': (0.1, 0.18725, 0.1745, 0.8),
        'diffuse': (0.396, 0.74151, 0.69102, 0.8),
        'specular': (0.297254, 0.30829, 0.306678, 0.8),
        'shininess': 12.8},
    'Black Plastic': {
        'ambient': (0.0, 0.0, 0.0, 1.0),
        'diffuse': (0.01, 0.01, 0.01, 1.0),
        'specular': (0.50, 0.50, 0.50, 1.0),
        'shininess': 32},
    'Black Rubber': {
        'ambient': (0.02, 0.02, 0.02, 1.0),
        'diffuse': (0.01, 0.01, 0.01, 1.0),
        'specular': (0.4, 0.4, 0.4, 1.0),
        'shininess': 10}
    }

linearSegmentedColormaps = {
    'detroit': {
        'colorList': [
            'white',
            'lightblue',
            'cadetblue',
            'black',
            'cadetblue',
            'lightblue',
            'white'
        ],
        'nodes' : [0.0, 0.4, 0.47, 0.5, 0.53, 0.6, 1.0]
    },
    'antidetroit': {
        'colorList': [
            'black',
            'cadetblue',
            'lightblue',
            'white',
            'lightblue',
            'cadetblue',
            'black'
        ],
        'nodes': [0.0, 0.4, 0.47, 0.5, 0.53, 0.6, 1.0]
    },
    'uppsala': {
        'colorList': [
            'slategrey',
            'goldenrod',
            'antiquewhite',
            'olivedrab',
            'lawngreen',
            'lightseagreen',
            'paleturquoise'
        ],
        'nodes': [0.0, 0.4, 0.46, 0.5, 0.58, 0.75, 1.0]
    },
    'oslo': {
        'colorList': [
            'black',
            'saddlebrown',
            'sienna',
            'white',
            'aliceblue',
            'skyblue'
        ],
        'nodes': [0.0, 0.48, 0.5, 0.51, 0.57, 1.0]
    },
    'Lochinver': {
        'colorList': [
            'floralwhite',
            'antiquewhite',
            'wheat',
            'cornflowerblue',
            'dodgerblue',
            'powderblue'
        ],
        'nodes': [0.0, 0.4, 0.45, 0.5, 0.65, 1.0]
    },
    'tromso': {
        'colorList': [
            'black',
            'black',
            'chocolate',
            'ivory',
            'sandybrown',
            'black',
            'black'
        ],
        'nodes': [0.0, 0.3, 0.47, 0.5, 0.53, 0.7, 1.0]
    },
    'osmort': {
        'colorList': [
            'linen',
            'linen',
            'chocolate',
            'black',
            'sandybrown',
            'linen',
            'linen'
        ],
        'nodes': [0.0, 0.4, 0.47, 0.5, 0.53, 0.6, 1.0]
    },
    'irkoutsk': {
        'colorList': [
            'darkkhaki',
            'olive',
            'dimgray',
            'seashell',
            'yellowgreen',
            'darkolivegreen',
            'olivedrab'
        ],
        'nodes': [0.0, 0.4, 0.47, 0.5, 0.53, 0.6, 1.0]
    },
    'krasnoiarsk': {
        'colorList': [
            'darkgoldenrod',
            'goldenrod',
            'dimgray',
            'cornsilk',
            'gold',
            'tan',
            'burlywood'
        ],
        'nodes': [0.0, 0.4, 0.47, 0.5, 0.53, 0.6, 1.0]
    },
    'Rejkjavik': {
        'colorList': [
            'rebeccapurple',
            'indigo',
            'darkorchid',
            'gold',
            'darkorange',
            'orange',
            'gold'
        ],
        'nodes': [0.0, 0.3, 0.47, 0.5, 0.53, 0.7, 1.0]
    },
    'hs': {
        'colorList': [
            'black',
            '#100926',
            'grey',
            'gold',
            'lemonchiffon',
            'azure'
        ],
        'nodes': [0.0, 0.2, 0.5, 0.6, 0.7, 1.0]
    },
    'vancouver': {
        'colorList': [
            (0.21785201149425293, 0.14180336580598935, 0.07745451176207403),
            (0.5810032894736842, 0.3098060344827587, 0.09019607843137259),
            (0.9607843137254902, 0.7535179093567251, 0.1373922413793104),
            (0.749019607843137, 0.8666666666666667, 0.43137254901960786),
            (0.39999999999999997, 0.6392156862745098, 0.41568627450980394),
            (0.023529411764706683, 0.4274509803921568, 0.08627450980392157),
            (0.5195761494252873, 0.8178453947368421, 0.6258979885057472)
        ],
        'nodes': [0.0,
                  0.10933388740335086,
                  0.19388647416804353,
                  0.39529263589967106,
                  0.4332937984905443,
                  0.7230526632459519,
                  1.0]
    },
    'papetee': {
        'colorList': [
            (0.0826456394103434, 0.507539047252722, 0.657507183908046),
            (0.23650574982326966, 0.670131908645612, 0.7839439655172413),
            (0.1657986111111111, 0.6230244252873562, 0.3988864942528736),
            (0.39999999999999997, 0.6392156862745098, 0.41960784313725497),
            (0.8734009502923977, 0.9879669540229884, 0.8213002873563219),
            (0.11024305555555555, 0.6979623538011696, 0.7322198275862069),
            (0.10588235294117743, 0.3763249269005848, 0.6057830459770115),
            (0.15117872807017543, 0.9477370689655172, 0.9506106321839081)
        ],
        'nodes': [0.0,
                  0.12247507774253703,
                  0.25139712822840715,
                  0.29562586087684434,
                  0.3492649621738852,
                  0.392552657957462,
                  0.6259298004428328,
                  1.0]
    },
    'honolulu': {
        'colorList': [
            (0.054902, 0.109804, 0.121569),
            (0.113725, 0.4, 0.278431),
            (0.082353, 0.588235, 0.082353),
            (0.556863, 0.8, 0.239216),
            (1, 0.984314, 0.901961),
            (0.988235, 0.988235, 0.870588),
            (0.85098, 0.8, 0.466667),
            (0.588235, 0.529412, 0.384314),
            (0.301961, 0.266667, 0.239216)
        ],
        'nodes': [0.0,
                  0.0823838379686773,
                  0.1647676759373546,
                  0.2471515139060322,
                  0.2883434328903706,
                  0.2883434328903706,
                  0.5018404030232599,
                  0.7865030298671107,
                  1.0]
    }
}

stackedColormaps = {
    'Boston': {
        'colorMapList': [
            'bone_r',
            'gray',
            'pink_r',
            'YlOrBr_r'],
        'proportions': [960, 160, 160, 960]
    },
    'malmo': {
        'colorMapList': [
            'bone_r',
            'Blues_r',
            'GnBu_r',
            'Greens'],
        'proportions': [512, 128, 512, 128]
    }
}


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
    try:
        _ = cm.get_cmap(name)
    except:
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
    try:
        _ = cm.get_cmap(name)
    except:
        cm.register_cmap(name=name, cmap=newcmp)
        cm.register_cmap(name=invertCmapName(name), cmap=newcmp_r)


def createColormaps():
    # definition of some custom colormaps
    for colorMapName in stackedColormaps.keys():
        colormaps = stackedColormaps[colorMapName]['colorMapList']
        proportions = stackedColormaps[colorMapName]['proportions']
        createAndRegisterCmap(colorMapName, colormaps, proportions)

    for colormapName in linearSegmentedColormaps.keys():
        colors = linearSegmentedColormaps[colormapName]['colorList']
        nodes = linearSegmentedColormaps[colormapName]['nodes']
        createAndRegisterLinearSegmentedCmap(colormapName, colors, nodes)


def plot_color_gradients(cmap_category, cmap_list):
    # Create figure and adjust figure height to number of colormaps
    nrows = len(cmap_list)
    figh = 0.35 + 0.15 + (nrows + (nrows-1)*0.1)*0.22
    fig, axs = plt.subplots(nrows=nrows, figsize=(6.4, figh))
    fig.subplots_adjust(top=1-.35/figh, bottom=.15/figh, left=0.2, right=0.99)

    axs[0].set_title(cmap_category + ' colormaps', fontsize=14)
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    for ax, cmap_name in zip(axs, cmap_list):
        ax.imshow(gradient, aspect='auto', cmap=cmap_name)
        ax.text(-.01, .5, cmap_name, va='center', ha='right', fontsize=10,
                transform=ax.transAxes)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axs:
        ax.set_axis_off()


def import_pearsons_types():
    species = {}
    fMin = 1.0
    fMax = 0.0
    kMin = 1.0
    kMax = 0.0
    type = None
    for each in pearson.keys():
        type = pearson[each]
        species[each] = [type['factors']['Da'],
                         type['factors']['Db'],
                         type['factors']['f'],
                         type['factors']['k'],
                         type['description']]
        fMin = min(fMin, type['factors']['f'])
        fMax = max(fMax, type['factors']['f'])
        kMin = min(kMin, type['factors']['k'])
        kMax = max(kMax, type['factors']['k'])
    return (species, fMin, fMax, kMin, kMax)


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


def createLightBox(size=(1024, 1024)):
    """
    DEBUG method that create a lighbox
    """
    lightBox = np.zeros((6, 1024, 1024, 4), dtype=np.float32)
    lightBox[2] = np.ones((1024, 1024, 4), dtype=np.float32) * np.array((0,0,0,1)) #DOWN
    lightBox[3] = np.ones((1024, 1024, 4), dtype=np.float32) * np.array((1,1,1,1)) #UP
    lightBox[0] = np.ones((1024, 1024, 4), dtype=np.float32) * np.array((0,1,0,1)) #LEFT
    lightBox[1] = np.ones((1024, 1024, 4), dtype=np.float32) * np.array((1,0,0,1)) #RIGHT
    lightBox[4] = np.ones((1024, 1024, 4), dtype=np.float32) * np.array((0,0,1,1)) #BACK
    lightBox[5] = np.ones((1024, 1024, 4), dtype=np.float32) * np.array((1,1,0,1)) #FRONT

    return lightBox
