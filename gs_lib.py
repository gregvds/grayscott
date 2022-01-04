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
