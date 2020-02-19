#!/usr/bin/env python

import matplotlib
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm

FONT_SIZE = 16
LABEL_FONT_SIZE = 12
COLOR_MAP="jet"

matplotlib.rc('font', **{'size': FONT_SIZE})

# got this from http://www.somersault1824.com/tips-for-designing-scientific-figures-for-color-blind-readers/
bw_readable_color_palette = [(   0,   0,   0), # 0
                             (   0,  73,  73),
                             (   0, 146, 146),
                             ( 255, 109, 182),
                             ( 255, 182, 119),
                             ( 146,   0,   0), # 10
                             ( 146,  73,   0),
                             ( 219, 209,   0),
                             (  36, 255,  36),
                             ( 255, 255, 109),
                             ( 255, 255, 255), # 15
                             ( 153, 153, 153),
                             ( 102, 102, 102),
                             (  51,  51,  51),
                             (   0,   0,   0),
                             (  73,   0, 146), # 5
                             (   0, 109, 219),
                             ( 182, 109, 255),
                             ( 109, 182, 255),
                             ( 182, 219, 255),
                             ]
bw_readable = [(c[0]/255., c[1]/255., c[2]/255., 1.) for c in bw_readable_color_palette]

# colormap from 7yr ps paper
ps_map_colors =  {'blue' : ((0.0, 0.0, 1.0),
                            (0.05, 1.0, 1.0),
                            (0.4, 1.0, 1.0),
                            (0.6, 1.0, 1.0),
                            (0.7, 0.2, 0.2),
                            (1.0, 0.0, 0.0)),
                  'green': ((0.0, 0.0, 1.0),
                            (0.05, 1.0, 1.0),
                            (0.5, 0.0416, 0.0416),
                            (0.6, 0.0, 0.0),
                            (0.8, 0.5, 0.5),
                            (1.0, 1.0, 1.0)),
                  'red':   ((0.0, 0.0, 1.0),
                            (0.05, 1.0, 1.0),
                            (0.5, 0.0416, 0.0416),
                            (0.6, 0.0416, 0.0416),
                            (0.7, 1.0, 1.0),
                            (1.0, 1.0, 1.0))}
ps_map = matplotlib.colors.LinearSegmentedColormap('ps_map', ps_map_colors, 256)


def add_text_relative(ax, text, x_rel, y_rel, ancor=None):
    """ adds text to ax at relative position x_rel, y_rel """

    if ancor is None:
        vert_orient = "top" if y_rel > 0.5 else "bottom"
        hori_orient = "right" if x_rel > 0.5 else "left"
    else:
        vert_orient = ancor.split("_")[0]
        hori_orient = ancor.split("_")[1]

    x_log = ax.get_xaxis().get_scale() == "log"
    y_log = ax.get_yaxis().get_scale() == "log"
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    xx = 10**(np.log10(x0) + x_rel*(np.log10(x1)-np.log10(x0))) if x_log else x0 + x_rel * (x1-x0)
    yy = 10**(np.log10(y0) + y_rel*(np.log10(y1)-np.log10(y0))) if y_log else y0 + y_rel * (y1-y0)

    ax.text(xx, yy, text, horizontalalignment=hori_orient, verticalalignment=vert_orient)

def plot_with_ratio():
    """ creates at figure with two subplots
    ax0 is the main plot and ax1 a smaller one below like for pull plots

    Returns fig, ax0, ax1
    """

    left, width = 0.1, 0.85

    rect_upper = [left, 0.35 , width, 0.6]
    rect_lower = [left, 0.1, width, 0.2]

    fig = plt.figure()

    ax0 = plt.axes(rect_upper)
    ax1 = plt.axes(rect_lower)

    return fig, ax0, ax1

def plot_side_by_side():
    """ creates at figure with two subplots
    ax0 and ax1 are same size and side by side

    Returns fig, ax0, ax1
    """

    left, width = 0.05, 0.4
    bottom, height = 0.1, .8

    rect_left = [left, bottom, width, height]
    rect_right = [left+width+0.1, bottom, width, height]

    fig = plt.figure(figsize=(16,8))

    ax0 = plt.axes(rect_left)
    ax1 = plt.axes(rect_right)

    return fig, ax0, ax1

def colorCycle():
    """ iterator returning colors"""

    i=0
    colors = bw_readable
    while(True):
        yield colors[i%len(colors)]
        i += 1

class AnyObject(object):
    def __init__(self, colors=colorCycle(), line=False, **kwargs):
        self.colors = colors
        self.line = line
        self.kwargs = kwargs

class AnyObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        colors = orig_handle.colors
        step = 0.5/len(colors)
        patches = []
        for i, c in enumerate(colors):
            j = len(colors)-i
            patch = mpatches.Rectangle([x0, y0+height*(0.5-j*step)], width, height*2*j*step,
                                       facecolor=c, lw=0, transform=handlebox.get_transform())
            patches.append(patch)
        if orig_handle.line:
            print("will draw the line")
            print(orig_handle.kwargs)
            patches.append( mlines.Line2D([x0, x0+width], [y0+0.5*height, y0+0.5*height], **(orig_handle.kwargs)) )

        for patch in patches:
            handlebox.add_artist(patch)

        return patches


class base_plot(Axes):

    def save(self, savepath):
        if not os.path.exists(os.path.dirname(savepath)):
            raise IOError("Directory does not exists where plot should be saved! You gave the savepath: {}".format(savepath))
        self.fig.savefig(savepath.replace(".png", ".pdf"), bbox_inches='tight')
        self.fig.savefig(savepath.replace(".pdf", ".png"), bbox_inches='tight', dpi=200)
        self.fig.close()
