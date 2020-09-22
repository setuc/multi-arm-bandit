from textwrap import wrap
import os.path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pickle import dump as pickle_dump
from datetime import datetime

import numpy as np
import seaborn as sns

DPI = 120  #: DPI to use for the figures
FIGSIZE = (16, 9)  #: Figure size, in inches!
FORMATS = ('png', 'pdf')

# Customize the colormap
HLS = True  #: Use the HLS mapping, or HUSL mapping
VIRIDIS = False  #: Use the Viridis colormap

# Bbox in inches. Only the given portion of the figure is saved. If 'tight', try to figure out the tight bbox of the figure.
BBOX_INCHES = "tight"  #: Use this parameter for bbox
BBOX_INCHES = None

signature = ""

def palette(nb, hls=HLS, viridis=VIRIDIS):
    """ Use a smart palette from seaborn, for nb different plots on the same figure.
    - Ref: http://seaborn.pydata.org/generated/seaborn.hls_palette.html#seaborn.hls_palette
    """
    if viridis:
        return sns.color_palette('viridis', nb)
    else:
        return sns.hls_palette(nb + 1)[:nb] if hls else sns.husl_palette(nb + 1)[:nb]

def maximizeWindow():
    """ Experimental function to try to maximize a plot.
    - Tries as well as possible to maximize the figure.
    - Cf. https://stackoverflow.com/q/12439588/
    .. warning:: This function is still experimental, but "it works on my machine" so I keep it.
    """
    # plt.show(block=True)
    # plt.tight_layout()
    figManager = plt.get_current_fig_manager()
    try:
        figManager.window.showMaximized()
    except Exception:
        try:
            figManager.frame.Maximize(True)
        except Exception:
            try:
                figManager.window.state('zoomed')  # works fine on Windows!
            except Exception:
                try:
                    figManager.full_screen_toggle()
                except Exception:
                    print("  Note: Unable to maximize window...")
                    # plt.show()


def show_and_save(showplot=True, savefig=None, formats=FORMATS, pickleit=False, fig=None):
    """ Maximize the window if need to show it, save it if needed, and then show it or close it.
    - Inspired by https://tomspur.blogspot.fr/2015/08/publication-ready-figures-with.html#Save-the-figure
    """
    if showplot:
        maximizeWindow()
    if savefig is not None:
        if pickleit and fig is not None:
            form = "pickle"
            path = "{}.{}".format(savefig, form)
            print("Saving raw figure with format {}, to file '{}'...".format(form, path))  # DEBUG
            with open(path, "bw") as f:
                pickle_dump(fig, f)
            print("       Saved! '{}' created of size '{}b', at '{:%c}' ...".format(path, os.path.getsize(path), datetime.fromtimestamp(os.path.getatime(path))))
        for form in formats:
            path = "{}.{}".format(savefig, form)
            print("Saving figure with format {}, to file '{}'...".format(form, path))  # DEBUG
            try:
                plt.savefig(path, bbox_inches=BBOX_INCHES)
                print("       Saved! '{}' created of size '{}b', at '{:%c}' ...".format(path, os.path.getsize(path), datetime.fromtimestamp(os.path.getatime(path))))
            except Exception as exc:
                print("Error: could not save current figure to {} because of error {}... Skipping!".format(path, exc))  # DEBUG
    try:
        plt.show(block=True) if showplot else plt.close()
    except (TypeError, AttributeError):
        print("Failed to show the figure for some unknown reason...")  # DEBUG