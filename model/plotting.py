"""
Utility functions for plotting

Author: Victor Shnayder <shnayder@gmail.com>
"""

import math
import matplotlib.pyplot as plt
import seaborn as sns

# Default figure dir -- empirics chapter
FIG_DIR = "/Users/shnayder/dissertation/tex/thesis/figures/ch3"

FIG_TITLE_SIZE = 14.0
DEFAULT_FONT_SIZE = 11.0

DEFAULT_ALPHA = 0.7

GOLDEN_RATIO = (math.sqrt(5)-1.0)/2.0
LATEX_TEXT_WIDTH_INCHES = 5.75
LATEX_MAX_HEIGHT_INCHES = 8.0

LINESTYLES = ['solid','dashed','dotted','dashdot']

def set_plotting_defaults():
    # Set seaborn defaults
    sns.set_style('white')
    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})

    plt.rcParams.update({'axes.labelsize': DEFAULT_FONT_SIZE,
                         'axes.titlesize': DEFAULT_FONT_SIZE,
                         'axes.linewidth': 0.5,

                         'xtick.labelsize': DEFAULT_FONT_SIZE,
                         'ytick.labelsize': DEFAULT_FONT_SIZE,

                         'legend.fontsize': DEFAULT_FONT_SIZE,

                         'font.size': DEFAULT_FONT_SIZE,
                         'font.family': 'Times New Roman',
#                         'font.serif': 'Computer Modern Roman',
                         'text.usetex': False,
                         'figure.figsize': (LATEX_TEXT_WIDTH_INCHES,
                                            LATEX_TEXT_WIDTH_INCHES*GOLDEN_RATIO),},
                         )

def set_width(fig, size, fig_width=None, fig_height=None,
              height_ratio=1):
    """
    Sets size so fonts will work out in print.

    Based on width parameters above.

    Args:
        fig: matplotlib Figure
        size: one of "full", "half", "third", "fourth", or a fraction between 0 and 1,
            to specify a custom multiplier of LATEX_TEXT_WIDTH
        if fig_width and/or fig_height specified, uses them.
             Can use same str values as size for fig_height.
        If not, uses golden ratio * height_ratio
    """
    # ratios of full page width
    WIDTHS = {"full": 1.0,
              "half": 0.49,
              "third": 0.32,
              "fourth": 0.24}

    if fig_width is None:
        mult = size if isinstance(size, float) else WIDTHS[size]
        fig_width = mult * LATEX_TEXT_WIDTH_INCHES

    if fig_height is None:
        # height in inches
        fig_height = fig_width * GOLDEN_RATIO * height_ratio

    # Allow same string values for fig_height
    if isinstance(fig_height, str):
        fig_height = WIDTHS[fig_height] * LATEX_TEXT_WIDTH_INCHES

    if fig_height > LATEX_MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large: {} so will reduce to {} inches.".format(
                   fig_height, LATEX_MAX_HEIGHT_INCHES))
        fig_height = LATEX_MAX_HEIGHT_INCHES

    fig.set_size_inches(fig_width, fig_height)


def save_figure(fig, filename, fig_dir=FIG_DIR):
    """
    Save fig in fig_dir/filename, with bbox_inches=tight
    """
    fig.tight_layout()
    fig.savefig(fig_dir + "/" + filename , bbox_inches='tight')
