# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Temporal plots (matplotlib only, no optional deps)
from .temporal_plots import plot_lag_correlations, plot_peak_latencies, plot_response_curves

# Brain surface visualization requires nilearn/pyvista (optional).
# Import lazily to avoid breaking users who only need temporal_plots.
try:
    from .base import BasePlotBrain
    from .cortical import PlotBrainNilearn
    from .cortical_pv import PlotBrainPyvista
    from .subcortical import get_subcortical_roi_indices, plot_subcortical
    from .utils import (
        combine_mosaics,
        convert_ax_to_2d,
        convert_ax_to_3d,
        get_cmap,
        get_pval_stars,
        label_ax,
        move_ax,
        plot_colorbar,
        plot_rgb_colorbar,
        saturate_colors,
        set_title,
        shrink_ax,
    )

    PlotBrain = PlotBrainPyvista
except ImportError:
    pass
