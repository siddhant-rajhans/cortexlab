# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import re
from functools import reduce

import colorcet
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def robust_normalize(
    array, axis=None, percentile=99, clip=True, final_range=None, two_sided=True
):
    """Normalize the input array using statistics robust to outliers."""
    hi = np.percentile(array, percentile, axis=axis, keepdims=True)
    if two_sided:
        lo = np.percentile(array, 100 - percentile, axis=axis, keepdims=True)
    else:
        lo = np.min(array, axis=axis, keepdims=True)
    out = (array - lo) / (hi - lo)
    if clip:
        out = np.clip(out, 0, 1)
    if final_range is not None:
        if final_range == "original":
            final_range = (lo, hi)
        out = out * (final_range[1] - final_range[0]) + final_range[0]
    return out


def get_scalar_mappable(
    data,
    cmap,
    vmin=None,
    vmax=None,
    symmetric_cbar=False,
    threshold=None,
    alpha_cmap=None,
):
    vmin = vmin if vmin is not None else np.nanmin(data)
    vmax = vmax if vmax is not None else np.nanmax(data)
    if symmetric_cbar:
        vmin, vmax = -vmax, vmax
    sm = get_thresholded_sm(vmin, vmax, threshold=threshold, cmap=cmap)
    return sm


def get_thresholded_sm(vmin, vmax, threshold=None, cmap=None):

    if cmap is None:
        cmap = matplotlib.cm.get_cmap("hot")
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmaplist = [cmap(i) for i in range(cmap.N)]

    # set colors to gray for absolute values < threshold
    if threshold is not None:
        istart = int(norm(-threshold, clip=True) * (cmap.N - 1))
        istop = int(norm(threshold, clip=True) * (cmap.N - 1))
        for i in range(istart, istop):
            cmaplist[i] = (0.5, 0.5, 0.5, 1.0)
    our_cmap = LinearSegmentedColormap.from_list("Custom cmap", cmaplist, cmap.N)
    sm = plt.cm.ScalarMappable(cmap=our_cmap, norm=norm)

    # fake up the array of the scalar mappable.
    sm._A = []

    return sm


def get_pval_stars(pval: float):
    if pval < 0.0005:
        return "***"
    elif pval < 0.005:
        return "**"
    elif pval < 0.05:
        return "*"
    else:
        return ""


def saturate_colors(rgb: np.ndarray, factor: float):
    """
    rgb: tuple/list/array of (R, G, B) in 0-1 range
    factor: >1 boosts saturation, 1 leaves unchanged, 0 makes gray
    """
    rgb = np.array(rgb, dtype=float)

    # Compute luminance (perceptual gray)
    # Using Rec.709 coefficients for a fairly natural grayscale
    grayscale_coeffs = np.array([0.2126, 0.7152, 0.0722])
    if rgb.ndim == 1:
        lum = np.dot(grayscale_coeffs, rgb)
    elif rgb.ndim == 2:
        lum = np.dot(grayscale_coeffs, rgb.T)
        lum = lum[:, None].repeat(3, axis=1)
    else:
        raise ValueError(f"Invalid number of dimensions: {rgb.ndim}")

    # Pull or push the channels relative to gray
    new_rgb = lum + factor * (rgb - lum)

    # Clamp to 0–1
    new_rgb = np.clip(new_rgb, 0, 1)
    return new_rgb


def get_alpha_cmap(cmap, threshold: float = 0, scale: float = 1, symmetric=False):
    """
    Takes a cmap and makes it transparent below a threshold.
    Transparency is linearly scaled between threshold and threshold + scale.
    """
    assert 0 <= threshold <= 1
    from matplotlib.colors import ListedColormap

    n_points = 1024
    new_cmap = cmap(np.linspace(0, 1, n_points))
    alpha = np.zeros_like(new_cmap[:, 3])
    # zeros before min, ramp 0 to 1 between min and max, 1 after max
    min_idx = int(threshold * (n_points - 1))
    max_idx = int((threshold + scale) * (n_points - 1))
    ramp = np.linspace(0, 1, max_idx - min_idx)
    alpha[min_idx : min(max_idx, n_points)] = ramp[: min(max_idx, n_points) - min_idx]
    alpha[min(max_idx, n_points) :] = 1
    # alpha[max_idx:] = 1
    if symmetric:
        alpha = np.concatenate([alpha[::-2], alpha[::2]])
    new_cmap[:, 3] = alpha
    new_cmap = ListedColormap(new_cmap)
    return new_cmap


def get_cmap(
    cmap_name: str | matplotlib.colors.Colormap,
    alpha_cmap: tuple[float, float] | None = None,
):
    if isinstance(cmap_name, str):
        cmap = (
            getattr(matplotlib.cm, cmap_name, None)
            or getattr(sns.cm, cmap_name, None)
            or getattr(colorcet.cm, cmap_name, None)
        )
    else:
        cmap = cmap_name
    if not cmap:
        raise ValueError(f"Invalid cmap: {cmap}")
    if alpha_cmap is not None:
        threshold, scale = alpha_cmap
        cmap = get_alpha_cmap(
            cmap,
            threshold=threshold,
            scale=scale,
            symmetric=(cmap_name in ["seismic", "bwr"]),
        )
    return cmap


def convert_ax_to_3d(ax):
    if hasattr(ax, "view_init"):
        return ax
    pos = ax.get_position()
    # subplotspec = ax.get_subplotspec()
    ax3d = ax.figure.add_axes(pos, projection="3d")
    # ax3d.set_position(pos)
    ax.remove()
    return ax3d


def convert_ax_to_2d(ax):
    pos = ax.get_position()
    ax2d = ax.figure.add_axes(pos)
    ax.remove()
    return ax2d


def lcm(a, b):
    return a * b // math.gcd(a, b) if a and b else max(a, b)


def _lcm_list(lst):
    return reduce(lcm, lst, 1)


def _repeat_chars(line, times):
    return "".join(c * times for c in line)


def _transpose(block):
    if not block:
        return []
    max_len = max(len(row) for row in block)
    block = [row.ljust(max_len) for row in block]
    return ["".join(block[r][c] for r in range(len(block))) for c in range(max_len)]


def _check_unique_letters(*blocks):
    """
    Ensure all blocks have unique letters across blocks.
    Raises an AssertionError if any letter appears in more than one block.
    """
    unique = set()
    for i, block in enumerate(blocks, 1):
        letters = set(block.replace("\n", ""))
        assert not (
            letters & unique
        ), f"Duplicate letters found in block {i}: {letters & unique}"
        unique.update(letters)


def _format_block(mosaic: str) -> str:
    return mosaic.replace(" ", "").lstrip("\n").rstrip("\n")


def combine_mosaics(*blocks, ratio=None, orient="v"):

    if len(blocks) < 2:
        raise ValueError("Need at least two blocks to combine")

    _check_unique_letters(*blocks)
    blocks = [_format_block(block) for block in blocks]

    # Normalize input
    blocks_lines = [block.split("\n") for block in blocks]

    # Normalize ratio
    if ratio is None:
        ratios = [1.0] * len(blocks_lines)
    else:
        try:
            ratios = list(ratio)
            if len(ratios) != len(blocks_lines):
                raise ValueError
        except Exception:
            ratios = [float(ratio)] * len(blocks_lines)

    # Transpose if horizontal
    transposed = False
    if orient == "v":
        blocks_lines = [_transpose(b) for b in blocks_lines]
        transposed = True

    # Horizontal expansion (columns)
    cols_list = [max(len(line) for line in b) if b else 0 for b in blocks_lines]
    Lw = _lcm_list(cols_list)
    blocks_expanded = []
    for b, c, r in zip(blocks_lines, cols_list, ratios):
        b = [line.ljust(c) for line in b]
        h = max(1, int(round(Lw / c * r)))
        blocks_expanded.append([_repeat_chars(line, h) for line in b])

    # Vertical expansion (rows)
    rows_list = [len(b) for b in blocks_expanded]
    Lh = _lcm_list(rows_list)
    blocks_tiled = []
    for b, r in zip(blocks_expanded, ratios):
        v = max(1, int(round(Lh / len(b))))
        blocks_tiled.append([line for line in b for _ in range(v)])

    # Combine all blocks
    combined = ["".join(lines) for lines in zip(*blocks_tiled)]

    # Transpose back if needed
    if transposed:
        combined = _transpose(combined)

    return _format_block("\n".join(combined))


def plot_colorbar(
    ax,
    sm=None,
    cmap=colorcet.cm.fire,
    vmin=0,
    vmax=1,
    label="R",
    label_orientation="vertical",
    orientation="vertical",
    **kwargs,
):

    # Hide the axis background, ticks, and spines
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Create a ScalarMappable for the colorbar
    if sm is None:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # Required for colorbar

    # Draw the colorbar inside the given axis
    cbar = plt.colorbar(sm, cax=ax, orientation=orientation, **kwargs)

    # Set the label if provided
    if label is not None:
        rotation = 0 if label_orientation == "horizontal" else 90
        cbar.set_label(label, rotation=rotation, labelpad=5)

    # Add border by setting the color and linewidth of all spines
    rect = matplotlib.patches.Rectangle(
        (0, 0),
        1,
        1,
        transform=cbar.ax.transAxes,
        fill=False,
        edgecolor="k",
        linewidth=0.5,
        clip_on=False,
    )
    # cbar.ax.add_patch(rect)
    return cbar


def shrink_ax(ax, shrink=0.1, horizontally=True, vertically=True):
    pos = ax.get_position()
    # shrink from all sides
    horizontal_shrink = pos.width * shrink if horizontally else 0
    vertical_shrink = pos.height * shrink if vertically else 0
    new_pos = [
        pos.x0 + horizontal_shrink / 2,
        pos.y0 + vertical_shrink / 2,
        pos.width - horizontal_shrink,
        pos.height - vertical_shrink,
    ]
    ax.set_position(new_pos)


def move_ax(ax, x=0, y=0):
    pos = ax.get_position()
    up = y * pos.height
    right = x * pos.width
    new_pos = [
        pos.x0 + right,
        pos.y0 + up,
        pos.width,
        pos.height,
    ]
    ax.set_position(new_pos)


def label_ax(
    ax,
    label,
    x_offset=0,
    y_offset=0.03,
    fontsize=14,
    fontweight="bold",
    facecolor="none",
    edgecolor="none",
):
    pos = ax.get_position()
    fig = ax.get_figure()
    fig.text(
        pos.x0 + x_offset,
        pos.y1 + y_offset,
        label,
        fontsize=fontsize,
        fontweight=fontweight,
        ha="center",
        va="center",
    )


def set_title(axes, title, x_offset=0, y_offset=0, **kwargs):
    if not isinstance(axes, list):
        axes = [axes]
    centers = [(ax.get_position().x0 + ax.get_position().x1) / 2 for ax in axes]
    x = np.mean(centers)
    x = x + x_offset
    y = axes[0].get_position().y1 + y_offset
    fig = axes[0].get_figure()
    if not "ha" in kwargs:
        kwargs["ha"] = "center"
    if not "va" in kwargs:
        kwargs["va"] = "top"
    fig.text(x, y, title, **kwargs)


def tight_crop(img, bg_color=(255, 255, 255), tol=5, w_pad=0, h_pad=0):
    if img.shape[2] == 4:  # alpha channel exists
        alpha = img[..., 3]
        ys, xs = np.where(alpha > 0)
    else:
        bg = np.array(bg_color)
        mask = np.any(np.abs(img[..., :3] - bg) > tol, axis=2)
        ys, xs = np.where(mask)

    if len(xs) == 0:
        return img  # nothing found
    left, right, bottom, top = xs.min(), xs.max(), ys.min(), ys.max()
    w_pad = int(w_pad * (right - left))
    h_pad = int(h_pad * (top - bottom))
    left, bottom = max(0, left - w_pad), max(0, bottom - h_pad)
    right, top = min(img.shape[1], right + w_pad), min(img.shape[0], top + h_pad)

    return img[bottom : top + 1, left : right + 1]


def plot_rgb_colorbar(n_cubes=4, alpha=1, labels=["Text", "Audio", "Video"]):
    # Use a dark background to make the colors pop
    # plt.style.use('dark_background')
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection="3d", proj_type="persp", focal_length=0.15)

    x = np.linspace(0, 1, n_cubes)
    y = np.linspace(0, 1, n_cubes)
    z = np.linspace(0, 1, n_cubes)
    X, Y, Z = np.meshgrid(x, y, z)
    X, Y, Z = np.ravel(X), np.ravel(Y), np.ravel(Z)
    colors = np.array([X, Y, Z]).T

    size = 0.2

    for i in range(len(X)):
        ax.bar3d(
            X[i] - size / 2,
            Y[i] - size / 2,
            Z[i] - size / 2,
            size,
            size,
            size,
            color=colors[i],
            alpha=alpha,
            edgecolor="none",
        )

    # --- AXIS ARROWS (QUINVERS) ---
    # We extend the arrows past the data (to 1.4) to show direction clearly
    arrow_props = dict(arrow_length_ratio=0.1, linewidth=1, pivot="tail")
    ax.quiver(0, 0, 0, 1.4, 0, 0, color="k", **arrow_props)
    ax.quiver(0, 0, 0, 0, 1.4, 0, color="k", **arrow_props)
    ax.quiver(0, 0, 0, 0, 0, 1.4, color="k", **arrow_props)

    # --- LABELS ---
    # Positioning labels at the tips of the arrows
    pos = 1.5
    ax.text(pos, 0, 0, labels[0], color="red", fontweight="bold", ha="center", va="top")
    ax.text(
        0, pos, 0, labels[1], color="green", fontweight="bold", ha="center", va="top"
    )
    ax.text(0, 0, pos, labels[2], color="blue", fontweight="bold", ha="center")

    # Remove all background clutter
    ax.set_axis_off()
    ax.set_facecolor((0, 0, 0, 0))  # Transparent pane

    # view_init: Azimuth -45 degrees keeps the origin cube (black) at the front
    # ax.view_init(elev=-40, azim=-135)
    ax.view_init(elev=45, azim=-135 + 180)
    ax.set_box_aspect(None, zoom=0.85)

    return fig


def get_rainbow_brain(mesh="fsaverage5", hemi="both"):
    import matplotlib.colors as mcolors
    from nilearn.datasets import fetch_surf_fsaverage
    from nilearn.surface import load_surf_mesh

    fsaverage = fetch_surf_fsaverage(mesh=mesh)
    sphere_l, _ = load_surf_mesh(fsaverage["sphere_left"])
    sphere_r, _ = load_surf_mesh(fsaverage["sphere_right"])
    if hemi == "both":
        coords = np.concatenate([sphere_l, sphere_r], axis=0)
    else:
        coords = sphere_l if hemi == "left" else sphere_r
    x, y, z = coords.T

    # SYMMETRY LOGIC:
    # On fsaverage, +x is Right, -x is Left.
    # To make them symmetric, we take the absolute value of X
    # or flip the X for the right hemisphere so that 'lateral' is always
    # the same direction relative to the color wheel.
    x_mapped = x if hemi == "left" else -x

    # Hue based on Longitude (using the corrected X)
    phi = np.arctan2(y, x_mapped)
    hues = (phi + np.pi) / (2 * np.pi)

    # Value based on Elevation (Z) to make it more distinct
    # (Optional: adds a slight brightness gradient from bottom to top)
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
    vals = np.clip(0.8 + (z_norm * 0.3), 0, 1)

    hsv = np.stack([hues, np.ones_like(hues) * 0.9, vals], axis=1)
    return mcolors.hsv_to_rgb(hsv)


# ---------------------------------------------------------------------------
# Segment helpers (moved from analyses/utils.py)
# ---------------------------------------------------------------------------


def has_video(segment) -> bool:
    return any(e.__class__.__name__ == "Video" for e in segment.ns_events)


def has_audio(segment) -> bool:
    return any(e.__class__.__name__ == "Audio" for e in segment.ns_events)


def get_clip(segment, start_offset=0, stop_offset=0):
    from moviepy import VideoFileClip

    if not has_video(segment):
        return None
    video = [e for e in segment.ns_events if e.__class__.__name__ == "Video"][0]
    clip = VideoFileClip(video.filepath)
    true_start = video.start - video.offset
    clip = clip.subclipped(
        max(segment.start + start_offset - true_start, 0),
        min(segment.stop + stop_offset - true_start, clip.duration),
    )
    return clip


def get_audio(segment, start_offset=0, stop_offset=0):
    from moviepy import AudioFileClip

    if not has_audio(segment):
        return None
    audio = [e for e in segment.ns_events if e.__class__.__name__ == "Audio"][0]
    clip = AudioFileClip(audio.filepath)
    true_start = audio.start - audio.offset
    clip = clip.subclipped(
        max(segment.start + start_offset - true_start, 0),
        min(segment.stop + stop_offset - true_start, clip.duration),
    )
    return clip


def get_words(segment, filter=(0, 1), remove_punctuation=True, remove_stopwords=False):
    start, duration = segment.start, segment.duration
    clean = (
        (lambda x: re.sub(r"[^\w\s]", "", x)) if remove_punctuation else (lambda x: x)
    )
    words = [
        clean(e.text.lower())
        for e in segment.ns_events
        if e.__class__.__name__ == "Word"
        and filter[0] <= (e.start - start) / duration <= filter[1]
    ]
    if remove_stopwords:
        from stopwords import get_stopwords

        words = [w for w in words if w not in get_stopwords("english")]
    return words


def get_text(segment, **kwargs) -> str:
    return " ".join(get_words(segment, **kwargs))


if __name__ == "__main__":
    fig = plot_rgb_colorbar()
    plt.show()
