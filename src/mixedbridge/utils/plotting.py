from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt

from mixedbridge.utils.sample_path import SamplePath

def plot_sample_path(
    sample_path: SamplePath, 
    plot_object: str = "xs",
    ax: plt.Axes = None, 
    colors: Tuple[str, ...] = ("grey",), 
    alpha: float = 0.7,
    linewidth: float = 1.0,
    linestyle: str = "-",
    label: str = None
):
    if ax is None:
        fig, ax = plt.subplots(layout="constrained")
    else:
        fig = ax.figure
    n_samples = sample_path.n_samples
    xs = sample_path.path[plot_object]
    ts = sample_path.ts[len(sample_path.ts) - xs.shape[1]:]
    dim = xs.shape[-1]
    assert len(colors) == dim, "Number of colors must match the dimension of the sample path"
    for j in range(dim):
        ax.plot(ts, xs[:, :, j].T, color=colors[j], alpha=alpha, linewidth=linewidth, linestyle=linestyle, label=label)
    
    if label is not None:
        ax.legend()
    
    ax.set_xlim(ts.min(), ts.max())
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$x$')
    ax.set_title(sample_path.name)
    
    return ax

def plot_sample_path_histogram(
    sample_path: SamplePath,
    plot_object: str = "xs",
    dim: int = 0,
    ax: plt.Axes = None,
    cmap: str = "plasma",
    vertical_bins: int = 100,
    norm: str = "log",
):  
    if ax is None:
        fig, ax = plt.subplots(layout="constrained")
    else:
        fig = ax.figure
    cmap = plt.get_cmap(cmap)
    cmap = cmap.with_extremes(bad=cmap(0))
    xs = sample_path.path[plot_object]
    n_samples, n_steps, _ = xs.shape
    xs = xs[..., dim]
    ts = sample_path.ts[len(sample_path.ts) - n_steps:]
    ts_fine = np.linspace(ts.min(), ts.max(), 1000)
    xs_fine = np.concatenate([
        np.interp(ts_fine, ts, xs_row) for xs_row in xs
    ])
    ts_fine = np.broadcast_to(ts_fine, (n_samples, 1000)).ravel()
    h, t_edges, x_edges = np.histogram2d(ts_fine, xs_fine, bins=[n_steps, vertical_bins])
    pcm = ax.pcolormesh(t_edges, x_edges, h.T, cmap=cmap, norm=norm, vmax=n_samples/10., rasterized=True)
    fig.colorbar(pcm, ax=ax, label="# samples", pad=0)
    
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$x$')
    ax.set_title(sample_path.name)
    
    return ax
