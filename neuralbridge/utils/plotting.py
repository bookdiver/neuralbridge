import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection

plt.style.use("dark_background")

from neuralbridge.setups import *
from neuralbridge.utils.sample_path import SamplePath

def plot_sample_path(
    sample_path: SamplePath, 
    ax: Optional[plt.Axes] = None, 
    color: Optional[Union[str, Sequence[str]]] = "grey",
    alpha: Optional[float] = 0.7,
    linewidth: Optional[float] = 1.0,
    linestyle: Optional[str] = "-",
    label: Optional[Union[str, Sequence[str]]] = None,
    title: Optional[Union[str, None]] = None,
    save_path: Optional[Union[str, None]] = None,
):
    if ax is None:
        fig, ax = plt.subplots(layout="constrained")
    else:
        fig = ax.figure
        
    xs = sample_path.xs
    ts = sample_path.ts[len(sample_path.ts) - xs.shape[1]:]
    dim = xs.shape[-1]
    
    if isinstance(color, str):
        colors = [color] * dim
    else:
        assert len(color) == dim, "Number of colors must match the dimension of the sample path"
        colors = color
    
    for j in range(dim):
        ax.plot(ts, xs[:, :, j].T, color=colors[j], alpha=alpha, linewidth=linewidth, linestyle=linestyle)
    
    if label is not None:
        if isinstance(label, str):
            labels = [label] * dim
        else:
            assert len(label) == dim, "Number of labels must match the dimension of the sample path"
            labels = label
        
        for j in range(dim):
            ax.plot([], [], color=colors[j], alpha=alpha, linewidth=linewidth, linestyle=linestyle, label=labels[j])
        ax.legend()
    
    ax.set_xlim(ts.min(), ts.max())
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$x$')
    
    if title is not None:
        ax.set_title(title)
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
    
    return ax

def plot_sample_path_histogram(
    sample_path: SamplePath,
    dim: int = 0,
    ax: Optional[plt.Axes] = None,
    cmap: Optional[str] = "plasma",
    vertical_bins: Optional[int] = 100,
    norm: Optional[str] = "log",
):  
    if ax is None:
        fig, ax = plt.subplots(layout="constrained")
    else:
        fig = ax.figure
    cmap = plt.get_cmap(cmap)
    cmap = cmap.with_extremes(bad=cmap(0))
    xs = sample_path.xs
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
    
    return ax

def plot_landmark_sample_path(
    sample_path: SamplePath,
    m_landmarks: int,
    ax: Optional[plt.Axes] = None,
    cmap: Optional[str] = "viridis",
    alpha: Optional[float] = 0.7,
    linewidth: Optional[float] = 1.0,
    title: Optional[str] = None,
):
    if ax is None:
        if m_landmarks == 3:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, layout="constrained")
        else:
            fig, ax = plt.subplots(layout="constrained")
    else:
        fig = ax.figure
    
    xs = sample_path.xs
    ts = sample_path.ts[len(sample_path.ts) - xs.shape[1]:]
    xs_landmarks = rearrange(xs, "b t (n m) -> b t n m", m=m_landmarks)
    
    cmap = plt.get_cmap(cmap)
    colors = cmap(np.linspace(0, 1, len(ts)))
    
    if m_landmarks == 1:
        x0s = xs_landmarks[:, 0, :, 0]
        xTs = xs_landmarks[:, -1, :, 0]
        for i in range(xs_landmarks.shape[0]):
            ax.scatter(jnp.zeros(xs_landmarks.shape[2]), x0s[i], color=colors[0], marker="o", s=50)
            ax.scatter(jnp.ones(xs_landmarks.shape[2]) * ts[-1], xTs[i], color=colors[-1], marker="*", s=50)
        
        # Add 1D colored trajectories
        for i in range(xs_landmarks.shape[0]):
            for j in range(xs_landmarks.shape[2]):
                points = jnp.array([ts, xs_landmarks[i, :, j, 0]]).T.reshape(-1, 1, 2)
                segments = jnp.concatenate([points[:-1], points[1:]], axis=1)
                norm = plt.Normalize(ts.min(), ts.max())
                lc = LineCollection(segments, cmap=cmap, norm=norm, alpha=alpha)
                
                lc.set_array(ts)
                lc.set_linewidth(linewidth)
                ax.add_collection(lc)
                
        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'$X_t$')
        x_max, x_min = xs_landmarks[..., 0].max(), xs_landmarks[..., 0].min()
        t_max, t_min = ts.max(), ts.min()
        x_padding = 0.2 * (x_max - x_min)
        t_padding = 0.2 * (t_max - t_min)
        ax.set_xlim(t_min - t_padding, t_max + t_padding)
        ax.set_ylim(x_min - x_padding, x_max + x_padding)
        
        fig.colorbar(lc, ax=ax, label=r'$time$', pad=0)
    
    elif m_landmarks == 2:
        ax.scatter(*xs_landmarks[:, 0, :].T, color=colors[0], marker="o", s=50)
        ax.scatter(*xs_landmarks[:, -1, :].T, color=colors[-1], marker="*", s=50)
        
        # Plot trajectories for each sample
        for i in range(xs_landmarks.shape[0]):
            for j in range(xs_landmarks.shape[2]): 
                x = xs_landmarks[i, :, j, 0]
                points = jnp.array([x, xs_landmarks[i, :, j, 1]]).T.reshape(-1, 1, 2)
                segments = jnp.concatenate([points[:-1], points[1:]], axis=1)
                norm = plt.Normalize(ts.min(), ts.max())
                lc = LineCollection(segments, cmap=cmap, norm=norm, alpha=alpha)
                
                lc.set_array(ts)
                lc.set_linewidth(linewidth)
                ax.add_collection(lc)
                
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$') 
        x_max, x_min = xs_landmarks[..., 0].max(), xs_landmarks[..., 0].min()
        y_max, y_min = xs_landmarks[..., 1].max(), xs_landmarks[..., 1].min()
        x_padding = 0.2 * (x_max - x_min)
        y_padding = 0.2 * (y_max - y_min)
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
        
        fig.colorbar(lc, ax=ax, label=r'$time$', pad=0)
    
    elif m_landmarks == 3:
        ax.scatter(*xs_landmarks[:, 0, :].T, color=colors[0], marker="o", s=50)
        ax.scatter(*xs_landmarks[:, -1, :].T, color=colors[-1], marker="*", s=50)
        
        # Plot trajectories for each sample
        for i in range(xs_landmarks.shape[0]):
            for j in range(xs_landmarks.shape[2]): 
                points = xs_landmarks[i, :, j].reshape(-1, 1, 3)
                segments = jnp.concatenate([points[:-1], points[1:]], axis=1)
                norm = plt.Normalize(ts.min(), ts.max())
                lc = Line3DCollection(segments, cmap=cmap, norm=norm, alpha=alpha)
                
                lc.set_array(ts)
                lc.set_linewidth(linewidth)
                ax.add_collection(lc)
        
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_zlabel(r'$z$')
        
        x_max, x_min = xs_landmarks[..., 0].max(), xs_landmarks[..., 0].min()   
        y_max, y_min = xs_landmarks[..., 1].max(), xs_landmarks[..., 1].min()   
        z_max, z_min = xs_landmarks[..., 2].max(), xs_landmarks[..., 2].min()   
        x_padding = 0.2 * (x_max - x_min)
        y_padding = 0.2 * (y_max - y_min)
        z_padding = 0.2 * (z_max - z_min)
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
        ax.set_zlim(z_min - z_padding, z_max + z_padding)
        
        fig.colorbar(lc, ax=ax, label=r'$time$', location='bottom', pad=0.05, shrink=0.5)
    
    else:
        raise ValueError(f"Unsupported number of landmark dimensions: {m_landmarks}")
    
    if title is not None:
        ax.set_title(title)
    
    return ax
