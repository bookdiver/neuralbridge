from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection

import scienceplots

from neuralbridge.setups import *
from neuralbridge.utils.sample_path import SamplePath

plt.style.use(["science", "grid"])
plt.rc('axes', prop_cycle=plt.cycler(color=DEFAULT_COLOR_WHEELS))

def plot_sample_path(
    sample_path: SamplePath, 
    ax: Optional[plt.Axes] = None, 
    plot_dims: Optional[Union[int, Sequence[int]]] = None,
    colors: Optional[Union[str, Sequence[str]]] = None,
    alpha: Optional[float] = 0.7,
    linewidth: Optional[float] = 1.0,
    linestyle: Optional[str] = "-",
    zorder: Optional[int] = 1,
    label: Optional[Union[str, Sequence[str]]] = None,
    title: Optional[Union[str, None]] = None,
):
    if ax is None:
        fig, ax = plt.subplots(layout="constrained")
    else:
        fig = ax.figure
        
    xs = sample_path.xs
    ts = sample_path.ts[len(sample_path.ts) - xs.shape[1]:]
    plot_dims = range(xs.shape[-1]) if plot_dims is None else plot_dims
    n_plot_dims = len(plot_dims)
    
    if colors is None:
        colors = [f'C{i}' for i in range(n_plot_dims)] 
    elif isinstance(colors, str):
        colors = [colors] * n_plot_dims
    else:
        assert len(colors) == n_plot_dims, "Number of colors must match the dimension of the sample path"
        colors = colors
    
    for i, j in enumerate(plot_dims):
        ax.plot(ts, xs[:, :, j].T, color=colors[i], alpha=alpha, linewidth=linewidth, linestyle=linestyle)
    
    if label is not None:
        if isinstance(label, str):
            labels = [label] * n_plot_dims
        else:
            assert len(label) == n_plot_dims, "Number of labels must match the dimension of the sample path"
            labels = label
        
        for i in range(n_plot_dims):
            ax.plot([], [], color=colors[i], alpha=alpha, linewidth=linewidth, linestyle=linestyle, label=labels[i], zorder=zorder)
    
    if title is not None:
        ax.set_title(title)
    
    return ax

def plot_mcmc_sample_path(
    mcmc_sample_path_logs: List[SamplePath], 
    sample_index: Optional[int] = 0,
    plot_dims: Optional[Union[int, Sequence[int]]] = None,
    ax: Optional[plt.Axes] = None, 
    cmaps: Optional[Union[str, Sequence[str]]] = None,
    alpha: Optional[float] = 0.7,
    linewidth: Optional[float] = 1.0,
    linestyle: Optional[str] = "-",
    label: Optional[Union[str, Sequence[str]]] = None,
    title: Optional[Union[str, None]] = None,
    n_iters: Optional[int] = 1000,
):
    if ax is None:
        fig, ax = plt.subplots(layout="constrained")
    else:
        fig = ax.figure
        
    xs = jnp.stack([sample_path.path[sample_index].xs for sample_path in mcmc_sample_path_logs])
    ts = mcmc_sample_path_logs[0].path[sample_index].ts[len(mcmc_sample_path_logs[0].path[sample_index].ts) - xs.shape[2]:]
    n_logs = xs.shape[0]
    
    if plot_dims is None:
        dims_to_plot = range(xs.shape[-1])
        dim = xs.shape[-1]
    else:
        dims_to_plot = [plot_dims] if isinstance(plot_dims, int) else plot_dims
        dim = len(dims_to_plot)
    
    if cmaps is None:
        cmaps = [plt.get_cmap(c) for c in DEFAULT_CMAP_WHEELS]
    elif isinstance(cmaps, str):
        cmaps = [plt.get_cmap(cmaps)] * dim
    else:
        assert len(cmaps) == dim, "Number of colors must match the dimension of the sample path"
        cmaps = [plt.get_cmap(c) for c in cmaps]
    
    colors = [cmaps[k](jnp.linspace(0, 1, n_logs)) for k in range(len(cmaps))]
    for k, d in enumerate(dims_to_plot):
        for i in range(n_logs):
            ax.plot(ts, xs[i, :, :, d].T, color=colors[k][i], alpha=alpha, linewidth=linewidth, linestyle=linestyle)
    
    if label is not None:
        if isinstance(label, str):
            labels = [label] * dim
        else:
            assert len(label) == dim, "Number of labels must match the dimension of the sample path"
            labels = label
        
        for k in range(len(dims_to_plot)):
            ax.plot([], [], color=colors[k][-1], alpha=alpha, linewidth=linewidth, linestyle=linestyle, label=labels[k])
    
    # Add colorbars for each dimension
    for k, d in enumerate(dims_to_plot):
        norm = plt.Normalize(0, n_iters)
        sm = plt.cm.ScalarMappable(cmap=cmaps[k], norm=norm)
        sm.set_array([])
        label_text = f"Component {d+1}" if label is None else labels[k]
        plt.colorbar(sm, ax=ax, label=f"{label_text} iterations", pad=0.01 + 0.03*k)
    
    if title is not None:
        ax.set_title(title)
    
    return ax

def plot_sample_path_histogram(
    sample_path: SamplePath,
    plot_dim: int = 0,
    ax: Optional[plt.Axes] = None,
    cmap: Optional[str] = "plasma",
    vertical_bins: Optional[int] = 100,
    norm: Optional[str] = "log",
    title: Optional[str] = None,
):  
    if ax is None:
        fig, ax = plt.subplots(layout="constrained")
    else:
        fig = ax.figure
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    else:
        cmap = cmap
    cmap = cmap.with_extremes(bad=cmap(0))
    xs = sample_path.xs
    n_samples, n_steps, _ = xs.shape
    xs = xs[..., plot_dim]
    ts = sample_path.ts[len(sample_path.ts) - n_steps:]
    ts_fine = np.linspace(ts.min(), ts.max(), 2000, endpoint=False)
    xs_fine = np.concatenate([
        np.interp(ts_fine, ts, xs_row) for xs_row in xs
    ])
    ts_fine = np.broadcast_to(ts_fine, (n_samples, 2000)).ravel()
    h, t_edges, x_edges = np.histogram2d(ts_fine, xs_fine, bins=[n_steps, vertical_bins])
    pcm = ax.pcolormesh(t_edges, x_edges, h.T, cmap=cmap, norm=norm, vmax=n_samples/10., rasterized=True)
    fig.colorbar(pcm, ax=ax, label="no. of samples", pad=0.05)
    
    ax.grid(False)
    
    if title is not None:
        ax.set_title(title)
    
    return ax

def plot_landmark_sample_path(
    sample_path: SamplePath,
    m_landmarks: int,
    ax: Optional[plt.Axes] = None,
    cmap: Optional[str] = "viridis",
    alpha: Optional[float] = 0.7,
    markersize: Optional[float] = 10,
    show_intermediate_trajectories: Optional[bool] = True,
    show_intermediate_shapes: Optional[bool] = False,
    show_colorbar: Optional[bool] = True,
    show_every: Optional[int] = 1,
    linewidth: Optional[float] = 1.0,
    title: Optional[str] = None,
    
):
    
    def close_curve(x):
        # x (n_landmarks, m_landmarks)
        # x_closed (n_landmarks + 1, m_landmarks)
        x_closed = jnp.concatenate([x, x[0:1, :]], axis=0)
        return x_closed
    
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
    
    if m_landmarks == 2:
        start_xs, end_xs = xs_landmarks[:, 0, :], xs_landmarks[:, -1, :]

        # Plot trajectories for each sample
        if show_intermediate_trajectories:
            for i in range(xs_landmarks.shape[0]):
                start_x_closed = close_curve(start_xs[i, :, :])
                end_x_closed = close_curve(end_xs[i, :, :])
                ax.plot(*start_x_closed.T, '-o', color=colors[0], alpha=1.0, linewidth=linewidth, markersize=markersize, label=r'$X_0$')
                ax.plot(*end_x_closed.T, '-o', color=colors[-1], alpha=1.0, linewidth=linewidth, markersize=markersize, label=r'$X_T$')
                for j in range(xs_landmarks.shape[2])[::show_every]: 
                    x = xs_landmarks[i, :, j, 0]
                    points = jnp.array([x, xs_landmarks[i, :, j, 1]]).T.reshape(-1, 1, 2)
                    segments = jnp.concatenate([points[:-1], points[1:]], axis=1)
                    norm = plt.Normalize(ts.min(), ts.max())
                    mappable = LineCollection(segments, cmap=cmap, norm=norm, alpha=alpha)
                    
                    mappable.set_array(ts)
                    mappable.set_linewidth(linewidth)
                    ax.add_collection(mappable)
                ax.legend()
                    
        if show_intermediate_shapes:
            for i in range(xs_landmarks.shape[0]):
                for j in range(xs_landmarks.shape[1])[::show_every]:
                    x = xs_landmarks[i, j, :, :]
                    x_closed = close_curve(x)
                    ax.plot(*x_closed.T, color=colors[j], alpha=alpha, linewidth=linewidth)
            
            norm = plt.Normalize(ts.min(), ts.max())
            mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            mappable.set_array([])
            
        if show_colorbar:
            fig.colorbar(mappable, ax=ax, label=r'$t$', pad=0)
            
                
        # x_max, x_min = xs_landmarks[..., 0].max(), xs_landmarks[..., 0].min()
        # y_max, y_min = xs_landmarks[..., 1].max(), xs_landmarks[..., 1].min()
        # x_padding = 0.2 * (x_max - x_min)
        # y_padding = 0.2 * (y_max - y_min)
        # ax.set_xlim(x_min - x_padding, x_max + x_padding)
        # ax.set_ylim(y_min - y_padding, y_max + y_padding)
        
    elif m_landmarks == 3:
        ax.scatter(*xs_landmarks[:, 0, :].T, color=colors[0], marker="o", s=markersize)
        ax.scatter(*xs_landmarks[:, -1, :].T, color=colors[-1], marker="*", s=markersize)
        
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
    
    if not show_colorbar:
        return ax, mappable
    else:
        return ax