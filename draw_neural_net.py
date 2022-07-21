import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def draw_neural_net(
    ax,
    left,
    right,
    bottom,
    top,
    layer_sizes,
    input_prefix,
    hidden_prefix,
    output_prefix,
    fontsize,
    node_kws={"color": "w", "ec": "k"},
    edge_kws={"color": "k", "lw": 1},
):
    """
    Draw a neural network cartoon using matplotilb.
    https://stackoverflow.com/questions/67279657/drawing-a-neural-network
    
    :usage:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])
    
    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
    """
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2.0 + (top + bottom) / 2.0
        for m in range(layer_size):
            circle = plt.Circle(
                (n * h_spacing + left, layer_top - m * v_spacing),
                v_spacing / 4.0,
                zorder=4,
                **node_kws,
            )
            ax.add_artist(circle)
            if n == 0:  # input layer
                ax.annotate(
                    f"${input_prefix}^{m}$",
                    xy=(n * h_spacing + left, layer_top - m * v_spacing),
                    ha="center",
                    va="center",
                    zorder=5,
                    fontsize=fontsize,
                )
            if (n >= 1) & (n < n_layers - 1):  # hidden layers
                ax.annotate(
                    f"${hidden_prefix}_{m+1}^{n}$",
                    xy=(n * h_spacing + left, layer_top - m * v_spacing),
                    ha="center",
                    va="center",
                    zorder=5,
                    fontsize=fontsize,
                )
            if n == n_layers - 1:  # output layer
                ax.annotate(
                    f"${output_prefix}_{m+1}$",
                    xy=(n * h_spacing + left, layer_top - m * v_spacing),
                    ha="center",
                    va="center",
                    zorder=5,
                    fontsize=fontsize,
                )
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(
        zip(layer_sizes[:-1], layer_sizes[1:])
    ):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2.0 + (top + bottom) / 2.0
        layer_top_b = v_spacing * (layer_size_b - 1) / 2.0 + (top + bottom) / 2.0
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D(
                    [n * h_spacing + left, (n + 1) * h_spacing + left],
                    [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing],
                    **edge_kws,
                )
                ax.add_artist(line)


def draw_weighted_neural_net(
    ax,
    left,
    right,
    bottom,
    top,
    layer_sizes,
    input_prefix,
    hidden_prefix,
    output_prefix,
    fontsize,
    node_wts=None,
    edge_wts=None,
    cmap=None,
):
    """
    Draw a neural network cartoon using matplotilb.
    https://stackoverflow.com/questions/67279657/drawing-a-neural-network
    
    :usage:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])
    
    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
    """
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)

    # ensure weights are correct dimensions
    use_node_wts = False
    if node_wts is not None:
        use_node_wts = True
        if check_weights(node_wts, layer_sizes):
            nvmin, nvmax = get_vlim(node_wts, layer_sizes)

    use_edge_wts = False
    if edge_wts is not None:
        use_edge_wts = True
        if check_weights(edge_wts, layer_sizes):
            evmin, evmax = get_vlim(edge_wts, layer_sizes)

    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2.0 + (top + bottom) / 2.0
        if n > 0 and use_node_wts:
            node_wt = np.sum(node_wts[n - 1].T, axis=0)
        for m in range(layer_size):
            if n > 0 and use_node_wts:
                _, color = get_rgb(node_wt[m], cmap=cmap, vmin=nvmin, vmax=nvmax)
            else:
                color = "w"
            circle = plt.Circle(
                (n * h_spacing + left, layer_top - m * v_spacing),
                v_spacing / 4.0,
                zorder=4,
                ec="k",
                color=color,
            )
            ax.add_artist(circle)
            if n == 0:  # input layer
                ax.annotate(
                    f"${input_prefix}^{m}$",
                    xy=(n * h_spacing + left, layer_top - m * v_spacing),
                    ha="center",
                    va="center",
                    zorder=5,
                    fontsize=fontsize,
                )
            if (n >= 1) & (n < n_layers - 1):  # hidden layers
                ax.annotate(
                    f"${hidden_prefix}_{m+1}^{n}$",
                    xy=(n * h_spacing + left, layer_top - m * v_spacing),
                    ha="center",
                    va="center",
                    zorder=5,
                    fontsize=fontsize,
                )
            if n == n_layers - 1:  # output layer
                ax.annotate(
                    f"${output_prefix}_{m+1}$",
                    xy=(n * h_spacing + left, layer_top - m * v_spacing),
                    ha="center",
                    va="center",
                    zorder=5,
                    fontsize=fontsize,
                )
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(
        zip(layer_sizes[:-1], layer_sizes[1:])
    ):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2.0 + (top + bottom) / 2.0
        layer_top_b = v_spacing * (layer_size_b - 1) / 2.0 + (top + bottom) / 2.0
        if use_edge_wts:
            edge_wt = edge_wts[n].T
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                if use_edge_wts:
                    _, color = get_rgb(edge_wt[m, o], cmap=cmap, vmin=evmin, vmax=evmax)
                    lw = np.abs(edge_wt[m, o]) + 1.0  # + 0.5
                else:
                    color = "k"
                    lw = 1
                line = plt.Line2D(
                    [n * h_spacing + left, (n + 1) * h_spacing + left],
                    [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing],
                    color=color,
                    lw=lw,
                )
                ax.add_artist(line)


def get_rgb(value, cmap, vmin, vmax):
    """get cmap rbg value normalized to [vmin, vmax]"""
    cmap = mpl.cm.get_cmap(cmap)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    rgb = sm.to_rgba(value)
    sm.set_array([])
    return sm, rgb


def get_vlim(weights, layer_sizes):
    """return vmin/max for cmap normalization"""
    vmin, vmax = [], []
    n_layers = len(layer_sizes)
    for i in range(n_layers - 1):
        vmin.append(np.min(weights[i]))
        vmax.append(np.max(weights[i]))
    return np.min(vmin), np.max(vmax)


def check_weights(weights, layer_sizes):
    """assert weight matricies are correct dimensions"""
    check = False
    n_layers = len(layer_sizes)
    assert len(weights) == n_layers - 1
    for i in range(n_layers - 1):
        assert weights[i].shape == (layer_sizes[i + 1], layer_sizes[i])
        check = True
    return check
