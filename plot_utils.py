"""
@File         : plot_utils.py
@Author(s)    : Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime  : 2025-08-23 15:28:02
@Email        : cuixuanstephen@gmail.com
@Description  :
"""

# import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as col
import numpy as np
from matplotlib.axes import Axes


def plot_2d_data(
    ax,
    X,
    y,
    s=20,
    alpha=0.95,
    xlabel=None,
    ylabel=None,
    title=None,
    legend=None,
    colormap="viridis",
):
    n_examples, n_features = X.shape
    if n_features != 2:
        raise ValueError("Data set is not 2D!")
    if n_examples != len(y):
        raise ValueError("Length of X is not equal to the length of y!")

    unique_labels = np.sort(np.unique(y))
    n_classes = len(unique_labels)

    markers = ["o", "s", "^", "v", "<", ">", "p"]
    cmap = plt.get_cmap(colormap)
    colors = cmap(np.linspace(0, 0.5, num=n_classes))

    if isinstance(s, np.ndarray):
        if len(s) != n_examples:
            raise ValueError("Length of s is not equal to the length of y!")
    else:
        s = np.full_like(y, fill_value=s)

    for i, label in enumerate(unique_labels):
        marker_color = col.rgb2hex(colors[i])
        marker_shape = markers[i % len(markers)]
        ax.scatter(
            X[y == label, 0],
            X[y == label, 1],
            s=s[y == label],
            marker=marker_shape,
            c=marker_color,
            edgecolor="k",
            alpha=alpha,
        )
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    if title:
        ax.set_title(title)
    if legend:
        ax.legend(legend, fontsize=12)


def plot_2d_classifier(
    ax: Axes,
    X,
    y,
    predict_function,
    predict_args=None,
    predict_proba=False,
    boundary_level=0.5,
    s=20,
    plot_data=True,
    alpha=0.75,
    xlabel=None,
    ylabel=None,
    title=None,
    legend=None,
    colormap="viridis",
):
    xMin, xMax = X[:, 0].min() - 0.25, X[:, 0].max() + 0.25
    yMin, yMax = X[:, 1].min() - 0.25, X[:, 1].max() + 0.25
    xMesh, yMesh = np.meshgrid(np.arange(xMin, xMax, 0.05), np.arange(yMin, yMax, 0.05))

    if predict_proba:
        zMesh = predict_function(np.c_[xMesh.ravel(), yMesh.ravel()])[:, 1]
    elif predict_args is None:
        zMesh = predict_function(np.c_[xMesh.ravel(), yMesh.ravel()])
    else:
        zMesh = predict_function(np.c_[xMesh.ravel(), yMesh.ravel()], predict_args)
    zMesh = zMesh.reshape(xMesh.shape)

    ax.contourf(xMesh, yMesh, zMesh, cmap=colormap, alpha=alpha, antialiased=True)
    if boundary_level is not None:
        ax.contour(xMesh, yMesh, zMesh, [boundary_level], linewidths=3, colors="k")

    if plot_data:
        plot_2d_data(
            ax,
            X,
            y,
            s=s,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            legend=legend,
            colormap=colormap,
        )
