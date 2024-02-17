from typing import Optional
import matplotlib.pyplot as plt
import numpy as np

def plot_contour_2D(model,
                    x_range: tuple,
                    y_range: tuple,
                    periods: Optional[np.ndarray]=None,
                    gaussians: Optional[list[int]]=None,
                    bin: int=100):
    """
    Generate a 2D contour plot for a given model.
    
    Parameters:
        model (GaussianMixtureModel): The model to be plotted.
        x_range (tuple): The range of x values for the plot.
        y_range (tuple): The range of y values for the plot.
        gaussians (list[int], optional): A list of indices of gaussians to be used in the model.
            If None, all gaussians will be used. Defaults to None.
        bin (float, optional): The bin size for generating the meshgrid. Defaults to 0.01.
    
    Returns:
        tuple: A tuple containing the figure and axis objects of the generated plot.
    """

    if gaussians is None:
        gaussians = [None]

    x, y = np.meshgrid(np.linspace(x_range[0], x_range[1], bin),
                       np.linspace(y_range[0], y_range[1], bin))
    shape = x.shape
    X = np.array(list(zip(np.concatenate(x), np.concatenate(y))))
    result = np.array([np.sum([model(i, idx) for idx in gaussians]) for i in X]).reshape(shape)

    fig, ax = plt.subplots()
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.contour(x, y, result)
    cs = ax.contourf(x, y, result)
    cbar = fig.colorbar(cs, ax=ax)
    if gaussians is None:
        cbar.set_label('Probability density')
    else:
        cbar.set_label('Probability of belonging to selected gaussians')

    return fig, ax

def cluster_distribution_2D(pamm,
                            use_index: Optional[list[int]]=None,
                            label_text: Optional[list[str]]=None,
                            size_scale: float=1e4,
                            fig_size: tuple[int, int]=(12, 12)) -> tuple[plt.Figure, plt.Axes]:
    """
    Generate a 2D scatter plot of the cluster distribution.

    Parameters:
        pamm (PAMM): The PAMM object containing the cluster data.
        use_index (Optional[list[int]]): The indices of the features to use for the scatter plot. 
            If None, the first three features will be used.
        label_text (Optional[list[str]]): The labels for the x and y axes. 
            If None, the labels will be set to 'Feature 0' and 'Feature 1'.
        size_scale (float): The scale factor for the size of the scatter points. Default is 1e4.
        fig_size (tuple[int, int]): The size of the figure. Default is (12, 12)

    Returns:
        tuple[plt.Figure, plt.Axes]: A tuple containing the matplotlib Figure and Axes objects.
    """
    if use_index is None:
        use_index = [0, 1]
    if label_text is None:
        label_text = [f'Feature {i}' for i in range(2)]

    fig, ax = plt.subplots(figsize=fig_size, dpi=100)
    ax.scatter(pamm.grid_pos[:, use_index[0]],
               pamm.grid_pos[:, use_index[1]],
               s=pamm.cluster_attributes['weights'] * size_scale,
               c=pamm.cluster_attributes['labels'])
    ax.set_xlabel(label_text[0])
    ax.set_ylabel(label_text[1])

    return fig, ax


def cluster_distribution_3D(pamm,
                            use_index: Optional[list[int]]=None,
                            label_text: Optional[list[str]]=None,
                            size_scale: float=1e4,
                            fig_size: tuple[int, int]=(12, 12)) -> tuple[plt.Figure, plt.Axes]:
    """
    Generate a 3D scatter plot of the cluster distribution.

    Parameters:
        pamm (PAMM): The PAMM object containing the cluster data.
        use_index (Optional[list[int]]): The indices of the features to use for the scatter plot. 
            If None, the first three features will be used.
        label_text (Optional[list[str]]): The labels for the x, y, and z axes. 
            If None, the labels will be set to 'Feature 0', 'Feature 1', and 'Feature 2'.
        size_scale (float): The scale factor for the size of the scatter points. Default is 1e4.
        fig_size (tuple[int, int]): The size of the figure. Default is (12, 12)

    Returns:
        tuple[plt.Figure, plt.Axes]: A tuple containing the matplotlib Figure and Axes objects.
    """
    if use_index is None:
        use_index = [0, 1, 2]
    if label_text is None:
        label_text = [f'Feature {i}' for i in range(3)]

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=fig_size, dpi=100)
    ax.scatter(pamm.grid_pos[:, use_index[0]],
               pamm.grid_pos[:, use_index[1]],
               pamm.grid_pos[:, use_index[2]],
               s=pamm.cluster_attributes['weights'] * size_scale,
               c=pamm.cluster_attributes['labels'])
    ax.set_xlabel(label_text[0])
    ax.set_ylabel(label_text[1])
    ax.set_zlabel(label_text[2])

    return fig, ax
