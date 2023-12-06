import os
from typing import Optional
import numpy as np
from rich.progress import track

def get_gabriel_graph(dist_matrix2: np.ndarray, outputname: Optional[str]=None):
    """
    Generate the Gabriel graph based on the given squared distance matrix.

    Parameters:
        dist_matrix2 (np.ndarray): The squared distance matrix of shape (n_points, n_points).
        outputname (Optional[str]): The name of the output file. Default is None.

    Returns:
        np.ndarray: The Gabriel graph matrix of shape (n_points, n_points).

    """

    n_points = dist_matrix2.shape[0]
    gabriel = np.full((n_points, n_points), True)
    for i in track(range(n_points), description='Calculating Gabriel graph'):
        gabriel[i, i] = False
        for j in range(i, n_points):
            if np.sum(dist_matrix2[i] + dist_matrix2[j] < dist_matrix2[i, j]):
                gabriel[i, j] = False
                gabriel[j, i] = False

    if outputname is not None:
        neigh_file = os.path.basename(outputname) + '.neigh'
        with open(neigh_file, 'w', encoding='utf-8') as wfl:
            for i in range(n_points):
                print(' '.join(gabriel[i, :]), file=wfl)
    return gabriel
