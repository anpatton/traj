import numpy as np
from typing import Union, List
from scipy.spatial.distance import cdist
from numpy.typing import ArrayLike

def calc_euclidean_distance(
    base: ArrayLike, 
    comps: List[ArrayLike]
) -> Union[float, List[float]]:
    """
    Calculate Euclidean distances between a base vector and comparison vectors.
    
    Args:
        base (list/array): Base vector for comparison (many vs. one)
        comps (list/array): Collection of comparison vectors
    
    Returns:
        Single distance if one comparison, list of distances if multiple comparisons
    """
    base_array = np.array(base).reshape(1, -1)
    comps_matrix = np.array(comps)
    distances = cdist(base_array, comps_matrix, metric='euclidean')[0]

    if len(distances) > 1:
        return distances.tolist()
    else:
        return distances[0]
