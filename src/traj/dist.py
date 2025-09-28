import numpy as np
from typing import Union, List
from scipy.spatial.distance import cdist
from numpy.typing import ArrayLike


def calc_distances_mvo(
    base_vector: ArrayLike,
    comparison_vectors: List[ArrayLike],
    distance_metric: str = "euclidean",
) -> Union[float, List[float]]:
    """
    Calculate distances between a base vector and comparison vectors (many vs. one) using a specified metric. 
    See scipy.spatial.distance.cdist for more. This *does not* return a distance matrix.

    Args:
        base_vector (list/array): Base vector for comparison (many vs. one)
        comparison_vectors (list/array): Collection of comparison vectors
        distance_metric (str): Distance metric (e.g. 'cityblock', 'cosine', 'minkowski', etc.)

    Returns:
        Single distance if one comparison, list of distances if multiple comparisons
    """
    base_vector_array = np.array(base_vector).reshape(1, -1)
    comparison_vectors_matrix = np.array(comparison_vectors)
    distances = cdist(
        base_vector_array, comparison_vectors_matrix, metric=distance_metric
    )
    distances=distances[0]

    if len(distances) > 1:
        return distances.tolist()
    else:
        return distances[0]

def calc_frechet_mvo(
    base_vector: ArrayLike,
    comparison_vectors: List[ArrayLike],
) -> Union[float, List[float]]:
    """
    Calculate discrete Frechet distances between a base vector and comparison vectors (many vs. one). This *does not* return a distance matrix. Claude-assisted.

    Args:
        base_vector (list/array): Base vector for comparison (many vs. one)
        comparison_vectors (list/array): Collection of comparison vectors

    Returns:
        Single distance if one comparison, list of distances if multiple comparisons
    """
    base_array = np.array(base_vector)
    results = []
    
    for comp in comparison_vectors:
        comp_array = np.array(comp)
        n, m = len(base_array), len(comp_array)
        
        dist_matrix = np.abs(base_array[:, np.newaxis] - comp_array[np.newaxis, :])
        
        dp = np.full((n, m), np.inf)
        dp[0, 0] = dist_matrix[0, 0]
        
        for i in range(1, n):
            dp[i, 0] = max(dp[i-1, 0], dist_matrix[i, 0])
        for j in range(1, m):
            dp[0, j] = max(dp[0, j-1], dist_matrix[0, j])
        
        for i in range(1, n):
            for j in range(1, m):
                dp[i, j] = max(dist_matrix[i, j], 
                              min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1]))
        
        results.append(float(dp[n-1, m-1]))
    
    if len(results) > 1:
        return results
    else:
        return float(results[0])


