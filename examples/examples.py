import numpy as np
from traj.dist import calc_euclidean_distance

# Example with lists
base_list = [1, 2, 3]
comps_list = [[2, 3, 4], [99, 99, 99]]
result1 = calc_euclidean_distance(base_list, comps_list)
print(result1)

# Example with arrays
base_array = np.array([1, 2, 3])
comps_array = [np.array([2, 3, 4]), np.array([99, 99, 99])]
result2 = calc_euclidean_distance(base_array, comps_array)
print(result2)