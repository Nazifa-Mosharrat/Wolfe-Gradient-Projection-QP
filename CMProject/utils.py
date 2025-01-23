import numpy as np
from scipy.io import loadmat

# define the objective function for the quadratic optimization problem
def objective(Q, q, x):
    return np.dot(x.T, np.dot(Q, x)) + np.dot(q, x)

# compute the gradient of the objective function
def gradient(Q, q, x):
    return 2 * np.dot(Q, x) + q

# partition indices into subsets for block-coordinate methods
def partition_indices(n, partition_number=None):
    num_partitions = max(1, int(np.sqrt(n))) if partition_number is None else partition_number
    indices = np.arange(n)
    np.random.shuffle(indices)
    partition_size = n // num_partitions
    I_sets = [indices[i * partition_size: (i + 1) * partition_size] for i in range(num_partitions - 1)]
    I_sets.append(indices[(num_partitions - 1) * partition_size:])
    return I_sets

# load matrix data from a .mat file and generate the optimization problem parameters
def Initialization(path):
    mat = loadmat(path)
    Q = mat['Problem']['A'][0, 0].toarray()
    q = np.random.randn(Q.shape[0])
    return Q, q, Q.shape[0]

# perform projection of a vector onto the probability simplex
def projection(v):
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    rho = np.where(u > cssv / (np.arange(len(u)) + 1))[0][-1]
    mu = cssv[rho] / (rho + 1)
    return np.maximum(v - mu, 0)
