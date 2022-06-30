import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import networkx as nx


def draw_graph(communities, A):
    """Wrapper for networkx drawing capabilities.
    
    Args:
        A ((n, n) np.array): adjacency matrix of the graph to be shown.

    Returns:
        None (pyplot window with graph)
    """

    G = nx.from_numpy_matrix(A)
    nx.draw(G, node_color=communities)
    plt.show()


def extract_upper_triang(A):
    """Extracts the upper triangular part of a matrix.

    Args:
        A ((n, n) np.array): matrix to have triangular part extracted.

    Returns:
        ((n, n) np.array): upper triangular part of A.
    """

    return np.triu(A) - np.diag(np.diag(A))


def normalize_rows(A):
    """Normalizes a matrix row-wise (no inbuilt method in numpy for this !).

    Args:
        A ((n, n) np.array): matrix to be normalized row-wise.

    Returns:
        norm_A ((n, n) np.array): row-wise normalized version of A.
    """
    
    sA = np.sum(A, axis=1)  # sum along rows
    k = A.shape[1]

    norm_A = A / np.matlib.repmat(sA.T, k, 1).T
    
    return norm_A


def print_info(i, ELBO_prev, diff_ELBO, tau, Gamma, Pi):
    """Prints parameter information for the i-th iteration of EM. I
    just coded this function so that the code in variational_em.py is cleaner.
    
    Args:
        i (int): iteration index.
        var_em_object (VariationalEM object): the algorithm object.
    """

    print('----------')
    print('\n', i, ' iterations \n')
    print('Current ELBO: ', ELBO_prev, '\n')
    if i>0:
        print('Current ELBO variation: ', diff_ELBO, '\n')
    print('Current Tau: \n', tau, '\n')
    print('Current Gamma: \n', Gamma, '\n')
    print('Current Pi: \n', Pi, '\n') 
    print('----------')

