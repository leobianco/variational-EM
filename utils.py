import numpy as np
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
