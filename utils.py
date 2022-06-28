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
