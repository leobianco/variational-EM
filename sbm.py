import numpy as np


class SBM:
    """Object representing a stochastic block model.

    Args:
        n (int): number of nodes on the model.
        Gamma ((k, k) np.array): connectivity parameters of communities.
        Pi ((k,) np.array): prior probability vector on community assignment.
    """

    def __init__(self, n, Gamma, Pi):
        # Attributes
        self.n = n
        self.Pi = Pi
        self.k = np.shape(Pi)[0]
        self.Gamma = Gamma

    # Methods
    def sample(self):
        """Samples a random graph with n nodes from the stochastic block model.

        Returns:
            Z ((n, k) np.array): latent matrix of community assignments.
            Z_v ((n,) np.array): vector with community number of node n.
            A ((n, n) np.array): symmetric adjacency matrix.
        """

        rng = np.random.default_rng()
        # Communities
        Z = rng.multinomial(1, self.Pi, size=self.n)
        Z_v = np.argmax(Z, axis=1)  # labeled from 0 to k-1
        # Adjacency matrix
        A = np.zeros([self.n, self.n])  # lazy loop
        for i in range(self.n-1):  # nothing to do on the last line
            for j in range(i+1, self.n):  
                A[i, j] = rng.binomial(1,
                            self.Gamma[Z_v[i], Z_v[j]])
                A[j, i] = A[i, j]

        return Z, Z_v, A 
