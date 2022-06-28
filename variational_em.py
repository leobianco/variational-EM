import numpy as np
import numpy.matlib
rng = np.random.default_rng()
from scipy.optimize import minimize
from utils import extract_upper_triang


class VariationalEM():
    """Object representing the variational EM algorithm for the SBM.

    Args:
        A ((n, n) np.array): adjacency matrix.
        k (int): number of communities.
        tol (float): tolerance for fixed point iteration, default is 10**(-6).

    Returns:
        Gamma_hat ((k, k) np.array): estimated connectivities.
        Pi_hat ((k,) np.array): estimated community priors.
    """

    def __init__(self, A, k, tol=10**(-6)):
        # Attributes from model
        self.A = A
        self.k = k
        self.n = np.shape(self.A)[0]
        # Parameters to estimate, randomly initialized
        self.tau = rng.uniform(size=(self.n, self.k))  # variational parameter
        self.Gamma = np.zeros([self.k, self.k])  # connectivities parameter
        for i in range(self.k):
            self.Gamma[i,i] = rng.uniform() 
            for j in range(i+1, self.k):
                self.Gamma[i,j] = rng.uniform()
                self.Gamma[j,i] = self.Gamma[i,j]
        self.Pi = rng.uniform(size=self.k)  # prior on communities parameter
        # Others
        self.tol = tol

    def E_step(self):
        """Updates tau via a fixed point relation."""

        def fixed_point(self, tau):
            """Function describing the fixed point relation satisfied by tau.
            
            Args:
                tau ((n, k) np.array): matrix of variational parameters.
                
            Returns:
                ((n, k) np.array): one application of the fixed point function.
            """
            
            L = extract_upper_triang(self.A)

            return np.exp(np.matlib.repmat(np.log(self.Pi), self.n, 1)\
                    + L @ (tau @ np.log(self.Gamma))\
                    + (1-L) @ (tau @ np.log(1 - self.Gamma)))

        # Iterate it until convergence
        diff1 = 1
        diff2 = 1  # indicates oscillation
        while diff1 > self.tol and diff2 > self.tol:
            tau_prev = self.tau
            self.tau = fixed_point(self, self.tau)
            diff1_prev = diff1
            diff1 = np.max(np.abs(self.tau - tau_prev))
            diff2 = np.max(np.abs(diff1_prev - diff1))

        return None


    def M_step(self):
        """Updates theta."""

        self.Pi = np.mean(self.tau, axis=0)
        self.Gamma =\
                (self.tau.T @ (self.A @ self.tau))\
                /(self.tau.T @ ((1-np.eye(self.n)) @ self.tau))
        
        return None

    def run(self, n_iter=10):
        """Alternates E and M steps.

        Args:
            n_iter (int): number of iterations.

        TO DO: AJOUTER LE CRITERE D'ARRETE
        """
        
        for i in range(n_iter):
            self.E_step()
            self.M_step()
    
        return None
