import numpy as np
import numpy.matlib
rng = np.random.default_rng()
from scipy.optimize import minimize
from utils import extract_upper_triang, normalize_rows


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

    def __init__(self, A, k, tol_pf=10**(-6), tol_em=10**(-6)):
        # Attributes from model
        self.A = A
        self.k = k
        self.n = np.shape(self.A)[0]
        # Parameters to estimate, randomly initialized
        self.tau =\
                np.array([rng.dirichlet(np.ones(self.k)) for i in range(self.n)])
        self.Gamma = np.zeros([self.k, self.k])  # connectivities parameter
        for i in range(self.k):
            self.Gamma[i,i] = rng.uniform() 
            for j in range(i+1, self.k):
                self.Gamma[i,j] = rng.uniform()
                self.Gamma[j,i] = self.Gamma[i,j]
        self.Pi = rng.dirichlet(np.ones(self.k))  # prior on communities parameter
        # Tolerances for convergence
        self.tol_pf = tol_pf
        self.tol_em = tol_pf
        self.n_iter = 0

    def ELBO(self):
        """Calculates ELBO at current parameters."""

        ELBO =\
            np.sum(self.tau*np.log(self.tau))\
                    + np.sum(np.dot(self.tau, np.log(self.Pi)))\
                    + .5*(np.sum(np.dot(self.tau,np.dot(np.log(self.Gamma),
                        self.tau.T))*self.A))\
                    + .5*(np.sum(np.dot(self.tau, np.dot(np.log(1-self.Gamma),
                        self.tau.T))*(1-np.eye(self.n)-self.A)))

        return ELBO


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

            unnorm = np.exp(np.matlib.repmat(np.log(self.Pi), self.n, 1)\
                    + L @ (tau @ np.log(self.Gamma))\
                    + (1-L) @ (tau @ np.log(1 - self.Gamma)))

            return normalize_rows(unnorm)

        # Iterate it until convergence
        diff1 = 1
        diff2 = 1  # indicates oscillation
        while diff1 > self.tol_pf and diff2 > self.tol_pf:
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

    def run(self, max_iter=100, verbose=False):
        """Alternates E and M steps.

        Args:
            max_iter (int): maximal number of iterations.

        TO DO: AJOUTER LE CRITERE D'ARRETE
        """
        
        diff_ELBO = 1
        i = 0

        while i < max_iter and diff_ELBO > self.tol_em:
            ELBO_prev = self.ELBO()

            self.E_step()
            self.M_step()

            ELBO = self.ELBO()
            diff_ELBO = np.abs(ELBO - ELBO_prev)
            i += 1
            
            if verbose and i%5==0:
                print('----------')
                print(i, ' iterations')
                print('Current ELBO variation: ', diff_ELBO, '\n')
                print('Current Tau: ', self.tau, '\n')
                print('Current Gamma: ', self.Gamma, '\n')
                print('Current Pi: ', self.Pi, '\n')
                print('----------')

        self.n_iter = i

        return None
