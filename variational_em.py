import numpy as np
import numpy.matlib
rng = np.random.default_rng()
np.set_printoptions(precision=2)  # for clarity
from scipy.optimize import minimize
from utils import *


class VariationalEM():
    """Object representing the variational EM algorithm for the SBM. Passing the
    optional argument Z initializes the variational parameters tau close to the
    true solution, for testing purposes.
    A note about initialization: first, tau is initialized (randomly or close to
    solution), then the model parameters are initialized by estimation from tau.
    This way, the starting parameters have more sense.

    Args:
        A ((n, n) np.array): adjacency matrix.
        k (int): number of communities.
        Z ((n, k) np.array): communities matrix (optional).
    """

    def __init__(self, A, k, Z=None):
        
        # Attributes
        self.Z = Z
        self.A = A
        self.k = k
        self.n = np.shape(self.A)[0]
        self.n_iter = 0
        self.tol_pf = 10**(-5)
        self.tol_mask = 10**(-1)/self.n

        # Initialization of variational parameters
        if self.Z is None:
            self.tau = np.array(
                    [rng.dirichlet(np.ones(self.k)) for i in range(self.n)])
        else:
            self.tau = np.where(self.Z==1, 0.95, 0.05)

        # Initialization of model parameters and of ELBO
        self.Gamma = self.estimate_Gamma()
        self.Pi = self.estimate_Pi()
        self.curr_ELBO = self.ELBO()


    def estimate_Gamma(self):
        """Estimates Gamma using current tau."""

        return (self.tau.T @ (self.A @ self.tau))\
                /(self.tau.T @ ((1-np.eye(self.n)) @ self.tau))


    def estimate_Pi(self):
        """Estimates Pi using current tau."""

        return np.mean(self.tau, axis=0)


    def ELBO(self):
        """Calculates ELBO at current parameters."""

        ELBO =\
            -np.sum(self.tau*np.log(self.tau))\
                    + np.sum(np.dot(self.tau, np.log(self.Pi)))\
                    + .5*(np.sum(self.tau*(np.dot(np.dot(self.A,
                        self.tau),np.log(self.Gamma)))))\
                    + .5*(np.sum(self.tau*(np.dot(np.dot(
                        (1-np.eye(self.n)-self.A), self.tau),
                        np.log(1-self.Gamma)))))

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
            L_compl = extract_upper_triang(1 - self.A)
            lik = np.matlib.repmat(np.log(self.Pi), self.n, 1)\
                    + L @ (tau @ np.log(self.Gamma))\
                    + L_compl @ (tau @ np.log(1 - self.Gamma))
            unnorm =\
                    np.exp(lik - np.matlib.repmat(np.max(lik, axis=1), self.k,
                        1).T)  # substracting the max is numerical trick
            result = normalize_rows(unnorm) 

            # Numerical tricks to leave tau a bit far to the border
            first_mask = np.where(result < self.tol_mask, self.tol_mask, result)
            second_mask = np.where(first_mask > 1-self.tol_mask, 1-self.tol_mask, first_mask)

            # Perform a second normalization after masking
            return normalize_rows(second_mask)

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

        self.Pi = self.estimate_Pi()
        self.Gamma = self.estimate_Gamma()
        
        return None


    def run(self, max_iter=100, verbose=False, tol_diff_ELBO=10**(-6)):
        """Alternates E and M steps.

        Args:
            max_iter (int): maximal number of iterations.
        """
        
        diff_ELBO = 1
        i = 0
        
        while i < max_iter and np.abs(diff_ELBO) > tol_diff_ELBO:
            if verbose and (i%5==0 or i==1):
                print_info(
                        i, self.curr_ELBO, diff_ELBO,
                        self.tau, self.Gamma, self.Pi)

            self.M_step()
            self.E_step()
            i += 1

            ELBO = self.ELBO()
            diff_ELBO = ELBO - self.curr_ELBO
            self.curr_ELBO = ELBO

        self.n_iter = i

        # Last iteration information
        print('Total number of iterations: ', self.n_iter)
        if self.Z is not None:
            print('Accuracy: ', accuracy(self.tau, self.Z))

        return None
