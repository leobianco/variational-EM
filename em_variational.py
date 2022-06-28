import numpy as np
rng = np.random.default_rng()
from scipy.optimize import minimize


class VariationalEM():
    """Object representing the variational EM algorithm for the SBM.

    Args:
        A ((n, n) np.array): adjacency matrix.
        k (int): number of communities.
        disp (bool): display optimizer information or not, default is false.
        eps (float): tolerance in optimizer bounds, default is 10**(-8).

    Returns:
        Gamma_hat ((k, k) np.array): estimated connectivities.
        Pi_hat ((k,) np.array): estimated community priors.
    """

    def __init__(self, A, k, disp=False, eps=10**(-8)):
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
        # Optimizer configuration
        self.eps = eps
        self.disp = disp

    def E_step(self):
        """Updates tau."""

        # Objective function
        def minus_ELBO(self, tau):
            """Minus ELBO as a function of tau.
            
            Args:
                tau ((n, k) np.array): vectorized (for optimizer) matrix tau.
            """
            
            tau = np.reshape(tau, [self.n, self.k])  # matrix form

            ELBO =\
            np.sum(tau*np.log(tau))\
                    + np.sum(np.dot(tau, np.log(self.Pi)))\
                    + .5*(np.sum(np.dot(tau,np.dot(np.log(self.Gamma),
                        tau.T))*self.A))\
                    + .5*(np.sum(np.dot(tau, np.dot(np.log(1-self.Gamma),
                        tau.T))*(1-np.eye(self.n)-self.A)))

            return -ELBO

        # Optimizer
        bounds = ((0 + self.eps, 1 - self.eps) for i in range(self.n*self.k))
        optim_results = minimize(
                lambda x: minus_ELBO(self, x),
                self.tau,
                bounds=bounds,
                options={'disp': self.disp})

        # Update estimated variational parameters
        self.tau = np.reshape(optim_results['x'], [self.n, self.k])

        return None


    def M_step(self):
        """Updates theta."""

        # Objective function
        def minus_complete_expectation(self, theta):
            """Minus the complete expectation as a function of parameters.
            
            Args: 
                theta ((k**2 + k,) np arrays): parameters Gamma and Pi, 
                respectively.

            Returns:
                complete_expectation (float): the complete expectation value.
            """
            
            # Unpack
            Gamma = np.reshape(theta[:self.k**2], [self.k, self.k])
            Pi = np.reshape(theta[-self.k:], [self.k,])

            complete_expectation =\
            np.sum(np.dot(self.tau, np.log(Pi)))\
                    + .5*(np.sum(np.dot(self.tau,np.dot(np.log(Gamma),
                        self.tau.T))*self.A))\
                    + .5*(np.sum(np.dot(self.tau, np.dot(np.log(1-Gamma),
                        self.tau.T))*(1-np.eye(self.n)-self.A)))

            return -complete_expectation

        # Optimizer
        bounds = ((0 + self.eps, 1 - self.eps) for i in range(self.k*(self.k+1)))
        theta_0 = np.concatenate((np.ravel(self.Gamma), self.Pi))
        optim_results = minimize(
                lambda x: minus_complete_expectation(self, x),
                theta_0,
                bounds=bounds,
                options={'disp': self.disp})

        # Update estimated model parameters
        self.Gamma=np.reshape(optim_results['x'][:self.k**2], [self.k, self.k])
        self.Pi = np.reshape(optim_results['x'][-self.k:], [self.k,])

        return None

    def run(self, n_iter=10):
        """Alternates E and M steps.

        Args:
            n_iter (int): number of iterations.
        """
        
        for i in range(n_iter):
            self.E_step()
            self.M_step()
    
        return None
