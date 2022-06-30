import numpy as np
from sbm import SBM
from variational_em import VariationalEM
from utils import draw_graph

if __name__=="__main__":

    # Test with two communities
    Gamma = np.array([
            [0.8, 0.6],
            [0.6, 0.8]
            ])
    Pi = np.array([0.45, 0.55])
    
    model = SBM(30, Gamma, Pi)
    Z, Z_v, A = model.sample()
    draw_graph(Z_v, A)

    var_em = VariationalEM(A, 2, Z)
    var_em.run(verbose=True, tol_diff_ELBO=10**(-10))
    print('Z: ', Z)
