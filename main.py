import numpy as np
from sbm import SBM
from variational_em import VariationalEM
from utils import draw_graph

if __name__=="__main__":

    # Test with two communities
    Gamma = np.array([
            [0.8, 0.01],
            [0.01, 0.8]
            ])
    Pi = np.array([0.5, 0.5])
    
    model = SBM(30, Gamma, Pi)
    Z, Z_v, A = model.sample()
    draw_graph(Z_v, A)

    var_em = VariationalEM(A, 2)
    var_em.run(verbose=True)
