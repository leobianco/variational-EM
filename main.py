import numpy as np
from sbm import SBM
from variational_em import VariationalEM
from utils import *

if __name__=="__main__":

    load_query = input('Load graph ? (y/n) ')

    if load_query=='n':
        Gamma = np.array([
                [0.8, 0.05],
                [0.05, 0.8]
                ])
        Pi = np.array([0.45, 0.55])
        
        model = SBM(100, Gamma, Pi)
        Z, Z_v, A = model.sample()
    elif load_query=='y':
        file_name = input('Enter name of graph: ') or 'graph'
        Gamma, Pi, Z, Z_v, A = load_graph(file_name)

    draw_graph(Z_v, A)

    var_em = VariationalEM(A, 2, Z)
    var_em.run(max_iter=500, verbose=True, tol_diff_ELBO=10**(-10))
    save_query = input('Save ? (y/n) ') or 'n'
    if save_query=='y':
        file_name = input('Name of this save: ') or 'graph'
        save_graph(file_name, Gamma, Pi, Z, Z_v, A)
