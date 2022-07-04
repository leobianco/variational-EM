import numpy as np
import argparse
from sbm import SBM
from variational_em import VariationalEM
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument(
        '-n', help='Number of points',
        type=int, default=100
        )
parser.add_argument(
        '-l', '--load',
        help='Loads saved graph', type=str)
parser.add_argument(
        '-v', '--verbose',
        help='Shows information on iterations', action='store_true')
parser.add_argument(
        '-vis', '--visual',
        help='Plots graph generated and relevant graphs', action='store_true')
parser.add_argument(
        '-s', '--sol',
        help='Initializes tau close to Z', action='store_true')
parser.add_argument(
        '--maxiter', help='Maximal number of iterations for EM',
        default=500, type=float)
parser.add_argument(
        '--tolELBO', help='Value of variation of ELBO to consider convergence',
        default=10**(-6))
args = parser.parse_args()


def main():
    # Model creation and graph sampling
    if args.load is None:
        # Generate a random graph
        Gamma = np.array([
                [0.8, 0.05],
                [0.05, 0.8]
                ])
        Pi = np.array([0.45, 0.55])
        model = SBM(args.n, Gamma, Pi)
        Z, Z_v, A = model.sample()
    else:
        # Alternatively, load a saved graph
        Gamma, Pi, Z, Z_v, A = load_graph(args.load)

    # Variational EM algorithm
    var_em = VariationalEM(A, 2, Z) if args.sol else VariationalEM(A, 2)
    var_em.run(
            max_iter=args.maxiter,
            tol_diff_ELBO=args.tolELBO,
            verbose=args.verbose)
    
    # Visualize the graph
    if args.visual:
        draw_graph(Z_v, var_em.tau, A)

    # Saving results
    if args.load is None:
        save_query = input('Save ? (y/n) ') or 'n'
        if save_query=='y':
            file_name = input('Name of this save: ') or 'graph'
            save_graph(file_name, Gamma, Pi, Z, Z_v, A)


if __name__=="__main__":
    main()
