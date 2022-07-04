# Variational EM for community detection

This repository contains part of the code I developed during my M2 internship. It generates a random graph according to the SBM models and runs variational EM on it to solve for the parameters and to estimate the communities.

## How to use it

**tl;dr** run ```python3 main.py -n 500 -vis -v``` in your terminal to generate a graph with 500 nodes and to run the algorithm on it.

------

Parameters $\Gamma$ and $\pi$ can be modified inside ```main.py```, default values are
$$
\Gamma = \begin{pmatrix}0.8 & 0.05 \\ 0.05 & 0.8 \end{pmatrix} \quad \text{and} \quad \pi = (0.45, 0.55).
$$
Run the ```main.py``` script along with the desired flags:

```-h```: shows help on terminal,

```-n```: number of points on the graph to be generated,

```-l``` or ```--load```: specify the name of the graph inside the ```saved_graphs``` folder you wish to load (without the ```.npy``` extension !),

```-v``` or ```--verbose```: runs algorithm in verbose mode, showing info on tau, the parameters, and the ELBO each five iterations,

```-vis``` or ```--visual```: shows the graph generated, along with ground truth (GT) community labels and estimated (E) community labels, in the format GT | E on each node.

```-s``` or ```--sol```: initializes $\tau$ "close to the solution", i.e., $\tau_0 = Z + \varepsilon$, mostly for debugging purposes,

```--maxiter```: number of maximal iterations allowed,

```--tolELBO```: if the ELBO varies less than this value, then it is considered the algorithm has converged (or not and it is stuck alternating points).

## References

