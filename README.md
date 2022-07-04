# Variational EM for community detection

This repository contains part of the code I developed during my M2 internship. It generates a random graph according to the SBM models and runs variational EM on it to solve for the parameters and to estimate the communities.

## How to use it

**tl;dr** run ```python3 main.py -n 300 -vis -v``` in your terminal to generate a graph with 300 nodes and to run the algorithm on it, and visualize the results.

------

Parameters $\Gamma$ and $\pi$ can be modified inside ```main.py```, default values are

$\Gamma=((0.8, 0.05), (0.05, 0.8))$ and $\pi = (0.45, 0.55)$. Run the ```main.py``` script along with the desired flags:

```-h```: shows help on terminal,

```-n```: number of points on the graph to be generated,

```-l``` or ```--load```: specify the name of the graph inside the ```saved_graphs``` folder you wish to load (without the ```.npy``` extension !),

```-v``` or ```--verbose```: runs algorithm in verbose mode, showing info on tau, the parameters, and the ELBO each five iterations,

```-vis``` or ```--visual```: shows the graph generated, along with ground truth (GT) community labels and estimated (E) community labels, in the format GT | E on each node.

```-s``` or ```--sol```: initializes $\tau$ "close to the solution", i.e., $\tau_0 = Z + \varepsilon$, mostly for debugging purposes,

```--maxiter```: number of maximal iterations allowed,

```--tolELBO```: if the ELBO varies less than this value, then it is considered the algorithm has converged (or not and it is stuck alternating points).

If you have not loaded a saved graph, at the end of the execution there will be a prompt ```Save ? (y/n)``` allowing you to save a result. To ignore it, type ```n``` or simply press ```Enter```.

## References

[1] Mariadassou, M., Robin, S. and Vacher, C. (2010). Uncovering latent structure in valued graphs: A variational approach. *The Annals of Applied Statistics* **4** 715–42.



