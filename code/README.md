Code in this directory is generally in the form of `experiment.py` and `experiment_figures.py`. 



`utils.py` contains a range of helpful scripts, like `calc_AVE_quick` which is used to return the AVE score of a split when you have a pre-calculated distance matrix. It also has `trim` which is used to trim a fraction of the ligands from the training set that are most similar to the test set. Also useful is `fast_dice` and `fast_jaccard` which calculate pairwise distance matrices from from sparse input matrices.



`paris_cluster.py` generates a sparse adjacency graph, and it uses `pynndescent` to do this so it scales to huge datasets. It also uses the PARIS clustering algorithm from `sknetwork` to do cluster splitting. 


`make_fingerprints.py` will generate all fingerprints, except for CATS, using `rdkit`. `make_cats.py` has our implementation of the CATS fingerprint (also using `rdkit`).
