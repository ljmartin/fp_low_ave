# fp_generalizability
 testing ability of different fingerprints to generalize to out-of-sample data

Ligand based virtual screening is the use of machine learning for drug discovery. In 2018 it was demonstrated that the performance in many benchmarks was strongly correlated with bias in the train/test splits, and the authors also proposed a genetic algorithm to reduce bias (1). Subsequently, using the genetic algorithm, it was shown that the _de facto_ standard featurization (i.e. ECFP a.k.a. Morgan fingerprints) performs poorly when the bias is low (2). We extend both results - we present an algorithm for fast (~1-2s) unbiased train/test splitting, which allows us to sample hundreds of splits quite quickly, and we use this to compare 11 different fingerprints. 

>(1) Wallach, Izhar, and Abraham Heifets. "Most ligand-based classification benchmarks reward memorization rather than generalization." Journal of chemical information and modeling 58.5 (2018): 916-932.
>(2) Sundar, Vikram, and Lucy Colwell. "The Effect of Debiasing Protein Ligand Binding Data on Generalisation." Journal of Chemical Information and Modeling (2019).
