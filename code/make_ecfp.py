from scipy import sparse
import numpy as np
import pandas as pd
import mmh3

from joblib import Parallel, delayed

from rdkit import Chem
from rdkit.Chem.rdReducedGraphs import GetErGFingerprint
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys
from rdkit.Chem.rdmolops import PatternFingerprint, LayeredFingerprint, RDKFingerprint
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D,Generate

def makeMols(num=None):
    smiles = pd.read_csv('./raw_data/all_chemicals.csv', header=0)
    mols = list()
    for smile in smiles['standard_smiles'].iloc[0:num]:
        mols.append(Chem.MolFromSmiles(smile))
    return np.array(mols)

def get_morgan(mols):
    gen_mo = rdFingerprintGenerator.GetMorganGenerator()
    fps = list()
    for mol in mols:
        fp = np.array(gen_mo.GetFingerprint(mol))
        fps.append(fp)
    fps = np.array(fps)
    return sparse.csr_matrix(fps).astype('int')

if __name__ == '__main__':

    mols = makeMols()

    #These ones are pickleable:
    funcs = [get_morgan]
    names = ['morgan']
    n_jobs = 8
    for func, name in zip(funcs, names):
        print(f'Making {name} fingerprints')
        split_fps = Parallel(n_jobs=n_jobs)(delayed(func)(i) for i in np.array_split(mols, n_jobs))
        fps = sparse.vstack([*split_fps])
        sparse.save_npz('./processed_data/fingerprints/'+name+'.npz', fps)
