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
    smiles = pd.read_csv('./raw_data/allSmiles.csv', header=None)
    mols = list()
    for smile in smiles[0].iloc[0:num]:
        mols.append(Chem.MolFromSmiles(smile))
    return np.array(mols)

def get_reduced_graphs(mols):
    fps = list()
    for mol in mols:
        fps.append(GetErGFingerprint(mol))
    fps = np.array(fps)
    return sparse.csr_matrix(fps).astype('float')

def get_maccs(mols):
    fps = list()
    for mol in mols:
        fps.append(np.array(MACCSkeys.GenMACCSKeys(mol)))
    fps = np.array(fps)
    return sparse.csr_matrix(fps).astype('int')

def get_rdk_fps(mols):
    fps = list()
    for mol in mols:
        fps.append(np.array(RDKFingerprint(mol)))
    fps = np.array(fps)
    return sparse.csr_matrix(fps).astype('int')

def get_pattern_fps(mols):
    fps = list()
    for mol in mols:
        fps.append(np.array(PatternFingerprint(mol)))
    fps = np.array(fps)
    return sparse.csr_matrix(fps).astype('int')

def get_layered_fps(mols):
    fps = list()
    for mol in mols:
        fps.append(np.array(LayeredFingerprint(mol)))
    fps = np.array(fps)
    return sparse.csr_matrix(fps).astype('int')

def get_2dpharm(mols, fp_size=2000):
    factory = Gobbi_Pharm2D.factory
    fps = list()
    for mol in mols:
        try:
            sig = Generate.Gen2DFingerprint(mol,factory)
            indices = np.array([mmh3.hash(str(i)) for i in sig.GetOnBits()])%fp_size
            fp = np.zeros(fp_size, dtype=int)
            if len(indices)>0:
                fp[indices]=1
            fps.append(fp)
        except Exception:
            print('ERROR')
            print(Chem.MolToSmiles(mol))
    fps = np.array(fps)
    return sparse.csr_matrix(fps).astype('int')

def get_atom_pair(mols):
    gen_ap = rdFingerprintGenerator.GetAtomPairGenerator()
    fps = list()
    for mol in mols:
        fp = np.array(gen_ap.GetFingerprint(mol))
        fps.append(fp)
    fps = np.array(fps)
    return sparse.csr_matrix(fps).astype('int')

def get_topological_torsion(mols):
    gen_tt = rdFingerprintGenerator.GetTopologicalTorsionGenerator()
    fps = list()
    for mol in mols:
        fp = np.array(gen_tt.GetFingerprint(mol))
        fps.append(fp)
    fps = np.array(fps)
    return sparse.csr_matrix(fps).astype('int')

def get_morgan(mols):
    gen_mo = rdFingerprintGenerator.GetMorganGenerator()
    fps = list()
    for mol in mols:
        fp = np.array(gen_mo.GetFingerprint(mol))
        fps.append(fp)
    fps = np.array(fps)
    return sparse.csr_matrix(fps).astype('int')

def get_morgan_features(mols):
    invGen =rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
    gen_mo = rdFingerprintGenerator.GetMorganGenerator(atomInvariantsGenerator=invGen)
    fps = list()
    for mol in mols:
        fp = np.array(gen_mo.GetFingerprint(mol))
        fps.append(fp)
    fps = np.array(fps)
    return sparse.csr_matrix(fps).astype('int')

if __name__ == '__main__':

    mols = makeMols()

    #These ones are pickleable:
    funcs = [get_2dpharm, get_maccs, get_atom_pair, get_topological_torsion,
             get_morgan, get_morgan_features]
    names = ['2dpharm', 'maccs', 'atom_pair', 'topo_torsion',
             'morgan', 'morgan_feat']
    n_jobs = 8
    for func, name in zip(funcs, names):
        print(f'Making {name} fingerprints')
        split_fps = Parallel(n_jobs=n_jobs)(delayed(func)(i) for i in np.array_split(mols, n_jobs))
        fps = sparse.vstack([*split_fps])
        sparse.save_npz('./processed_data/fingerprints/'+name+'.npz', fps)


    #these ones are not pickleable:
    funcs = [get_reduced_graphs, get_rdk_fps, get_pattern_fps,
             get_layered_fps]
    names = ['erg', 'rdk', 'pattern', 'layered']
    for func, name in zip(funcs, names):
        print(f'Making {name} fingerprints')
        fps = func(mols)
        sparse.save_npz('./processed_data/fingerprints/'+name+'.npz', fps)
