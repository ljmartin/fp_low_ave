import pandas as pd
from scipy import sparse
import copy
from rdkit import Chem
from rdkit.Chem import SaltRemover
from rdkit.Chem import Descriptors
import tqdm
import numpy as np



interaction_data = pd.read_csv('raw_data/interaction_data_pchembl.csv')
interaction_data=interaction_data.rename(columns = {'chembl_id.1':'ligand_ID'}).sort_values(by='year')
interaction_data = interaction_data.drop_duplicates(['chembl_id', 'ligand_ID'], keep='first')



remover = Chem.SaltRemover.SaltRemover(defnData="[Cl,Br,I,Ca,Zn,Li,Na,K,Mg,B,Ag]")

def sanitizeMols(smiles_input):
    flags = np.zeros(len(smiles_input)).astype(bool)
    saltmols = list()
    smiles_output = list()
    for count, smi in tqdm.tqdm(enumerate(smiles_input), total=len(smiles_input)):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                wt = Descriptors.MolWt(mol)
                if wt>48 and wt<650:
                    nosaltmol = remover.StripMol(mol)
                    molwt = Chem.Descriptors.MolWt(nosaltmol)
                    Chem.RemoveHs(nosaltmol)
                    smiles_output.append(Chem.MolToSmiles(nosaltmol))
                    flags[count]=True
        except Exception:
            pass
    return smiles_output, flags


smiles, flags = sanitizeMols(interaction_data['canonical_smiles'])
interaction_data = interaction_data[flags]
interaction_data['standard_smiles']=smiles
interaction_data_filtered = interaction_data[interaction_data['pchembl_value']>6]


df = interaction_data_filtered

num_instances = df['ligand_ID'].unique().shape[0]
num_targets = df['chembl_id'].unique().shape[0]

#interaction matrix:
interaction_matrix = np.zeros([num_instances, num_targets])
#interaction dates:
interaction_dates = copy.copy(interaction_matrix)

###setting up column indices, to use in filling in the matrices above
tids = df.sort_values('chembl_id')['chembl_id'].unique()
cids = df.sort_values('ligand_ID')['ligand_ID'].unique()

target_indices = dict()
for count, i in enumerate(tids):
    target_indices[i]=count

instance_indices = dict()
for count, i in enumerate(cids):
    instance_indices[i]=count


#Actually filling the values:

for count, item in tqdm.tqdm(df.iterrows()):
    t_id = item['chembl_id']
    i_id = item['ligand_ID']
    date = item['year']

    row = instance_indices[i_id]
    column = target_indices[t_id]
    
    interaction_matrix[row, column] = 1
    interaction_dates[row, column] = date


##Make sure there hasnt been a mistake:
for _ in range(100):
    row = np.random.choice(interaction_matrix.shape[0]) #select random instance
    col = np.random.choice(interaction_matrix[row].nonzero()[0]) #select from positives of that instance
    assert tids[col] in list(df[df['ligand_ID']==cids[row]]['chembl_id'])

sparse.save_npz('y.npz', sparse.csr_matrix(interaction_matrix))
sparse.save_npz('dates.npz', sparse.csr_matrix(interaction_dates))

df.sort_values('ligand_ID').drop_duplicates(['ligand_ID'])[['ligand_ID', 'canonical_smiles']].to_csv('all_chemicals.csv', index=False)
df.sort_values('chembl_id').drop_duplicates(['chembl_id'])['pref_name'].to_csv('targets_prefname.dat', index=False, header=None)
df.sort_values('chembl_id').drop_duplicates(['chembl_id']).to_csv('targets.dat', index=False)
