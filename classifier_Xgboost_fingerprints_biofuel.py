import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# Read the CSV file
csv_file = 'new dataset.csv'
df = pd.read_csv(csv_file)

# Define a function to compute properties of molecules
def compute_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    properties = {
        'CAS Number': Chem.MolToSmiles(mol),
        'MeltingPoint': Descriptors.MolWt(mol),
        'BoilingPoint': Descriptors.ExactMolWt(mol),
        'LHV': Descriptors.MolWt(mol),
        'FlashPoint': Descriptors.MolWt(mol),
        'Density': Descriptors.MolWt(mol),
        'OC': Descriptors.MolWt(mol),
        'MolWt': Descriptors.MolWt(mol),
        'LFL': Descriptors.MolWt(mol),
        'UFL': Descriptors.MolWt(mol)
    }
    return properties

# Apply the function to each row
df_properties = df['SMILES'].apply(compute_properties)

# Merge the results into the original DataFrame
df = pd.concat([df, df_properties.apply(pd.Series)], axis=1)

# Save the results to a new CSV file
df.to_csv('new molecules_with_properties.csv', index=False)

print('Molecular properties calculated and saved to new molecules_with_properties.csv')








