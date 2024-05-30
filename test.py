import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen
from rdkit.Chem import rdMolDescriptors
import os

# Read the CSV file
csv_file = 'new dataset.csv'
df = pd.read_csv(csv_file)

# Define a function to compute molecular properties
def compute_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f'Error: Invalid SMILES string encountered: {smiles}')
        return None
    else:
        properties = {
            'CAS': Chem.MolToSmiles(mol),  # Placeholder for CAS number
            'MolWt': Descriptors.MolWt(mol),
            'ExactMolWt': Descriptors.ExactMolWt(mol),
            'LogP': Crippen.MolLogP(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'TPSA': rdMolDescriptors.CalcTPSA(mol),
            'MolVolume': Descriptors.MolMR(mol)  # Molecular volume can be approximated by the molar refractivity
        }
        print(f'Properties for {smiles}: {properties}')
        return properties

# Apply the function to each row
df_properties = df['SMILES'].apply(compute_properties)

# Check the computed results
print(df_properties.head())

# Merge the results into the original DataFrame
df = pd.concat([df, df_properties.apply(pd.Series)], axis=1)

# Ensure the output folder exists
output_folder = '.venv'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Save the results to a new CSV file
output_file = os.path.join(output_folder, 'molecules_properties.csv')
df.to_csv(output_file, index=False)

print(f'Molecular properties calculated and saved to {output_file}')


