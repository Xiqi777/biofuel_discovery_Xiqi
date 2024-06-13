import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

# Attempt to read CSV file using different encodings
try:
    df = pd.read_csv('new dataset.csv', encoding='latin1')
except UnicodeDecodeError:
    df = pd.read_csv('new dataset.csv', encoding='iso-8859-1')

# Initialize list for storing results
results = []

# Iterate through each row to process molecular information
for index, row in df.iterrows():
    smiles = row['SMILES']
    mol = Chem.MolFromSmiles(smiles)

    if mol is not None:
        # Calculate fingerprint
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

        # Calculate physicochemical properties
        mol_weight = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        num_h_donors = Descriptors.NumHDonors(mol)
        num_h_acceptors = Descriptors.NumHAcceptors(mol)
        tpsa = Descriptors.TPSA(mol)
        num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        num_aromatic_rings = Descriptors.NumAromaticRings(mol)
        num_saturated_rings = Descriptors.NumSaturatedRings(mol)
        mol_volume = Descriptors.MolMR(mol)  # Molecular volume (approximation using refractivity)
        heavy_atom_count = Descriptors.HeavyAtomCount(mol)
        num_valence_electrons = Descriptors.NumValenceElectrons(mol)

        # Append results to the list
        results.append({
            'Molecule': row['Molecule'],
            'SMILES': smiles,
            'Type': row['Type'],
            'Fingerprint': fingerprint.ToBitString(),
            'MolWeight': mol_weight,
            'LogP': logp,
            'NumHDonors': num_h_donors,
            'NumHAcceptors': num_h_acceptors,
            'TPSA': tpsa,
            'NumRotatableBonds': num_rotatable_bonds,
            'NumAromaticRings': num_aromatic_rings,
            'NumSaturatedRings': num_saturated_rings,
            'MolVolume': mol_volume,
            'HeavyAtomCount': heavy_atom_count,
            'NumValenceElectrons': num_valence_electrons
        })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Save results to a new CSV file
results_df.to_csv('datasets_with_properties.csv', index=False)



