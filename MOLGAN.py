import deepchem
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from collections import OrderedDict

import deepchem as dc
import deepchem.models
import torch
from deepchem.models.torch_models import BasicMolGANModel as MolGAN
from deepchem.models.optimizers import ExponentialDecay
from torch.nn.functional import one_hot
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw

from deepchem.feat.molecule_featurizers.molgan_featurizer import GraphMatrix

# Function to read the dataset
def read_data(csv_file):
    print("Reading dataset...")
    df = pd.read_csv(csv_file)
    assert 'smiles' in df.columns, "smiles column is missing in the dataset"
    print("Dataset reading completed.")
    return df

# Set the file name
csv_file = 'new dataset（biofuel）.csv'

# Read data
df = read_data(csv_file)

# Set the maximum number of atoms
num_atoms = 12

# Create featurizer
feat = dc.feat.MolGanFeaturizer(max_atom_count=num_atoms, atom_labels=[0, 5, 6, 7, 8, 9, 11, 12, 13, 14])

# Extract SMILES
smiles = df['smiles'].values
filtered_smiles = [x for x in smiles if Chem.MolFromSmiles(x).GetNumAtoms() < num_atoms]

# Featurize molecules
features = feat.featurize(filtered_smiles)
indices = [i for i, data in enumerate(features) if isinstance(data, GraphMatrix)]
print(indices)
features = [features[i] for i in indices]

# Create model
gan = MolGAN(learning_rate=ExponentialDecay(0.001, 0.9, 5000), vertices=num_atoms)
dataset = dc.data.NumpyDataset([x.adjacency_matrix for x in features], [x.node_features for x in features])

# Define iterator
def iterbatches(epochs):
    for i in range(epochs):
        for batch in dataset.iterbatches(batch_size=gan.batch_size, pad_batches=True):
            flattened_adjacency = torch.from_numpy(batch[0]).view(-1).to(dtype=torch.int64)
            invalid_mask = (flattened_adjacency < 0) | (flattened_adjacency >= gan.edges)
            clamped_adjacency = torch.clamp(flattened_adjacency, 0, gan.edges-1)
            adjacency_tensor = one_hot(clamped_adjacency, num_classes=gan.edges)
            adjacency_tensor[invalid_mask] = torch.zeros(gan.edges, dtype=torch.long)
            adjacency_tensor = adjacency_tensor.view(*batch[0].shape, -1)

            flattened_node = torch.from_numpy(batch[1]).view(-1).to(dtype=torch.int64)
            invalid_mask = (flattened_node < 0) | (flattened_node >= gan.nodes)
            clamped_node = torch.clamp(flattened_node, 0, gan.nodes-1)
            node_tensor = one_hot(clamped_node, num_classes=gan.nodes)
            node_tensor[invalid_mask] = torch.zeros(gan.nodes, dtype=torch.long)
            node_tensor = node_tensor.view(*batch[1].shape, -1)

            yield {gan.data_inputs[0]: adjacency_tensor, gan.data_inputs[1]: node_tensor}

# Train the GAN model
gan.fit_gan(iterbatches(25), generator_steps=0.2, checkpoint_interval=5000)

# Generate data
generated_data = gan.predict_gan_generator(1000)
nmols = feat.defeaturize(generated_data)
print("{} molecules generated".format(len(nmols)))
nmols = list(filter(lambda x: x is not None, nmols))
print("{} valid molecules".format(len(nmols)))

# Extract and display unique valid SMILES
nmols_smiles = [Chem.MolToSmiles(m) for m in nmols]
nmols_smiles_unique = list(OrderedDict.fromkeys(nmols_smiles))
nmols_viz = [Chem.MolFromSmiles(x) for x in nmols_smiles_unique]
print("{} unique valid molecules".format(len(nmols_viz)))

# Display molecules in a grid
img = Draw.MolsToGridImage(nmols_viz[0:100], molsPerRow=5, subImgSize=(250, 250), maxMols=100, legends=None, returnPNG=False)
