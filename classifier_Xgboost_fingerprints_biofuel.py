import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen
from rdkit.Chem import rdMolDescriptors
import os

# 读取 CSV 文件
csv_file = 'molecules.csv'
df = pd.read_csv(csv_file)

# 定义一个函数来计算分子的某些性质
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

# 应用该函数到每一行
df_properties = df['SMILES'].apply(compute_properties)

# 检查计算结果
print(df_properties.head())

# 将结果合并到原始 DataFrame 中
df = pd.concat([df, df_properties.apply(pd.Series)], axis=1)

# 确保输出文件夹存在
output_folder = 'output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 保存结果到新的 CSV 文件
output_file = os.path.join(output_folder, 'molecules_properties.csv')
df.to_csv(output_file, index=False)

print(f'Molecular properties calculated and saved to {output_file}')









