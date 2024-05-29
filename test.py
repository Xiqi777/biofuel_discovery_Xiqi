import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# 读取 CSV 文件
csv_file = 'new dataset.csv'
df = pd.read_csv(csv_file)

# 定义一个函数来计算分子的某些性质
def compute_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f'Error: Invalid SMILES string encountered: {smiles}')
        return None
    else:
        properties = {
            'CAS': Chem.MolToSmiles(mol),
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

# 应用该函数到每一行
df_properties = df['SMILES'].apply(compute_properties)

# 将结果合并到原始 DataFrame 中
df = pd.concat([df, df_properties.apply(pd.Series)], axis=1)

# 保存结果到新的 CSV 文件
df.to_csv('molecules_with_properties.csv', index=False)

print('分子性质计算完成，并已保存到 molecules_with_properties.csv')

