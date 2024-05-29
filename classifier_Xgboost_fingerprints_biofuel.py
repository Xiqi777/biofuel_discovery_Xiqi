import deepchem
from xgboost import XGBClassifier, XGBRegressor

# For classification
model = XGBClassifier()

model.fit(X_train, y_train)

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from sklearn.metrics import pairwise_distances

# 读取Excel文件
file_path = 'your_excel_file.xlsx'
df = pd.read_excel(file_path)

# 假设Excel文件有两列：SMILES 和 Type（biofuel 或 not biofuel）
smiles_list = df['SMILES']
fuel_type = df['Type']

# 计算分子指纹
def calculate_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        return fingerprint
    else:
        return None

# 计算所有分子的指纹
fingerprints = [calculate_fingerprint(smiles) for smiles in smiles_list]

# 过滤掉无法计算指纹的分子
valid_indices = [i for i, fp in enumerate(fingerprints) if fp is not None]
fingerprints = [fingerprints[i] for i in valid_indices]
fuel_type = fuel_type[valid_indices]
smiles_list = smiles_list[valid_indices]

# 将指纹转换为numpy数组
fingerprint_array = []
for fp in fingerprints:
    arr = [0] * 2048
    DataStructs.ConvertToNumpyArray(fp, arr)
    fingerprint_array.append(arr)

# 转换为DataFrame
fingerprint_df = pd.DataFrame(fingerprint_array)

# 添加燃料类型列
fingerprint_df['Type'] = fuel_type.values

# 计算生物燃料和非生物燃料之间的距离矩阵
biofuel_fps = fingerprint_df[fingerprint_df['Type'] == 'biofuel'].drop('Type', axis=1)
not_biofuel_fps = fingerprint_df[fingerprint_df['Type'] == 'not biofuel'].drop('Type', axis=1)

# 使用Tanimoto相似度来计算距离矩阵
distance_matrix = pairwise_distances(biofuel_fps, not_biofuel_fps, metric='jaccard')

print(distance_matrix)
