import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置输出目录
output_dir = "XGBoost_output"
os.makedirs(output_dir, exist_ok=True)

# 加载模型评估指标和参数
baseline_metrics_df = pd.read_csv(os.path.join(output_dir, 'baseline_model_metrics.csv'))
final_metrics_df = pd.read_csv(os.path.join(output_dir, 'final_model_metrics.csv'))
best_params_df = pd.read_csv(os.path.join(output_dir, 'best_params.csv'))

# 加载重要特征的pairplot数据（如果有的话）
if os.path.exists(os.path.join(output_dir, 'important_features_pairplot.png')):
    important_features_pairplot = os.path.join(output_dir, 'important_features_pairplot.png')
else:
    important_features_pairplot = None

# 打印基本信息
print("Baseline Model Metrics:")
print(baseline_metrics_df)
print("\nFinal Model Metrics:")
print(final_metrics_df)
print("\nBest Hyperparameters:")
print(best_params_df)
