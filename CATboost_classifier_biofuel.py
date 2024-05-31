import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score, roc_curve, auc
from sklearn.feature_selection import VarianceThreshold
from catboost import CatBoostClassifier
from tpot import TPOTClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Create output directory
output_dir = "CATBoost_outputs"
os.makedirs(output_dir, exist_ok=True)

# Function to read data
def read_data(csv_file):
    print("Reading dataset...")
    df = pd.read_csv(csv_file)
    assert 'SMILES' in df.columns, "SMILES column is missing in the dataset"
    assert 'Type' in df.columns, "Type column is missing in the dataset"
    print("Dataset reading completed.")
    return df

# Function to convert SMILES to fingerprints
def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"RDKit could not parse SMILES: {smiles}"
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp)

# Function to apply the fingerprint function to the dataset
def apply_fingerprint(df):
    print("Applying fingerprint conversion...")
    df['Fingerprint'] = df['SMILES'].apply(smiles_to_fingerprint)
    X = np.stack(df['Fingerprint'])
    y = df['Type'].values
    smiles = df['SMILES'].values  # Store SMILES strings
    print("Fingerprint conversion completed.")
    return X, y, smiles

# Function to split the data into training and external test sets
def split_data(X, y, smiles):
    print("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(X, y, smiles, test_size=0.2, random_state=42)
    assert len(X_train) > 0 and len(X_test) > 0, "Training or testing set is empty"
    print("Data splitting completed.")
    return X_train, X_test, y_train, y_test, smiles_train, smiles_test

# Preprocessing functions
def low_variance_filter(X, threshold=0.01):
    selector = VarianceThreshold(threshold=threshold)
    X_filtered = selector.fit_transform(X)
    assert X_filtered.shape[1] > 0, "All features removed by low variance filter"
    return X_filtered

def correlation_filter(X_train, X_test, threshold=0.95):
    corr_matrix = np.corrcoef(X_train, rowvar=False)
    upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    to_drop = [column for column in range(corr_matrix.shape[1]) if any(corr_matrix[column, row] > threshold for row in range(corr_matrix.shape[0]))]
    X_train_filtered = np.delete(X_train, to_drop, axis=1)
    X_test_filtered = np.delete(X_test, to_drop, axis=1)
    assert X_train_filtered.shape[1] > 0, "All features removed by correlation filter"
    return X_train_filtered, X_test_filtered

def manipulate_fingerprints(smiles, radius=3, n_bits=4096):
    return np.array([smiles_to_fingerprint(s, radius=radius, n_bits=n_bits) for s in smiles])

# Function to preprocess data
def preprocess_data(X_train, X_test, smiles_train, smiles_test, apply_low_variance=True, apply_correlation_filter=True,
                    apply_fingerprint_manipulation=True):
    print("Starting preprocessing...")
    if apply_low_variance:
        print("Applying low variance filter...")
        X_train = low_variance_filter(X_train)
        X_test = low_variance_filter(X_test)  # This should use X_test
    if apply_correlation_filter:
        print("Applying correlation filter...")
        X_train, X_test = correlation_filter(X_train, X_test)
    if apply_fingerprint_manipulation:
        print("Applying fingerprint manipulation...")
        X_train = manipulate_fingerprints(smiles_train)
        X_test = manipulate_fingerprints(smiles_test)
    print("Preprocessing completed.")
    return X_train, X_test

# Function to compute all relevant metrics
def compute_metrics(y_true, y_pred, y_proba):
    y_true_bin = [1 if y == 'biofuel' else 0 for y in y_true]  # Ensure binary encoding
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true_bin, y_proba),  # Use binary labels for AUC calculation
        'f1_score': f1_score(y_true, y_pred, pos_label='biofuel'),
        'precision': precision_score(y_true, y_pred, pos_label='biofuel'),
        'recall': recall_score(y_true, y_pred, pos_label='biofuel'),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred, output_dict=True)
    }
    return metrics

# Function to plot and save ROC curve
def plot_and_save_roc(y_true, y_proba, filename='roc_curve.png'):
    # Ensure binary classification
    y_true_bin = [1 if y == 'biofuel' else 0 for y in y_true]
    fpr, tpr, thresholds = roc_curve(y_true_bin, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    return roc_auc

# Function to plot pairplot with the most important features
def plot_pairplot(df, features, target_column, filename='pairplot.png'):
    sns.pairplot(df[features + [target_column]], hue=target_column)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# Baseline model with no preprocessing
def baseline_model(X_train, y_train, X_test, y_test):
    print("Training baseline model...")
    model = CatBoostClassifier(verbose=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 0]
    metrics = compute_metrics(y_test, y_pred, y_proba)
    print("Baseline model training completed.")
    return model, metrics

# Hyperparameter optimization using TPOT with genetic algorithm
def optimize_hyperparameters(X, y):
    print("Starting hyperparameter optimization...")
    tpot = TPOTClassifier(generations=50, population_size=20, cv=5, random_state=42, verbosity=2, config_dict='TPOT sparse')
    tpot.fit(X, y)
    print("Hyperparameter optimization completed.")
    return tpot.fitted_pipeline_, tpot.fitted_pipeline_.get_params()

# Function to get important features
def get_important_features(model, feature_names, top_n=5):
    feature_importances = model.get_feature_importance()
    indices = np.argsort(feature_importances)[-top_n:]
    important_features = [feature_names[i] for i in indices]
    return important_features

# Main function
def main():
    csv_file = 'new_dataset.csv'  # Ensure there is no space in the filename
    perform_hyperparameter_optimization = True  # Boolean flag to control hyperparameter optimization

    # Read and preprocess data
    df = read_data(csv_file)
    X, y, smiles = apply_fingerprint(df)
    X_train, X_test, y_train, y_test, smiles_train, smiles_test = split_data(X, y, smiles)

    # Train and evaluate the baseline model
    baseline_model_result, baseline_metrics = baseline_model(X_train, y_train, X_test, y_test)

    print("Baseline Metrics:")
    for key, value in baseline_metrics.items():
        if key != 'confusion_matrix' and key != 'classification_report':
            print(f"{key}: {value}")
    print(f"Confusion Matrix:\n{baseline_metrics['confusion_matrix']}")
    print(f"Classification Report:\n{pd.DataFrame(baseline_metrics['classification_report']).transpose()}")

    # Save baseline metrics to CSV
    baseline_metrics_df = pd.DataFrame([baseline_metrics])
    baseline_metrics_df.to_csv(os.path.join(output_dir, 'baseline_model_metrics.csv'), index=False)

    # Plot and save ROC curve for the baseline model
    baseline_roc_auc = plot_and_save_roc(y_test, baseline_model_result.predict_proba(X_test)[:, 0], filename='baseline_roc_curve.png')

    # Print AUC for verification
    print(f"Baseline ROC AUC: {baseline_roc_auc}")

    # Preprocess the data with user options
    apply_low_variance = False
    apply_correlation_filter = False
    apply_fingerprint_manipulation = False

    X_train_preprocessed, X_test_preprocessed = preprocess_data(X_train, X_test, smiles_train, smiles_test, apply_low_variance,
                                                                apply_correlation_filter, apply_fingerprint_manipulation)
    assert X_train_preprocessed.shape[1] > 0 and X_test_preprocessed.shape[1] > 0, "Preprocessing removed all features"

    if perform_hyperparameter_optimization:
        # Perform hyperparameter optimization
        best_pipeline, best_params = optimize_hyperparameters(X_train_preprocessed, y_train)

        # Save the best hyperparameters
        best_params_df = pd.DataFrame([best_params])
        best_params_df.to_csv(os.path.join(output_dir, 'best_params.csv'), index=False)
        print("Best Hyperparameters:", best_params)
    else:
        best_params = {}

    # Remove any unexpected keyword arguments from best_params
    valid_params = CatBoostClassifier().get_params().keys()
    best_params = {k: v for k, v in best_params.items() if k in valid_params}

    # Retrain the model with the best hyperparameters
    best_model = CatBoostClassifier(**best_params, verbose=0)
    best_model.fit(X_train_preprocessed, y_train)
    y_pred = best_model.predict(X_test_preprocessed)
    y_proba = best_model.predict_proba(X_test_preprocessed)[:, 0]

    # Evaluate the final model
    final_metrics = compute_metrics(y_test, y_pred, y_proba)

    # Save the final metrics and confusion matrix
    final_metrics_df = pd.DataFrame([final_metrics])
    final_metrics_df.to_csv(os.path.join(output_dir, 'final_model_metrics.csv'), index=False)

    confusion_matrix_df = pd.DataFrame(final_metrics['confusion_matrix'],
                                       index=['Actual Negative', 'Actual Positive'],
                                       columns=['Predicted Negative', 'Predicted Positive'])
    confusion_matrix_df.to_csv(os.path.join(output_dir, 'final_confusion_matrix.csv'))

    print("Final Model Metrics:")
    for key, value in final_metrics.items():
        if key != 'confusion_matrix' and key != 'classification_report':
            print(f"{key}: {value}")
    print(f"Confusion Matrix:\n{final_metrics['confusion_matrix']}")
    print(f"Classification Report:\n{pd.DataFrame(final_metrics['classification_report']).transpose()}")

    # Plot and save ROC curve for the final model
    final_roc_auc = plot_and_save_roc(y_test, best_model.predict_proba(X_test_preprocessed)[:, 0], filename='final_roc_curve.png')

    # Print AUC for verification
    print(f"Final ROC AUC: {final_roc_auc}")

    # Plot pairplot with the most important features
    important_features = get_important_features(best_model, range(X_train_preprocessed.shape[1]), top_n=5)
    df_imp_features = pd.DataFrame(X_train_preprocessed, columns=range(X_train_preprocessed.shape[1]))
    df_imp_features['Type'] = y_train

    plot_pairplot(df_imp_features, important_features, 'Type', filename='important_features_pairplot.png')

    # Additional assertions for model evaluation and data integrity
    assert len(important_features) == 5, "Important features extraction did not return 5 features"
    assert len(final_metrics) > 0, "Final metrics are not computed"
    assert best_model is not None, "Best model training failed"
    assert baseline_model_result is not None, "Baseline model training failed"
    assert X_train_preprocessed.shape[0] == X_train.shape[0], "Preprocessing changed the number of training samples"
    assert X_test.shape[0] > 0, "Test set is empty after split"
    assert y_pred is not None, "Prediction failed on the test set"
    assert y_proba is not None, "Probability prediction failed on the test set"
    assert 'accuracy' in final_metrics, "Accuracy metric is missing in the final metrics"
    assert 'roc_auc' in final_metrics, "ROC AUC metric is missing in the final metrics"

if __name__ == "__main__":
    main()
