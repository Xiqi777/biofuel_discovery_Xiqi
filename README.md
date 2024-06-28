# Generative Adversarial Network for Novel Biofuel Design

This project aims to design novel biofuel molecules using Generative Adversarial Networks (GANs). The main process of the project includes data collection, classifier development, fuel classification, and generation of novel biofuel molecules.

## Table of Contents

- [Background](#background)
- [File_Structure](#file_structure)
- [Project_Steps](#project_steps)
- [Scripts_Usage](#scripts_usage)
- [Results](#results)
- [Contribution](#contribution)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Background

The global energy landscape is changing, with sustainable and renewable energy sources receiving increasing attention. Biofuels, derived from biomass, have the potential to replace fossil fuels, aiding in energy transition and mitigating climate change. However, the production of biofuels still faces technical, economic, and environmental challenges. To address these issues, the use of machine learning techniques, such as Generative Adversarial Networks (GANs), has become crucial. This project involves collecting SMILES data of biofuel and non-biofuel molecules, using XGBoost and CATBoost to build classifiers, extracting useful biofuel molecules from a large biomass database, and employing MOLGAN to generate new biofuel molecules. The newly generated molecules are then integrated into the original database, and the classifiers are retrained, forming a closed-loop iterative process that continuously optimizes and develops biofuels with desired properties.

## File_Structure

```bash
new datasets.csv               # The SMILES and Type of biofuels and non-biofuels molecules
classifier_Xgboost_fingerprint_biofuel.py # Code for using XGBoost to create a classifier
CATboost_classifier_biofuel.py # Code for using CATBoost to create a classifier
XGBoost_outputs/               # Results obtained by running the latest and currently used datasets with XGBoost
CATBoost_outputs/              # Results obtained by running the latest and currently used datasets with CATBoost
biomass.csv                    # A large dataset of molecule SMILES derived from biomass
biofuels.csv                   # Used to run MOLGAN on Myriad to generate additional potential biofuel molecules
MOLGAN.py                      # Code for using MOLGAN to create potential biofuel molecules
```

## Project_Steps

1. **Data Collection**
   - Collect data on biofuel and non-biofuel molecules to build a small dataset containing SMILES strings of biofuels and non-biofuels. The dataset is stored in `new dataset.csv`.
   - File used: `new dataset.csv`

2. **Classifier Development**
   - Use XGBoost and CATBoost to run the scripts `classifier_Xgboost_fingerprint_biofuel.py` and `CATboost_classifier_biofuel.py` to gain the fingerprints of molecules in `new dataset.csv`, and create classifiers that can differentiate between biofuels and non-biofuels.
   - Files used: `classifier_Xgboost_fingerprint_biofuel.py`, `CATboost_classifier_biofuel.py`
   - Input: `new dataset.csv`
   - Output: `XGBoost_outputs/`, `CATBoost_outputs/`

3. **Classification of Large Biomass Dataset**
   - Use the established classifiers to classify a large biomass dataset stored in `biomass.csv`. This step helps in extracting usable biofuel molecules' SMILES.
   - Files used: `classifier_Xgboost_fingerprint_biofuel.py`, `CATboost_classifier_biofuel.py`
   - Input: `biomass.csv`
   - Output: SMILES of biofuel molecules extracted from `biomass.csv`

4. **Creating a Biofuel Dataset**
   - Combine the biofuel SMILES from `new datasets.csv` and the SMILES extracted in step 4 to create a biofuel database named `biofuels.csv`, which will be provided to MOLGAN.
   - Files used: Data from previous steps
   - Input: The SMILES of biofuel molecules from step 1 and step 4
   - Output: `biofuels.csv`

5. **Generation of Novel Biofuel Molecules**
   - Run `MOLGAN.py` on Myriad to generate more potential novel biofuel molecules using the biofuel database `biofuels.csv`.
   - File used: `MOLGAN.py`
   - Input: `biofuels.csv`, `MOLGAN.py`
   - Output: Generated biofuel molecules

6. **Integration and Iteration**
   - Integrate the generated biofuel molecules from step 5 into the small dataset `new dataset.csv`. Re-establish classifiers and repeat steps 2, 3, 4, and 5 to form a closed loop between MOLGAN and classifier development.


## Scripts_Usage

Here are the detailed instructions on how to use these codes:

### 1. `classifier_Xgboost_fingerprint_biofuel.py`

This segment creates a directory named "XGBoost_output" to store output files. You can change the `output_dir` variable to a different directory name if you prefer.

```bash
# Create output directory
output_dir = "XGBoost_output"
os.makedirs(output_dir, exist_ok=True)
```
This function reads a CSV file into a DataFrame and ensures the presence of necessary columns. You can change the `csv_file` parameter to read a different dataset file.

```bash
# Function to read data
def read_data(csv_file):
    print("Reading dataset...")
    df = pd.read_csv(csv_file)
    assert 'SMILES' in df.columns, "SMILES column is missing in the dataset"
    assert 'Type' in df.columns, "Type column is missing in the dataset"
    print("Dataset reading completed.")
    return df
```

This function encodes categorical labels into numerical values using `LabelEncoder`. 

```bash
# Function to encode the target labels
def encode_labels(y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le
```

This function converts SMILES strings into molecular fingerprints using RDKit. You can modify the `radius` and `n_bits` parameters to change the fingerprint generation process.

```bash
# Function to convert SMILES to fingerprints
def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"RDKit could not parse SMILES: {smiles}"
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp)
```

This function applies the fingerprint conversion to the dataset, encoding the target labels and returning the features, labels, and SMILES strings.

```bash
# Function to apply the fingerprint function to the dataset
def apply_fingerprint(df):
    print("Applying fingerprint conversion...")
    df['Fingerprint'] = df['SMILES'].apply(smiles_to_fingerprint)
    X = np.stack(df['Fingerprint'])
    y, label_encoder = encode_labels(df['Type'].values)
    smiles = df['SMILES'].values  # Store SMILES strings
    print("Fingerprint conversion completed.")
    return X, y, smiles, label_encoder
```

This function splits the data into training and test sets. You can modify the `test_size` parameter to change the proportion of data used for testing.

```bash
# Function to split the data into training and external test sets
def split_data(X, y, smiles):
    print("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(X, y, smiles, test_size=0.2, random_state=42)
    assert len(X_train) > 0 and len(X_test) > 0, "Training or testing set is empty"
    print("Data splitting completed.")
    return X_train, X_test, y_train, y_test, smiles_train, smiles_test
```

These functions apply various preprocessing steps, including low variance filtering, correlation filtering, and fingerprint manipulation.

```bash
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
```

This function orchestrates the preprocessing steps based on user-defined options. You can set `apply_low_variance`, `apply_correlation_filter`, and `apply_fingerprint_manipulation` to `True` or `False` to control which preprocessing steps are applied.

```bash
# Function to preprocess data
def preprocess_data(X_train, X_test, smiles_train, smiles_test, apply_low_variance=True, apply_correlation_filter=True, apply_fingerprint_manipulation=True):
    print("Starting preprocessing...")
    if apply_low_variance:
        print("Applying low variance filter...")
        X_train = low_variance_filter(X_train)
        X_test = low_variance_filter(X_test)
    if apply_correlation_filter:
        print("Applying correlation filter...")
        X_train, X_test = correlation_filter(X_train, X_test)
    if apply_fingerprint_manipulation:
        print("Applying fingerprint manipulation...")
        X_train = manipulate_fingerprints(smiles_train)
        X_test = manipulate_fingerprints(smiles_test)
    print("Preprocessing completed.")
    return X_train, X_test
```

This function computes various performance metrics for the classification model. 

```bash
# Function to compute all relevant metrics
def compute_metrics(y_true, y_pred, y_proba, label_encoder):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'f1_score': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred, output_dict=True, target_names=label_encoder.classes_)
    }
    return metrics
```

This function trains the baseline XGBoost model and evaluates its performance. You can modify the `XGBClassifier` parameters if needed for different model configurations.

```bash
# Function to train the baseline model
def baseline_model(X_train, y_train, X_test, y_test, label_encoder):
    print("Training baseline model...")
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    assert len(np.unique(y_pred)) > 1, "Prediction resulted in only one class"
    assert y_proba.min() >= 0 and y_proba.max() <= 1, "Probabilities are not between 0 and 1"
    print(f"Debug: y_test: {y_test[:10]}, y_proba: {y_proba[:10]}")
    metrics = compute_metrics(y_test, y_pred, y_proba, label_encoder)
    print("Baseline model training completed.")
    return model, metrics
```

This function plots and saves the ROC curve for the model's performance. You can change the `filename` parameter to save the plot with a different name.

```bash
# Function to plot and save ROC curve
def plot_and_save_roc(y_true, y_proba, label_encoder, filename='roc_curve.png'):
    pos_label = label_encoder.transform(['biofuel'])[0]
    fpr, tpr, thresholds = roc_curve(y_true, y_proba, pos_label=pos_label)
    print(f"Debug: FPR: {fpr}, TPR: {tpr}, Thresholds: {thresholds}")
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
```

This function creates and saves a pairplot of the most important features. You can change the `filename` parameter to save the plot with a different name.

```bash
# Function to plot pairplot with the most important features
def plot_pairplot(df, features, target_column, filename='pairplot.png'):
    sns.pairplot(df[features + [target_column]], hue=target_column)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
```

This function optimizes hyperparameters using TPOT, which employs genetic algorithms. You can modify `generations`, `population_size`, `cv`, and `config_dict` for different optimization settings.

```bash
# Hyperparameter optimization using TPOT with genetic algorithm
def optimize_hyperparameters(X, y):
    print("Starting hyperparameter optimization...")
    tpot = TPOTClassifier(generations=50, population_size=20, cv=5, random_state=42, verbosity=2, config_dict='TPOT sparse')
    tpot.fit(X, y)
    print("Hyperparameter optimization completed.")
    return tpot.fitted_pipeline_, tpot.fitted_pipeline_.get_params()
```

This function extracts the most important features from the model. You can modify the `top_n` parameter to get a different number of top features.

```bash
# Function to get important features
def get_important_features(model, feature_names, top_n=5):
    feature_importances = model.feature_importances_
    indices = np.argsort(feature_importances)[-top_n:]
    important_features = [feature_names[i] for i in indices]
    return important_features
```

This section of code specifies the CSV file to be read and sets a boolean flag to determine whether hyperparameter optimization should be performed.

```bash
# Set CSV file name
csv_file = 'new dataset.csv'
# Boolean flag to control hyperparameter optimization
perform_hyperparameter_optimization = True  # Set this to True if you want to perform hyperparameter optimization
```

This code segment reads the specified CSV file and applies fingerprinting to extract features. It then splits the dataset into training and testing sets and prints the classes of labels.

```bash
# Read and preprocess data
df = read_data(csv_file)
X, y, smiles, label_encoder = apply_fingerprint(df)
X_train, X_test, y_train, y_test, smiles_train, smiles_test = split_data(X, y, smiles)
print("Label classes:", label_encoder.classes_)
```

This section of code preprocesses the data based on user-defined options. You can choose whether to apply low variance filtering, correlation filtering, and fingerprint manipulation by setting the corresponding flags (`apply_low_variance`, `apply_correlation_filter`, `apply_fingerprint_manipulation`).

```bash
# Preprocess the data with user options
apply_low_variance = False
apply_correlation_filter = False
apply_fingerprint_manipulation = True  # Set this to True if you want to manipulate fingerprints

X_train_preprocessed, X_test_preprocessed = preprocess_data(X_train, X_test, smiles_train, smiles_test, apply_low_variance,
                                                            apply_correlation_filter, apply_fingerprint_manipulation)
assert X_train_preprocessed.shape[1] > 0 and X_test_preprocessed.shape[1] > 0, "Preprocessing removed all features"
```

This code segment trains the baseline XGBoost model and evaluates its performance. It prints out metrics such as accuracy, ROC AUC, confusion matrix, and classification report.

```bash
# Train and evaluate the baseline model with preprocessed data
baseline_model_result, baseline_metrics = baseline_model(X_train_preprocessed, y_train, X_test_preprocessed, y_test, label_encoder)

print("Baseline Model Metrics:")
for key, value in baseline_metrics.items():
    if key != 'confusion_matrix' and key != 'classification_report':
        print(f"{key}: {value}")
print(f"Confusion Matrix:\n{baseline_metrics['confusion_matrix']}")
print(f"Classification Report:\n{pd.DataFrame(baseline_metrics['classification_report']).transpose()}")
```

This part of the code saves the metrics of the baseline model to a CSV file named `baseline_model_metrics.csv`.

```bash
# Save baseline metrics to CSV
baseline_metrics_df = pd.DataFrame([baseline_metrics])
baseline_metrics_df.to_csv(os.path.join(output_dir, 'baseline_model_metrics.csv'), index=False)
```

If `perform_hyperparameter_optimization` is True, this code section performs hyperparameter optimization and saves the best parameters to `best_params.csv`. Otherwise, it initializes `best_params` as an empty dictionary.

```bash
# Perform hyperparameter optimization if perform_hyperparameter_optimization is set to True
if perform_hyperparameter_optimization:
    best_pipeline, best_params = optimize_hyperparameters(X_train_preprocessed, y_train)

    # Save the best hyperparameters
    best_params_df = pd.DataFrame([best_params])
    best_params_df.to_csv(os.path.join(output_dir, 'best_params.csv'), index=False)
    print("Best Hyperparameters:", best_params)
else:
    best_params = {}
```

This code snippet removes any unexpected keyword arguments from `best_params` and retrains the XGBoost model with the best hyperparameters.

```bash
# Remove any unexpected keyword arguments from best_params
valid_params = XGBClassifier().get_params().keys()
best_params = {k: v for k, v in best_params.items() if k in valid_params}

# Retrain the model with the best hyperparameters
best_model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
best_model.fit(X_train_preprocessed, y_train)
y_pred = best_model.predict(X_test_preprocessed)
y_proba = best_model.predict_proba(X_test_preprocessed)[:, 1]  # Use index 1 for the positive class
```

This part of the code computes and prints various evaluation metrics of the final model, such as accuracy, ROC AUC, confusion matrix, and classification report. It also saves these metrics to CSV files (`final_model_metrics.csv` and `final_confusion_matrix.csv`).

```bash
# Evaluate the final model
final_metrics = compute_metrics(y_test, y_pred, y_proba, label_encoder)

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
```

This code snippet plots the pair plot of the most important features and includes additional assertion statements to ensure model evaluation and data integrity.

```bash
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
```

### 2. `CATboost_classifier_biofuel.py`

This code snippet creates an output directory named XGBoost_output to store the generated output files. You can change the output path by modifying the output_dir part.

```bash
# Create output directory
output_dir = "XGBoost_output"
os.makedirs(output_dir, exist_ok=True)
```

The `read_data` function is used to read a CSV file and return a DataFrame object containing the data. The function checks if the dataset contains the 'SMILES' and 'Type' columns, and if either is missing, it raises an assertion error.

```bash
# Function to read data
def read_data(csv_file):
    print("Reading dataset...")
    df = pd.read_csv(csv_file)
    assert 'SMILES' in df.columns, "SMILES column is missing in the dataset"
    assert 'Type' in df.columns, "Type column is missing in the dataset"
    print("Dataset reading completed.")
    return df
```

The `encode_labels` function encodes the target labels using a `LabelEncoder` object, which converts categorical variables into numerical labels. It returns the encoded label array and the `LabelEncoder` object.
```bash
# Function to encode the target labels
def encode_labels(y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le
```

The `smiles_to_fingerprint` function converts molecules represented by SMILES into Morgan fingerprints. It uses RDKit to extract molecular fingerprints with a specified radius and number of bits, and returns the fingerprints as a NumPy array.

```bash
# Function to convert SMILES to fingerprints
def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"RDKit could not parse SMILES: {smiles}"
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp)
```

Convert all SMILES strings in the DataFrame to molecular fingerprints, and return the feature matrix, label array, SMILES array, and label encoder.

```bash
# Function to apply the fingerprint function to the dataset
def apply_fingerprint(df):
    print("Applying fingerprint conversion...")
    df['Fingerprint'] = df['SMILES'].apply(smiles_to_fingerprint)
    X = np.stack(df['Fingerprint'])
    y, label_encoder = encode_labels(df['Type'].values)
    smiles = df['SMILES'].values  # Store SMILES strings
    print("Fingerprint conversion completed.")
    return X, y, smiles, label_encoder
```

The split_data function is used to divide the dataset (X, y, smiles) into training and test sets, and returns the split training and test feature data, label data, and SMILES string arrays.

```bash
# Function to split the data into training and external test sets
def split_data(X, y, smiles):
    print("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(X, y, smiles, test_size=0.2, random_state=42)
    assert len(X_train) > 0 and len(X_test) > 0, "Training or testing set is empty"
    print("Data splitting completed.")
    return X_train, X_test, y_train, y_test, smiles_train, smiles_test
```

Remove features with variance below the threshold to reduce noise in the data. The sensitivity of feature selection can be adjusted by changing the `threshold` value.

```bash
# Preprocessing functions

def low_variance_filter(X, threshold=0.01):
    selector = VarianceThreshold(threshold=threshold)
    X_filtered = selector.fit_transform(X)
    assert X_filtered.shape[1] > 0, "All features removed by low variance filter"
    return X_filtered
```

Remove features in the training set with correlation above the threshold to reduce feature redundancy. The strictness of the correlation filter can be adjusted by changing the `threshold` value.

```bash
def correlation_filter(X_train, X_test, threshold=0.95):
    corr_matrix = np.corrcoef(X_train, rowvar=False)
    upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    to_drop = [column for column in range(corr_matrix.shape[1]) if any(corr_matrix[column, row] > threshold for row in range(corr_matrix.shape[0]))]
    X_train_filtered = np.delete(X_train, to_drop, axis=1)
    X_test_filtered = np.delete(X_test, to_drop, axis=1)
    assert X_train_filtered.shape[1] > 0, "All features removed by correlation filter"
    return X_train_filtered, X_test_filtered
```

Convert SMILES to fingerprints based on the given radius and number of bits. The radius for fingerprint calculation can be adjusted by changing the `radius` value, and the number of bits can be modified by changing the `n_bits` value.

```bash
def manipulate_fingerprints(smiles, radius=3, n_bits=4096):
    return np.array([smiles_to_fingerprint(s, radius=radius, n_bits=n_bits) for s in smiles])
```

The `preprocess_data` function calls the previously defined feature selection and fingerprint processing functions, preprocesses the data based on user-specified options, and returns the preprocessed training and test feature datasets.

```bash
# Function to preprocess data
def preprocess_data(X_train, X_test, smiles_train, smiles_test, apply_low_variance=True, apply_correlation_filter=True, apply_fingerprint_manipulation=True):
    print("Starting preprocessing...")
    if apply_low_variance:
        print("Applying low variance filter...")
        X_train = low_variance_filter(X_train)
        X_test = low_variance_filter(X_test)
    if apply_correlation_filter:
        print("Applying correlation filter...")
        X_train, X_test = correlation_filter(X_train, X_test)
    if apply_fingerprint_manipulation:
        print("Applying fingerprint manipulation...")
        X_train = manipulate_fingerprints(smiles_train)
        X_test = manipulate_fingerprints(smiles_test)
    print("Preprocessing completed.")
    return X_train, X_test
```

Calculate various evaluation metrics such as accuracy, ROC AUC, F1 score, precision, recall, confusion matrix, and classification report.

```bash
# Function to compute all relevant metrics
def compute_metrics(y_true, y_pred, y_proba, label_encoder):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'f1_score': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred, output_dict=True, target_names=label_encoder.classes_)
    }
    return metrics
```

Train a baseline XGBoost model, predict on the test set, and calculate evaluation metrics.

```bash
# Adjust the baseline_model function to pass the label encoder to compute_metrics
def baseline_model(X_train, y_train, X_test, y_test, label_encoder):
    print("Training baseline model...")
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    # Additional assertions
    assert len(np.unique(y_pred)) > 1, "Prediction resulted in only one class"
    assert y_proba.min() >= 0 and y_proba.max() <= 1, "Probabilities are not between 0 and 1"
    print(f"Debug: y_test: {y_test[:10]}, y_proba: {y_proba[:10]}")
    metrics = compute_metrics(y_test, y_pred, y_proba, label_encoder)
    print("Baseline model training completed.")
    return model, metrics
```

Plot and Save ROC Curve. 

```bash
# Function to plot and save ROC curve
def plot_and_save_roc(y_true, y_proba, label_encoder, filename='roc_curve.png'):
    # Find the positive label
    pos_label = label_encoder.transform(['biofuel'])[0]
    fpr, tpr, thresholds = roc_curve(y_true, y_proba, pos_label=pos_label)
    print(f"Debug: FPR: {fpr}, TPR: {tpr}, Thresholds: {thresholds}")
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
```

plot a Pairplot with the most important features and color the data points according to the `target_column`.

```bash
def plot_pairplot(df, features, target_column, filename='pairplot.png'):
    sns.pairplot(df[features + [target_column]], hue=target_column)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
```

Optimize the hyperparameters of the model using TPOT

```bash
def optimize_hyperparameters(X, y):
    print("Starting hyperparameter optimization...")
    tpot = TPOTClassifier(generations=50, population_size=20, cv=5, random_state=42, verbosity=2, config_dict='TPOT sparse')
    tpot.fit(X, y)
    print("Hyperparameter optimization completed.")
    return tpot.fitted_pipeline_, tpot.fitted_pipeline_.get_params()
```

Get the most important features from the model.

```bash
def get_important_features(model, feature_names, top_n=5):
    feature_importances = model.feature_importances_
    indices = np.argsort(feature_importances)[-top_n:]
    important_features = [feature_names[i] for i in indices]
    return important_features
```

Complete the entire process of data reading, preprocessing, model training, evaluation, and results saving. This includes training the baseline model, optimizing the model, evaluating performance, and visualizing results.

```bash
def main():
    csv_file = 'new dataset.csv'
    perform_hyperparameter_optimization = True  # Boolean flag to control hyperparameter optimization

    # Read and preprocess data
    df = read_data(csv_file)
    X, y, smiles, label_encoder = apply_fingerprint(df)
    X_train, X_test, y_train, y_test, smiles_train, smiles_test = split_data(X, y, smiles)
    print("Label classes:", label_encoder.classes_)

    # Preprocess the data with user options
    apply_low_variance = False
    apply_correlation_filter = False
    apply_fingerprint_manipulation = True  # Ensure this is set to True

    X_train_preprocessed, X_test_preprocessed = preprocess_data(X_train, X_test, smiles_train, smiles_test, apply_low_variance,
                                                                apply_correlation_filter, apply_fingerprint_manipulation)
    assert X_train_preprocessed.shape[1] > 0 and X_test_preprocessed.shape[1] > 0, "Preprocessing removed all features"

    # Train and evaluate the baseline model with preprocessed data
    baseline_model_result, baseline_metrics = baseline_model(X_train_preprocessed, y_train, X_test_preprocessed, y_test, label_encoder)

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
    plot_and_save_roc(y_test, baseline_model_result.predict_proba(X_test_preprocessed)[:, 0], label_encoder, filename='baseline_roc_curve.png')

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
    valid_params = XGBClassifier().get_params().keys()
    best_params = {k: v for k, v in best_params.items() if k in valid_params}

    # Retrain the model with the best hyperparameters
    best_model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
    best_model.fit(X_train_preprocessed, y_train)
    y_pred = best_model.predict(X_test_preprocessed)
    y_proba = best_model.predict_proba(X_test_preprocessed)[:, 1]  # Use index 1 for the positive class

    # Additional assertions
    assert len(np.unique(y_pred)) > 1, "Prediction resulted in only one class"
    assert y_proba.min() >= 0 and y_proba.max() <= 1, "Probabilities are not between 0 and 1"
    print(f"Debug: y_test: {y_test[:10]}, y_proba: {y_proba[:10]}")

    # Evaluate the final model
    final_metrics = compute_metrics(y_test, y_pred, y_proba, label_encoder)

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
    plot_and_save_roc(y_test, best_model.predict_proba(X_test_preprocessed)[:, 0], label_encoder, filename='final_roc_curve.png')

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
```

### 3. `MOLGAN.py`


## Results

### Classifier Results

CATBoost classifier results: see `CATBoost_outputs` folder.
XGBoost classifier results: see `XGBoost_outputs` folder.

### Generated Novel Biofuel Molecules

The generated biofuel molecules can be viewed after running `MOLGAN.py.` The molecules will be output to the console and displayed as images.


## Contribution

We welcome any form of contribution. If you have any suggestions for improvement or find any bugs, please submit a Pull Request.

1. Fork this repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

· Thanks to DeepChem for providing the tools and libraries.

· Thanks to the teams behind XGBoost and CATBoost for their development work.

· Thanks to RDKit for providing cheminformatics tools.
