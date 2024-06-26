# Design of Novel Biofuels Using Machine Learning GANs

This project aims to design novel biofuel molecules using machine learning and Generative Adversarial Networks (GANs). The main process of the project includes data collection, classifier development, fuel classification, and generation of novel biofuel molecules.

## Table of Contents

- [Background](#background)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [File Structure](#file-structure)
- [Contribution](#contribution)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Background

1. **Data Collection**
   - Collect data on biofuel and non-biofuel molecules to build a small dataset containing SMILES strings of biofuels and non-biofuels. The dataset is stored in `new datasets.csv`.
   - File used: `new datasets.csv`

2. **Fingerprint and Properties Extraction**
   - Use `dataset run.py` to obtain the fingerprints and properties of the molecules from the collected data. This script processes the `new datasets.csv` and produces `datasets_with_properties.csv`.
   - File used: `dataset run.py`
   - Input: `new datasets.csv`
   - Output: `datasets_with_properties.csv`

3. **Classifier Development**
   - Use XGBoost and CATBoost to run the scripts `classifier_Xgboost_fingerprint_biofuel.py` and `CATboost_classifier_biofuel.py` to create classifiers that can differentiate between biofuels and non-biofuels.
   - Files used: `classifier_Xgboost_fingerprint_biofuel.py`, `CATboost_classifier_biofuel.py`
   - Input: `datasets_with_properties.csv`
   - Output: `XGBoost_outputs/`, `CATBoost_outputs/`

4. **Classification of Large Biomass Dataset**
   - Use the established classifiers to classify a large biomass dataset stored in `biomass.csv`. This step helps in extracting usable biofuel molecules' SMILES.
   - Files used: `classifier_Xgboost_fingerprint_biofuel.py`, `CATboost_classifier_biofuel.py`
   - Input: `biomass.csv`
   - Output: SMILES of biofuel molecules extracted from `biomass.csv`

5. **Creating a Biofuel Dataset**
   - Combine the biofuel SMILES from `new datasets.csv` and the SMILES extracted in step 4 to create a biofuel database named `biofuels.csv`, which will be provided to MOLGAN.
   - Files used: Data from previous steps
   - Input: The SMILES of biofuel molecules from step 1 and step 4
   - Output: `biofuels.csv`

6. **Generation of Novel Biofuel Molecules**
   - Run `MOLGAN.py` on Myriad to generate more potential novel biofuel molecules using the biofuel database `biofuels.csv`.
   - File used: `MOLGAN.py`
   - Input: `biofuels.csv`, `MOLGAN.py`
   - Output: Generated biofuel molecules

7. **Integration and Iteration**
   - Integrate the generated biofuel molecules from step 6 into the small dataset `datasets_with_properties.csv`. Re-establish classifiers and repeat steps 3, 4, 5, and 6 to form a closed loop between MOLGAN and classifier development.

## Installation

Please ensure you have the following dependencies installed:

```bash
pip install -r requirements.txt
```

## Usage

Here are the detailed instructions on how to use this project:

### 1. Obtain fingerprints and properties of fuel molecules

Run `dataset run.py` to obtain the fingerprints and properties of fuel molecules.

```bash
python dataset\ run.py
```

### 2. Create classifiers

Use either of the following scripts to create classifiers:

· Using CATBoost: `CATboost_classifier_biofuel.py`

```bash
python CATboost_classifier_biofuel.py
```

· Using XGBoost: `classifier_Xgboost_fingerprint_biofuel.py`

```bash
python classifier_Xgboost_fingerprint_biofuel.py
```

### 3. Generate novel biofuel molecules

Run `MOLGAN.py` on Myriad to generate potential novel biofuel molecules.

```bash
python MOLGAN.py
```

## Examples

### Classifier Results

CATBoost results: see `CATBoost_outputs` folder.
XGBoost results: see `XGBoost_outputs` folder.

### Generated Novel Biofuel Molecules

The generated biofuel molecules can be viewed after running `MOLGAN.py.` The molecules will be output to the console and displayed as images.

## File Structure

```bash
CATBoost_outputs/              # Results obtained by running latest and currently used datasets with CATBoost
XGBoost_outputs/               # Results obtained by running latest and currently used datasets with XGBoost
CATboost_classifier_biofuel.py # Code for using CATBoost to create a classifier
classifier_Xgboost_fingerprint_biofuel.py # Code for using XGBoost to create a classifier
MOLGAN.py                      # Code for using MOLGAN to create potential biofuel molecules
dataset run.py                 # Code for obtaining fingerprints and properties of fuel molecules
datasets_with_properties.csv   # Fingerprints and properties of biofuel and non-biofuel molecules
new datasets.csv               # Name/SMILES/Type of biofuel and non-biofuel molecules
biomass.csv                    # A large dataset of molecule SMILES derived from biomass
biofuels.csv                   # Used to run MOLGAN on Myriad to generate additional potential biofuel molecules
```

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
