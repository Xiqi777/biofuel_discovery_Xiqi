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

## Installation

Please ensure you have the following dependencies installed:

```bash
pip install -r requirements.txt
```

## Usage

Here are the detailed instructions on how to use this project:

### 1. Obtain fingerprints and Create classifiers

Use either of the following scripts to create classifiers:

· Using CATBoost: `CATboost_classifier_biofuel.py`

```bash
python CATboost_classifier_biofuel.py
```

· Using XGBoost: `classifier_Xgboost_fingerprint_biofuel.py`

```bash
python classifier_Xgboost_fingerprint_biofuel.py
```

Input the name of `new datasets.csv` into the `#main function` section of the above scripts to obtain the classifiers. Running the `classifier_Xgboost_fingerprint_biofuel.py` script also can generate the fingerprints of the molecules in `new datasets.csv`.

### 2. Generate novel biofuel molecules

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
new datasets.csv               # The SMILES and Type of biofuels and non-biofuels molecules
classifier_Xgboost_fingerprint_biofuel.py # Code for using XGBoost to create a classifier
CATboost_classifier_biofuel.py # Code for using CATBoost to create a classifier
XGBoost_outputs/               # Results obtained by running the latest and currently used datasets with XGBoost
CATBoost_outputs/              # Results obtained by running the latest and currently used datasets with CATBoost
biomass.csv                    # A large dataset of molecule SMILES derived from biomass
biofuels.csv                   # Used to run MOLGAN on Myriad to generate additional potential biofuel molecules
MOLGAN.py                      # Code for using MOLGAN to create potential biofuel molecules
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
