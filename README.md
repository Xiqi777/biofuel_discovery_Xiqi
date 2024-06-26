# Design of Novel Biofuels Using Machine Learning GANs

This project aims to design novel biofuel molecules using machine learning and Generative Adversarial Networks (GANs). The main process of the project includes data collection, classifier development, fuel classification, and generation of novel biofuel molecules.

## Table of Contents

- [Background](#background)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Background

1. Collect data on biofuel and non-biofuel molecules to build a small database.
2. Use XGBoost and CATBoost to run code and create classifiers to compare the two types of fuels.
3. Use the established classifiers to classify a large fuel database and extract usable biofuels.
4. Organize the biofuels obtained in steps 1 and 3 and provide them to MOLGAN.
5. Run MOLGAN on Myriad to obtain more potential novel biofuel molecules.

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

## Generated Novel Biofuel Molecules

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
