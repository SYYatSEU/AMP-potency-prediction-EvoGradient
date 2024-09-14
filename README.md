# AMP-Potency-Prediction-and-EvoGradient
A PyTorch implementation of "Automatic Identification and Virtual Directed Evolution of Antimicrobial Peptides with Explainable Deep Learning".


![overview](overview.jpg)

## Installation
Environments are listed in [environment.yaml](./environment.yaml). 

The recommended method of installation is through [conda](https://github.com/conda/conda). 
To install, run the following command:

```
conda env create -f environment.yaml -n myenv
source activate myenv
```

## Usage

### AMP classification
To predict peptide sequences as AMPs or non-AMPs, run the following command:
```
python AMP_classification.py --testPath './data/classification/PathToFastaFile' --savePath 'outputFilePath'
```
Note: input file must be fasta file. 

One Example is:
```
python AMP_classification.py --testPath './data/classification/demo.fasta' --savePath 'output/classification_result.csv'
```



### AMP regression
To predict antimicrobial activity of peptide sequences, run the following command:
```
python AMP_regression.py --testPath './data/regression/PathToFastaFile' --savePath 'outputFilePath'
```
Note: input file must be fasta file. 

One Example is:
```
python AMP_regression.py --testPath './data/regression/demo.fasta' --savePath 'output/regression_result.csv'
```


### EvoGradient: Directed Evolution 
To perform the directed evolution to increase antimicrobial activity of peptides, run the following command:
```
cd EvoGradient
python EvoGradient.py --peptide PeptideToOptimize
```
One Example is:
```
cd EvoGradient
python EvoGradient.py --peptide RPLIKLRSTAGTGYTYVTRK
```



## License
This repository as a whole is under the Apache-2.0 license.


