# AMP-Potency-Prediction-and-EvoGradient
A PyTorch implementation of "Automatic Identification and Virtual Directed Evolution of Antimicrobial Peptides with Explainable Deep Learning".


![overview](overview.jpg)



## Installation
The environments and their corresponding versions are specified in [environment.yaml](./environment.yaml).

The recommended method of installation is through [conda](https://github.com/conda/conda). 
To install, run the following command:

```
conda env create -f environment.yaml -n myenv
source activate myenv
```
The installation should be within 1 hour on normal desktop computers. 

**Our program does not require any non-standard hardware. We recommend running it on device with GPUs to speed up execution.*

## Usage

### AMP classification
To predict peptide sequences as AMPs or non-AMPs, run the following command:
```
python AMP_classification.py --testPath './data/classification/PathToFastaFile' --savePath 'outputFilePath'
```
Note: input file must be fasta file. 

One Example is (*cost approximately 11 seconds*):
```
python AMP_classification.py --testPath './data/classification/demo.fasta' --savePath 'output/classification_result.csv'
```
Expected output: [`./output/classification_result.csv`](./output/classification_result.csv).


### AMP regression
To predict antimicrobial activity of peptide sequences, run the following command:
```
python AMP_regression.py --testPath './data/regression/PathToFastaFile' --savePath 'outputFilePath'
```
Note: input file must be fasta file. 

One Example is (*cost approximately 7 seconds*):
```
python AMP_regression.py --testPath './data/regression/demo.fasta' --savePath 'output/regression_result.csv'
```
Expected output: [`./output/regression_result.csv`](./output/regression_result.csv).

### EvoGradient: Directed Evolution 
To perform the directed evolution to increase antimicrobial activity of peptides, run the following command:
```
cd EvoGradient
python EvoGradient.py --peptide PeptideToOptimize
```
One Example is (*cost approximately 9 seconds*):
```
cd EvoGradient
python EvoGradient.py --peptide RPLIKLRSTAGTGYTYVTRK
```
Expected output: [`./EvoGradient/EvoResult`](./EvoGradient/EvoResult/RPLIKLRSTAGTGYTYVTRK).


## License
[This repository as a whole is under the Apache-2.0 license.](./LICENSE)


