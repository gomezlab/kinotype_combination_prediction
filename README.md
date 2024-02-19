# Modeling Responses to Kinase Inhibitor Combinations using Kinome Inhibition States
This repository contains code for processing kinobeads data for cell line response prediction to combination therapies, as outlined in the paper: [https://pubmed.ncbi.nlm.nih.gov/38160286/](https://pubmed.ncbi.nlm.nih.gov/38160286/) 

## Repository Structure 
This repository is divided into three main folders:
* [`src`](src): source code for generating all results and figures
* [`data`](data): raw data used by the source code (included in zenodo: 10.5281/zenodo.10680994)
* [`results`](results): results generated by source code (included in zenodo: 10.5281/zenodo.10680994)
* [`figures`](figures): figures generated by source code 

## Data Organization
The folder [`src/data_organization`](src/data_organization) contains code to process kinome profiling data from the kinobeads assay, and link it to cell line responses from NCI ALMANAC and baseline gene expression data from CCLE. 

## Modeling 
The folder [`src/ALMANAC_klaeger_johnson_modelling`](src/ALMANAC_klaeger_johnson_modelling) contains code to build machine learning models using the combined dataset, predicting combination cell viability. This also includes code to process experimental data and validate model predictions. 

## Figures
All the figures published in the paper can be found in the folder [`figures`](figures) and some specific figures generated as part of model building code can be found in [`figures/ALMANAC_klaeger_johnson`](figures/ALMANAC_klaeger_johnson)



