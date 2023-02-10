[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# Black-box attack-based security evaluation framework for credit card fraud detection models
This archive is distributed in association with the [INFORMS Journal on
Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).

## Cite

To cite this software, please cite the paper using its DOI and the software itself, using the following DOI.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8475.svg)](https://doi.org/10.5281/zenodo.8475)
Below is the BibTex for citing this version of the code.

```
@article{Xiao2023Black,
  author =        {J. Xiao, Y.H. Tian, Y.L. Jia, X.Y. Jiang, L.A. Yu, and S.Y. Wang},
  publisher =     {INFORMS Journal on Computing},
  title =         {Black-box attack-based security evaluation framework for credit card fraud detection models, v2021.0076},
  year =          {2023},
  doi =           {10.5281/zenodo.8475},
  url =           {https://github.com/INFORMSJoC/2021.0076},
}  
```


## Description
The security of credit card fraud detection (CCFD) models based on machine learning is important but rarely considered in the existing research. To this end, we propose a black-box attack-based security evaluation framework for CCFD models. Under this framework, the semi-supervised learning technique and transfer-based black-box attack are combined to construct two versions of semi-supervised transfer black-box attack (STBA) algorithm. Moreover, we introduce a new nonlinear optimization model to generate the adversarial examples against CCFD models and a security evaluation index to quantitatively evaluate the security of them. 

This project contains four folders: `data`, `results`, `src`, `scripts`. 
- `data`：include two datasets used in the paper and a toy dataset for debugging.
- `results`: include the experimental results.  
- `src`: include the source code. 
- `scripts`: include two scripts for evaluating the security of machine learning models based on substitute models LR and SVM.  

## Setup
### Install dependencies
- To run the code, you will need to make sure that you have already installed Anaconda3.  
- You also need to install the packages listed in requirements.txt.
```bash
pip install -r requirements.txt
```

### Preparing the datasets
- The cleaning process of the Lending club credit dataset is described in subsection 3.1 of the paper, and the cleaned dataset can be obtained by extracting the zip file in the data folder.  

- Download Vesta credit dataset from https://www.kaggle.com/competitions/ieee-fraud-detection/data, and run `feature_engineering.py` to perform feature engineering.  

- A toy dataset is also provided for quick start.





## Replicating
First, put the processed dataset into `src` folder;  

Second, run `main.py` to get the accuracy of the target models under different attack strengths and save it. We provide a variety of semi-supervised methods that can be used directly, including self-training, Co-Forest, semi-supervised GAN. The FlexMatch implementation uses the project https://github.com/TorchSSL/TorchSSL.  

Finally, run `calculateSEI.py` to calculate the SEI index.
