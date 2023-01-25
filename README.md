[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# Black-box attack-based security evaluation framework for credit card fraud detection models
This archive is distributed in association with the [INFORMS Journal on
Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).

## Setup
### Install dependencies
To run the code, you will need to make sure that you have already installed Anaconda3.  
You also need to install the packages in requirements.txt.
```bash
pip install -r requirements.txt
```

### Preparing the datasets
The cleaning process of the Lending club credit dataset is described in subsection 3.1 of the paper, and the cleaned dataset can be obtained by extracting the zip file in the data folder.  

Download Vesta credit dataset from https://www.kaggle.com/competitions/ieee-fraud-detection/data, and run feature_engineering.py to perform feature engineering.  

A toy dataset is also provided for quick start.


## Description
The security of credit card fraud detection (CCFD) models based on machine learning is important but rarely considered in the existing research. To this end, we propose a black-box attack-based security evaluation framework for CCFD models. Under this framework, the semi-supervised learning technique and transfer-based black-box attack are combined to construct two versions of semi-supervised transfer black-box attack (STBA) algorithm. Moreover, we introduce a new nonlinear optimization model to generate the adversarial examples against CCFD models and a security evaluation index to quantitatively evaluate the security of them. 

## Replicating
First, put the processed dataset into src folder;  

Second, run main.py to get the accuracy of the target models under different attack strengths and save it. We provide a variety of semi-supervised methods that can be used directly, including self-training, Co-Forest, semi-supervised GAN. The FlexMatch implementation uses the project https://github.com/TorchSSL/TorchSSL.  

Finally, run calculateSEI.py to calculate the SEI index.
