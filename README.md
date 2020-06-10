# LDTC

## Introduction
We propose a novel temporal clustering solution: **L**ifelong **D**eep **T**emporal **C**lustering (**LDTC**), which has the following unique properties: 

 1. With the aid of the designed new autoencoder (Dilated Causal Convolutions + Attention LSTM) with a clustering layer, it effectively integrates both the dimensionality reduction and temporal clustering  into an  end-to-end unsupervised deep learning framework resulting in clustering data effectively and efficiently;
 2. On this basis, by a new hierarchical jointly optimizing of both data latent representation and clustering objective, the LDTC can achieve  high-quality clustering results comparing with the-state-of-the-art temporal clustering algorithms;
 3. By being equipped with the novel unsupervised lifelong learning mechanisms, the LDTC has gained the unique abilities to effectively deal with dynamic changes and learn new concepts/classes in the sequential tasks learning without the catastrophic forgetting and degradation of the model accuracy over its lifetime.


## Install & Running
###  Environment
This implementation was written for Python 3.x.

### Dependencies
```
tensorflow==1.12
keras==2.2.4
scikit-learn==0.22.2
numpy==1.18.2
tslearn==0.3.1
```

### Running
To train the LDTC model, run the main script `LL_train.py` with command-line arguments. Available arguments are explained with:
```
python LL_train.py --help
```

## Datasets
### Multivariate time series datasets
We used seven multivariate time series datasets from real-world collected by [titu1994](https://github.com/titu1994/MLSTM-FCN/releases) for evaluation, which are **EEG2**, **NetFlow**, **Wafer**, **HAR**, **AREM**, **Uwave**, and **ArabicDigits**, respectively. You can download the whole multivariate time series data set through [this link](https://github.com/titu1994/MLSTM-FCN/releases/tag/v1.0).

Only the **Uwave** dataset is provided in this repository, and the rest of the datasets need to be downloaded by yourself.

The hyperparameter settings that the program needs to supplement under different data sets as follows.

Hyper-parameters | EEG2| NetFlow| Wafer| HAR| AREM| Uwave| ArabicDigits
:-: | :-: | :-: | :-: | :-:| :-:| :-:| :-:
timesteps| 256 | 994 | 198 | 128|  480|  315|  93|
input_dim| 64| 4 | 6 | 9| 7| 3| 13|
pool_size| 16| 71 | 11| 8| 24| 15| 3|
batch_size| 32| 32| 32| 64| 2| 64| 32|

### Preparation of lifelong clustering dataset
In order to simulate the process of lifelong clustering, this repository provides a script `.\data_util\datasets.py` that splits the original dataset to help users create a lifelong learning training dataset.

For example:
```
python .\data_util\datasets.py --data_path=='.\data\Uwave' --save_path=='.\data\Uwave\ll' --cluster_nums='3,4,5'
```

[cluster_nums]: The number of clusters contained in each piece of data, separated by ';'.

