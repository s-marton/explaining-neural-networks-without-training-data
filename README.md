# Explaining Neural Networks without Access to Training Data

Official implementation of the paper "Explaining Neural Networks without Access to Training Data" by Sascha Marton, Stefan Lüdtke Christian Bartelt, Andrej Tschalzev and Heiner Stuckenschmidt.

This repository contains the code to replicate the experiments from the paper. The used packages and corresponding versions can be found in the requirements.txt.

Replicating the requirements comprises three steps:
  1. Generate Datasets --> Execute 01_generate_data.ipynb for the corresponding number of variables and the predefined setting (We have created one file for each number of variables considered in the paper). To reproduce the results of the paper, you need to run all notebooks starting with "01"
  2. Train Networks Lambda --> Execute 02_lambda_net.ipynb for the corresponding number of variables and the predefined setting (We have created one file for each number of variables considered in the paper). To reproduce the results of the paper, you need to run all notebooks starting with "02"
  3. Train I-Net --> The Notebooks starting with "03" are used to train an I-Net and display all results. For the Case study, execute the files starting with "03_interpretation_net-n23" for each function family. To generate the results for the performance comparison on the real world datasets, please run the files starting "05_evaluate" for the corresponding function family. The visual comparison can be performed by running the notebooks starting with "03_interpretation_net-n02".

For the ablation study, the data generation is more elaborate. An expemplary file for the data generation is given in the corresponding notebooks "01b" and "02b". To reproduce the experiments, you need to run those notebooks for each number of variables by changing "number_of_variables" in the config at the top of the notebook to the corresponding value. This has to be performed for each function family (an expemplary notebook is given for each function family). Afterward, you can simply run the python scripty ending with "BENCHMARK" for the corresponding function family to reproduce the results of the ablation study.

The function families are identified in th code as follows:
    - vanilla --> standard decision tree
    - SDT-1 --> SDT

The network parameters used for training the I-Net have to be generated manually, since they are too large to upload. However, I can provide you with them upon request.

## Datasets

The datasets are located in the folder "real_world_datasets". We used standard preprocessing for all datasets which includes the following steps:

1. Remove all features comprising identifier features (e.g., IDs, Names).
2. Impute missing values: For numeric values, we used the mean for imputation, and for ordinal, categorical, and nominal features, we used the mode.
3. Transform ordinal features to numeric values.
4. One-hot-encode categorical and nominal features.
5. Scale features in [0, 1] using min-max normalization.
6. Split data into distinct train (85%), valid (5%), and test set (10%).
7. Rebalance train data if the number of samples of the minority class is less than 25%.

| **Dataset**               | **Number of Features Preprocessed (Raw)** | **Number of Samples (True/False)** | **Citation**                                                                                            | **Source**                                                                                                           | **Network Performance** |
|---------------------------|--------------------------------------------|------------------------------------|---------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|------------------------|
| **Titanic**               | 9 (12)                                     | 891 (342/549)                     | -                                                                                                       | [https://www.kaggle.com/c/titanic/data](https://www.kaggle.com/c/titanic/data)                                       | 83.15                  |
| **Medical Insurance**     | 9 (7)                                      | 1338 (626/712)                    | [dataset_medical_insurance](citet{dataset_medical_insurance})                                           | [https://www.kaggle.com/datasets/mirichoi0218/insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance) | 95.49                  |
| **Brest Cancer Wisconsin** | 9 (10)                                     | 699 (241/458)                      | [UCI_ml_repository](citet{UCI_ml_repository}), [dataset_breast_cancer_original](citet{dataset_breast_cancer_original}) | [https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+\%28Original\%29](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+\%28Original\%29) | 97.10                  |
| **Wisconsin Diagnostic**   | **Breast Cancer**                          | 10 (10)                            | 569 (212/357)                                                                                           | [UCI_ml_repository](citet{UCI_ml_repository})                                                                         | 98.21                  |
| **Heart Disease**         | 13 (65)                                    | 303 (164/139)                     | [UCI_ml_repository](citet{UCI_ml_repository}), [dataset_heart_disease](citet{dataset_heart_disease})       | [https://archive.ics.uci.edu/ml/datasets/heart+disease](https://archive.ics.uci.edu/ml/datasets/heart+disease)     | 93.33                  |
| **Cervical Cancer**       | 15 (36)                                    | 858 (55/803)                      | [UCI_ml_repository](citet{UCI_ml_repository}), [dataset_cervical_cancer](citet{dataset_cervical_cancer})   | [https://archive.ics.uci.edu/ml/datasets/Cervical+cancer+\%28Risk+Factors\%29](https://archive.ics.uci.edu/ml/datasets/Cervical+cancer+\%28Risk+Factors\%29) | 84.71                  |
| **Loan House**            | 16 (12)                                    | 614 (422/192)                     | -                                                                                                       | [https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/](https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/) | 77.05                  |
| **Credit Card Default**   | 23 (23)                                    | 30000 (23364/6636)                | [UCI_ml_repository](citet{UCI_ml_repository}), [dataset_credit_card](citet{dataset_credit_card})         | [https://archive.ics.uci.edu/ml/datasets/default+of+credit+card


## Hyperparameters

The hyperparameters for the $\mathcal{I}$-Net (Table~\ref{tab:i-net-params}) were tuned using a greedy neural architecture search with autokeras, followed by a manual fine-tuning of the selected values. To measure the performance during the optimization, we used the validation loss on a distinct validation set $\Theta_\lambda$ comprising $1000$ network parameters.

**$\mathcal{I}$-Net Training Parameters.**

| **Parameter**                 | **Value**          |
|------------------------------|--------------------|
| **DT**                       |                    |
| Hidden Layer Neurons         | [1792, 512, 512]   |
| Hidden Layer Activation      | Sigmoid            |
| Dropout                      | [0, 0, 0.5]        |
| **Univariate SDT**           |                    |
| Hidden Layer Neurons         | [4096, 2048]       |
| Hidden Layer Activation      | Swish\*            |
| Dropout                      | [0, 0.5]           |
| **Standard SDT**             |                    |
| Hidden Layer Neurons         | [1792, 512, 512]   |
| Hidden Layer Activation      | Swish\*            |
| Dropout                      | [0.3, 0.3, 0.3]    |
| **Batch Size**               | 256                |
| **Optimizer**                | Adam               |
| **Learning Rate**            | 0.001              |
| **Loss Function**            | $\mathcal{L}_{\mathcal{I}\text{-Net}}$ |
| **Training Epochs**          | 500                |
| **Early Stopping**           | Yes                |
| **Number of Training Samples** | 9,000              |
\*The Swish activation function proposed by [Swish Activation](https://arxiv.org/abs/1710.05941) is defined by $swish(x) = x \times sigmoid(x)$ and is claimed to consistently match or outperform a ReLU activation.



**$\lambda$-Net Training Parameters.**

| **Parameter**              | **Value**    |
|---------------------------|--------------|
| Hidden Layer Neurons      | [128]        |
| Hidden Layer Activation   | ReLU         |
| Dropout                   | No           |
| Batch Size                | 64           |
| Optimizer                 | Adam         |
| Learning Rate             | 0.001        |
| Loss Function             | binary_crossentropy |
| Training Epochs           | 1,000        |
| Early Stopping            | Yes          |
| Number of Training Samples| 5,000        |



**Standard DT Training Parameters.**

| **Parameter**         | **Value** |
|-----------------------|-----------|
| max_depth             | 3         |
| criterion             | gini      |
| min_samples_split     | 2         |
| min_samples_leaf      | 1         |



**SDT Training Parameters.**

| **Parameter**                | **Value** |
|------------------------------|-----------|
| depth                        | 3         |
| learning_rate                | 0.01      |
| criterion                    | binary_crossentropy |
| lambda                       | 0.001     |
| beta                         | 1         |
| weight_decay                 | 0.0005    |
| maximum_path_probability     | True      |


