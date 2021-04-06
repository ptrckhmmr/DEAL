# DEAL: Deep Evidential Active Learning for Image Classification

This repository implements the Active Learning (AL) approach Deep Evidential Active Learning (DEAL) together with the following benchmark approaches:

* Minimal Margin sampling with softmax
* k-Center-Greedy for Core-Set by Sener and Savarese (2017) (https://arxiv.org/abs/1708.00489)
* MC-Dropout with Variation Ratio by Gal et al. (2017) (https://arxiv.org/pdf/1703.02910.pdf)
* Deep Ensembles with Variation Ratio by Beluch et al. (2018) (http://openaccess.thecvf.com/content_cvpr_2018/html/Beluch_The_Power_of_CVPR_2018_paper.html)
* Uniform Sampling (baseline)

Two different network architectures are implemented with the regular softmax 
output and the DEAL modifications. The folder [`01_LeNet/`](01_LeNet/) implements the AL framework with the LeNet architecture, and the folder
[`02_ResNet/`](02_ResNet/) implements the ResNet architecture. 

Within each of the respective architecture folders, the folder `sampling_methods` contains all implemented acquisition functions. 
They must be defined in `sampling_methods/constants.py`. Implemented methods are for instance:
 * Entropy for DEAL
 * k-Center-Greedy 
 * Minimal Margin for DEAL and softmax 
 * Uncertainty derived directly by DEAL
 * Varation Ratio for DEAL and softmax.


The main scripts to execute the experiments are:

* `run_experiment<placeholder>.py`: 
Placeholder has to be replaced by LeNet, LeNet_modelEnsemble, ResNet, ResNet_modelEnsemble.
When using the ResNet architecture, one has to add the identifier `--dataset` followed by the respective data set to be used in the terminal. 
When using the LeNet architecture, one has to adapt the data set to be used in the `__init__` method in the respective files [`lenet_evidence.py`](01_LeNet/utils/lenet_evidence.py) and [`lenet_softmax.py`](01_LeNet/utils/lenet_softmax.py)
Additionally, the file `run_experiment<placeholder>.py` includes several flags to specify the run options:

    * `dataset`: Defines the name of the data set. It must match the name 
    used in `utils/create_data.py` and has to be downloaded before.
    Available data sets are MNIST (identifier: mnist_keras), CIFAR-10 (identifier: cifar10_keras), and SVHN (identifier: svhn). 
    Additionally, the AL framework is evaluated on a real-world use case on a pediatric pneumonia chest 
    X-ray image data set (identifier: medical) published by Kermany et al. (2018) (https://www.sciencedirect.com/science/article/pii/S0092867418301545).

    * `sampling_method`: Specifies the active learning method. 
    Must be defined in `sampling_methods/constants.py`.
    Possible options are: uniform, margin, kcenter, varRatio, varRatio_ensemble,
    entropyEDL, uncertaintyEDL, varRatioEDL, marginEDL.

    * `warmstart_size`: Initial batch of uniformly sampled examples to use as seed
    data. Float indicates percentage of total training data and integer
    indicates raw size.

    * `batch_size`: Number of datapoints to request in each batch. Float indicates
    percentage of total training data and integer indicates raw size.
    
    * `trials`: Specifies the total number of experiments that are conducted.
    
    * `seed`: Defines the seed to use for random state.

    * `score_method`: Model used to evaluate the performance of the sampling
    method. Must be in `get_model` method of `utils/utils.py`. Possible options are: LeNet_Softmax, LeNet_DEAL, 
    ResNet_Softmax, ResNet_DEAL. 
    When using the ResNet architecture, specify in the file `utils/utils.py` the ResNet hyperparameters and specific ResNet architecture.
    
    * `select method`: Model used to query next batch of data instances. 
    Must be in `get_model` method of `utils/utils.py`. Possible options are: LeNet_Softmax, LeNet_DEAL, 
    ResNet_Softmax, ResNet_DEAL.
    When using the ResNet architecture, specify in the file `utils/` the ResNet hyperparameters and specific ResNet architecture.
    
    * `save_dir`: Directory to save results.

    * `data_dir`: Directory with saved data sets.
    
    * `max_dataset_size`: The maximum number of datapoints to include in data. "0" indicates no limit.
    
    * `train_horizon`: How far to extend the learning curve as a percent of training data instances. 

    * There are further flags that can be specified. 


* `utils/create_data.py`: MNIST, CIFAR-10, and SVHN data sets can be downloaded and used with the implemented approaches. The pediatric pneumonia data set can be downloaded from kaggle.com. 
    Specify data set to be downloaded in the flag `--dataset`. When using the ResNet architecture, one has to add the identifier `--dataset` followed by the respective data set to be used in the terminal. 
    
    
Dependencies can be found in [`requirements.txt`](requirements.txt).


## Results

The accuracy per active learning round is stored as json-file in the folder `test_accuracy/`.

