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

Within each of the respective architecture folders, a folder [`utils/`](utils/) contains a file [`sampling_methods`](sampling_methods/). Here, all used acquisition functions are implemented. 
They must be defined in [`sampling_methods/constants.py`](sampling_methods/constants.py):
 * Entropy for DEAL
 * k-Center-Greedy 
 * Minimal Margin for DEAL and softmax 
 * Uncertainty derived directly by DEAL
 * Varation Ratio for DEAL and softmax.


The main scripts to execute the experiments are:

* [`run_experiment<placeholder>.py`](run_experiment<placeholder>.py): 
placeholder has to be replaced by LeNet, LeNet_modelEnsemble, ResNet, ResNet_modelEnsemble.
When using the ResNet architecture, in the terminal one has to enter the identifier --dataset with the respective data set to be used.
Additionally, the file run_experiment<placeholder>.py includes several flags to specify the run options:

    * `dataset`: Defines the name of the data set. It must match the name 
    used in [`utils/create_data.py`](utils/create_data.py) and has to be downloaded before.
    Available data sets are MNIST (identifier: mnist_keras) and CIFAR-10 (identifier: cifar10_keras). 
    Additionally, the AL fameworks are evaluated on a real-world use case on a pediatric pneumonia chest 
    X-ray image dataset (identifier: medical) published by Kermany et al. (2018) (https://www.sciencedirect.com/science/article/pii/S0092867418301545).

    * `sampling_method`: Specifies the active learning method. 
    Must be defined in [`sampling_methods/constants.py`](sampling_methods/constants.py).
    Possible options are: uniform, margin, kcenter, varRatio, varRatio_ensemble,
    entropyEDL, uncertaintyEDL, varRatioEDL, marginEDL.

    * `warmstart_size`: Initial batch of uniformly sampled examples to use as seed
    data. Float indicates percentage of total training data and integer
    indicates raw size.

    * `batch_size`: Number of datapoints to request in each batch. Float indicates
    percentage of total training data and integer indicates raw size.
    
    * `trials`: Specifies the total number of experiments that are conducted.
    
    * `seed`: Defines the seed to use for random state.

    *   `score_method`: Model used to evaluate the performance of the sampling
    method. Must be in `get_model` method of
    [`utils/utils.py`](utils/utils.py). Possible options are: LeNet_Softmax, LeNet_DEAL, 
    ResNet_Softmax, ResNet_DEAL. 
    When using the ResNet architecture, specify in the file [`utils/`](utils/) the ResNet hyperparameters and specific ResNet architecture.
    
    * `select method`: Model used to query next batch of data instances. 
    Must be in `get_model` method of [`utils/utils.py`](utils/utils.py). Possible options are: LeNet_Softmax, LeNet_DEAL, 
    ResNet_Softmax, ResNet_DEAL.
    When using the ResNet architecture, specify in the file [`utils/`](utils/) the ResNet hyperparameters and specific ResNet architecture.
    
    * `save_dir`: Directory to save results.

    * `data_dir`: Directory with saved data sets.
    
    * `max_dataset_size`: The maximum number of datapoints to include in 
    data. "0" indicates no limit.
    
    * `train_horizon`: How far to extend the learning curve as a percent of training data instances. 

    * There are further flags that can be specified. 


* [`utils/create_data.py`](utils/create_data.py): MNIST and CIFAR-10 data sets can be downloaded and used with the implemented approaches. Specify data set to be downloaded in the flag `--dataset`
    * The data sets will be saved to [`utils/tmp/data/`](utils/tmp/data/) by default. The directory 
    can be defined with the `--save_dir` flag.
    * The pediatric pneumonia data set can be downloaded on kaggle.com 
    
    
Dependencies can be found in [`requirements.txt`](requirements.txt).


## Results

The accuracy per active learning round is stored as json-file in the folder
[`test_accuracy/`](test_accuracy).

