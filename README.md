Trustworthy Machine Learning for Health Care: Scalable Data Valuation with the Shapley Value
==================================================================================================================================================
This repository contains the code used for the paper "Trustworthy Machine Learning for Health Care: Scalable Data Valuation with the Shapley Value". The code was tested on a workstation running with Python 3.8 and PyTorch 1.6.0. For more details and references, please see the paper (open access).

If you use this repository and/or the corresponding paper, please cite it as follows:

```
@inproceedings{pandl2021trustworthy,
  title={Trustworthy Machine Learning for Health Care: Scalable Data Valuation with the Shapley Value},
  author={Pandl, Konstantin D. and Feiland, Fabian and Thiebes, Scott and Sunyaev, Ali},
  booktitle={Proceedings of the ACM conference on health, inference, and learning},
  year={2021}
}
```

Project Organization
--------------------------------------------------------------------------------------------------------------------------------------------------

General code to load and preprocess data
```
General section
    ├── data                    
    |    ├── CheXpert-v1.0-small   		<- dataset used (not checked in as the size is ~10GB; needs to be downloaded from https://stanfordmlgroup.github.io/competitions/chexpert/)
    |    ├── datasets              		<- CSV files containing the split used for training, validation and testing
    ├── arrays							
    |    ├── deep_features			<- stores the deep features created with get_deep.py
    |    ├── raw_data				<- stores the raw data created with get_array.py
    ├── models							
    |    ├── densenet.py			<- src defining Densenet-121 for classification task
    |    ├── resnet.py				<- src defining ResNet-34 for classification task
    ├── config.json				<- config file stores path to data
    ├── import_utils.py				<- src for importing pictures and labels from dataset
    ├── get_array.py				<- src for loading and storing raw data in arrays
    ├── get_deep.py				<- src for training, obtaining and storing deep features in arrays
```

Code to conduct various application experiments
```
Applications
    ├── applications
    |    ├── runtime comparison
    |           ├── DShap.py			<- src for running different data valuation methods
    |           ├── shap_utils.py		<- src for utils of DShap.py
    |           ├── runtime.py			<- src for runtime experiment
    |           ├── fit_module.py		<- src for training and evaluation of ML model

    |    ├── point_removal
    |           ├── knn_shap_calculation.py	<- src for calculating KNN-Shapley values
    |           ├── plot_densenet.py		<- src for training and evaluating model after points are removed
    |           ├── point_removal.py		<- src for point removal experiment
    |           ├── utils.py			<- src for utils of knnn_shap_calculation.py and plot_densenet.py
    
    |    ├── noisy_label					
    |           ├── flip_arrays					
    |                  ├── deep_features	<- stores the new deep features for flipped data created with get_new_deep.py
    |                  ├── raw_data		<- stores the flipped raw data created with generate_data.py
    |           ├── generate_flip_data.py	<- src for generating and storing flipped data in arrays
    |           ├── get_deep_flip.py		<- src for training, obtaining and storing deep features of flipped data in arrays
    |           ├── knn_shap_calculation_flip.py<- src for calculating KNN-Shapley values for flipped data
    |           ├── label_detection.py		<- src for detecting noisy label experiment
    |           ├── utils.py			<- src for utils of knnn_shap_calculation_flip.py and generate_flip_data.py
```

Code to plot the results
```
Plotting results
    ├── plot_results						
    |    ├── model
    |           ├── loss_plot.py		<- src for plotting Figure 2
    |           ├── auroc_plot.py		<- src for plotting Figure 3
    |           ├── pred.npy			<- stores predicted labels of trained model on validation set
    |           ├── true.npy			<- stores true labels of validation set

    |    ├── runtime_plot.py			<- src for plotting Figure 4
    
    |    ├── point_removal
    |           ├── point_removal_plot.py	<- src for plotting Figure 6
    |           ├── val_result_HtoL.npz		<- stores results from applications/point_removal/point_removal.py
    
    |    ├── label_detection					
    |           ├── label_plot.py		<- src for plotting Figure 7				
    |           ├── f_knn.pkl			<- stores information about fraction of incorrect labels detected
    |           ├── f_random.pkl		<- stores information about fraction of incorrect labels detected
    |           ├── x_knn.pkl			<- stores information about fraction of data inspected
    |           ├── x_random.pkl		<- stores information about fraction of data inspected
```

--------------------------------------------------------------------------------------------------------------------------------------------------