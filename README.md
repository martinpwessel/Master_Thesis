# "Improving Media Bias Detection through the usage of State-of-the-Art Transformers"

This is the repository for my Master Thesis presented at the University of Konstanz. 

## Abstract

"The thesis introduces MBIB, the Media Bias Identification Benchmark. MBIB, inspired
by GLUE [Wang et al., 2019b], consists of eight unified media bias tasks and
associated datasets that allow for comprehensive performance analysis and comparison
of models aimed at detecting media bias. To find suitable datasets for each MBIB
task an extensive overview of existing relevant datasets is created. Furthermore, a
framework is developed to test and compare models on MBIB in a unified way. The
framework is then utilized to compare transformer models on the benchmark and
set baseline performances. With MBIB this thesis introduces an in-depth and challenging
task collection for creating comprehensive media bias-detecting methods.
Additionally, it shows that the transformer model choice matters less for performance
than originally presumed. Finally, it can serve as an inventory of existing
datasets and offers insight into gaps in the existing media bias research."


## Description

The code is structured as follows:

ProxyTask - contains the training and testing Code for the Proxy Task as well as the results for every model and fold  
Experiment - contains the training and testing Code for the Experiment:  
    Batch_size.ipynb - Script that analysis the optimal batch size  
    Text_length.ipynb - Calls MBDataLoader and analyzes the text length for one task  
    MBDataLoader.py - Script to combine datasets for each task, prepocesses and balances the data  
    MBWrapper.py - Wrapper for one training and testing, loads data and initializes stratified k-fold  
    MBTraining.py - Main training script for one fold  
    ModelSpecifications.py - sets the main model parameters  
    Scripts - Contains the scripts run on the cluster  
Preprocessing - Contains preprocessing scripts for every dataset  
Results - Results of the experiment for every fold   
Analysis - Contains scripts for the visualizations in the thesis as well as summarized results  



