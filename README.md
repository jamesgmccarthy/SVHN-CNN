# SVHN-CNN
## Introduction
Keras implementation of the [Multi-digit Number Recognition](https://arxiv.org/abs/1312.6082) model proposed by Ian Goodfellow et al using the popular SVHN dataset. 
The dataset comes in three parts, the training, testing and extra datasets. The training dataset is small enough to load into memory when training the model, but the extra dataset is too large. To solve this, there are two models in this project. svhn.py uses only the training dataset and simply loads it into memory during the training, whereas svhn_gen.py uses a data generator to load batches of data into memory and train the model on these batches. This process is a lot slower than just loading the whole dataset into memory during the training phase, I am currently attempting to make an implementation in tensorlow 2 that will solve this problem.

## Getting Started
To run this code, first the python libraries used must be installed. To do this simply run the following line in a terminal, it may be best to set up a virtual environment first to avoid any clashes with current package verisons
```
pip install -r requirements.txt
```
After that, in the terminal, navigate to the folder where this repository's files are saved to and run the following line in the terminal to run the model that trains on only the training data set.
```
python svhn.py
```
Or to run the model that trains on the training and extra dataset, using a data generator, run the following line in the terminal.
```
python svhn_gen.py
```
