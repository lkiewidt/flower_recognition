# Image classifier: flowers


## Introduction
The image classifier project is part of Udacity's Data Scientist Nanodegree. The goal is to build and train a neural network that identifies 102 different flower species.

In the project, training, validation, and test images are load and transformed. Subsequently, a pre-trained neural network (e.g. VGG or DenseNet) is combined with a fully-connected classifier. Afterwards, a training-validation-testing pipeline for the classifier is implemented.

## Python Libraries
The following Python libraries are key in the project:
* numpy
* torch
* torchvision
* matplotlib / seaborn


## Key Findings
* the pre-trained networks (e.g. VGG) provide excellent feature selectors for image recognition
* the training of large neural networks is sifnificantly speed up by GPU computing
* the implemented network reached an accuracy of 70% in 5 epochs

