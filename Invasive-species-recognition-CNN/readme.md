# MLND-Capstone

Capstone project for Udacity's Machine Learning Engineer Nanodegree.

## Problem Statement

This work is to use deep learning convolutional network to [distinguish invasive species]((https://www.kaggle.com/c/invasive-species-monitoring)) in the forest (hosted by Kaggle Competition). The invasive species can put ecosystems at risk. To track and locate these species requires tremendous efforts and resources through a traditional way, that is, to rely on experts to visit a identified place and take actions. This method is very difficult to scale up because the area can be sampled by experts are quite limited. To overcome this problem, [Kaggle](https://www.kaggle.com/c/invasive-species-monitoring) set up this platform for users to develop algorithm to distinguish invasive species from pictures of the forest. A efficient and accurate CNN model can be easily scaled up to identify invasive species in the forest with minimum resources.

## Dataset

Training and testing data sets for this project has been provided on [kaggle website](https://www.kaggle.com/c/invasive-species-monitoring/data). They are labeled as 0 (non-invasive) or 1 (invasive).The training data in this problem has 2295 images (2.8GB), and the test set has 1531 images(1.3GB). All images for this case have size of 1154 by 866 with 3 channels.<br>

## Environment

The script of this work was implemented with:

- Python 2.7
- Theano 0.8.2
- Keras 1.1.0
- Model was trained on AWS EC2.

## Scripts

There are 3 notebooks in this work. They are:<br>

- _00-Data-Exploring.ipynb_: Exploring the data characteristics such as balance of category distribution.
- _01-data-Preparation.ipynb_: It is to set up work folders and prepare data before feed them into the model.<br>

- _02-model-basic-linear-CNN.ipynb_: This notebook is to start with neural nets and simple convolutional network, and set up baseline for this work.In addition, the influence of two kinds of decay learning rate were also tested in this notebook.

- _03-CNN-Bn-Aug-Pseudo.ipynb_: This notebook uses a CNN with deeper and wider strucrture. It also further refines the results by using more advanced techniques such as Batch-normalizaiton, data augmentation, and Pseudo labeling.

## Reference

The code and techniques used in the work was based on a great [deep learning course](http://www.fast.ai/) by [Jeremy Howard](https://twitter.com/jeremyphoward?lang=en). And also the work by , as well as the works on [Kaggle kernels](https://www.kaggle.com/finlay/naive-bagging-cnn-pb0-985) for invasive species.<br>
