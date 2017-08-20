# Machine Learning Engineer Nanodegree - Capstone Proposal

JY<br>
June 15th, 2017

## Abstract

A framework of a project that utilizing CNN to solve image classification problem will be presented in this proposal. This proposal introduces the framework in term of: (1) background (2) dataset (3) solution statement (4) benchmark model (5) evaluation metric and (6) project design.

## 1\. Domain Background

Since AlexNet was developed in 2012 and won the [ILSVRC (ImageNet Large Scale Visual Recognition Competition)](http://www.image-net.org/challenges/LSVRC/) with an top-5 error as low as 15.4%, CNN became the golden standard for image classification. Thereafter, the training techniques (such as relu/[selu](https://arxiv.org/abs/1706.02515) activations, [batchnorm](https://arxiv.org/abs/1502.03167), speudo labeling) and neural network structure (such as ZF Net-2013, VGG-2014, [LeNet/Inception V1-2014](https://arxiv.org/abs/1409.4842), [ResNet-2015](https://arxiv.org/abs/1512.03385), [GoogLeNet/Inception V3-2015](https://arxiv.org/abs/1512.00567), [Xeption-2016](https://arxiv.org/abs/1610.02357)) were consistently improved to make more efficient and accurate predictions. CNN that was developed in 2015 (ResNet) has already [outperformed humans](http://karpathy.github.io/2014/09/02/what-i-learned-from-competing-against-a-convnet-on-imagenet/) in ILVRC. Beside ILSVRC, CNN has also been used as a powerful tool for various applications such as object recognition/detection in different scenarios, neural styles, image captioning etc.

In this work, we will look at applying CNN to [distinguish invasive species in the forest in Kaggle Competition](https://www.kaggle.com/c/invasive-species-monitoring). The invasive species can put ecosystems at risk. To track and locate these species requires tremendous efforts and resources by using the traditional way, that is, to rely on experts to visit a identified place and take actions. This method is very difficult to scale up because the area can be sampled by experts are quite limited. To overcome this problem, [Kaggle](https://www.kaggle.com/c/invasive-species-monitoring) set up this playground for users to develop algorithm to distinguish invasive species from pictures of the forest. A efficient and accurate CNN model can be easily scaled up to identify invasive species in the forest with minimum resources.

## 2\. Problem Statement

The invasive species monitoring problem is a binary classification problem. The input data are images with height of 866 and width of 1154 pixels, that are taken from different places in the forest. At the mean time, the output is probability of an image containing invasive species. In this case, a CNN with softmax activation following the last dense layer could be a suitable solution.

## 3\. Datasets and Inputs

The training/test data and training labels for this problem are provided and [can be downloaded from Kaggle website](https://www.kaggle.com/c/invasive-species-monitoring/data). They are pictures taken in a Brazilian national forest. The training pictures are labeled as 0 (non-invasive) or 1 (invasive) as illustrated below.

![non-invasive](89.jpg)

![invasive](19.jpg)

The training data in this problem has 2295 images, and the test set has 1531 images. The training set will be split into training and validation set to train and verify model accuracy. Then the trained model will be used to make predictions of the test data （the probability of a picture has invasive species).

## 4\. Solution Statement

To solve this problem, a CNN network can be a good choice. This is a typical image classification problem (although it is not one picture one object). So we can start with a simple CNN structure model to have a baseline. It can consist of convolutional layers to extract features at different levels, maxpooling layer, flatten layer, and dense layers with a softmax activation function to convert the output to probability of invasive species. Once obtain the baseline results, we can gradually improve the model by using more advanced techniques (such as batchnorm, data augmentation) and developed network structures (such as VGG net).

## 4\. Benchmark Model

For this problem, the results from a basic CNN model (e.g. 1~3 conv layers, 1 maxpooling 1~2 flatten, and ) can be used as a benchmark results for comparison. As a binary classification problem, the performance of the model can be evaluated based on the prediction [accuracy](https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers) on validation data.

## 5\. Evaluation Metrics

Accuracy can be a suitable evaluation metrics for this problems. It is defined as:

$(\sum TP+ \sum TF)/Total Population $

Where TP stands for true positive, and TF stands for true false. Since accuracy is usually used for the case when number of instance per class is balanced, we can also utilize [confusion matrix](http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/) to show the detailed:

- TP: true positive
- TN: true negative
- FP: false positive
- FN: false negative

of predictions, and examine the balance of instance from two classes.

## 7\. Project Design

The following workflow is designed to establish a basic model and improve the predictions gradually.

1. Prepare data: load data, arrange data properly to feed into the model.
2. Establish basic model (simple CNN in this case).
3. Improve model and spot-check model performance (1) VGG model (2) VGG + batchnorm (3) VGG + batchnorm + data augmentation.
4. Ensemble (combine multiple predictions from CNN model).
5. Results visualization: visualize correctly and incorrectly predicted images and analyze the model performance.
6. Conclusion: compare and summerize performance of different models.

## References

[1] Adit Deshpande, The 9 Deep Learning Papers You Need To Know About (Understanding CNNs Part 3), <https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html>.<br>
[2] Kaggle Invasive Species Monitoring, <https://www.kaggle.com/c/invasive-species-monitoring>.
