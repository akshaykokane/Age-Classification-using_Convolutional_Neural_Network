# Age Detection using Convolutional Neural Network (TensorFlow)

The problem discussed is about the age classification using Convolutional Neural Network (CNN) and improving the existing model and training speed of the model. I am using Adience dataset for training model as it contains variety of images as used by https://talhassner.github.io/home/publication/2015_CVPR. 

## Getting Started

  Neural network  take large number of training examples as input and develop a system which can learn from the training examples. Convolutional neural networks (CNN) are the current state-of-the-art model architecture for image classification tasks. CNN is a class of deep, feed-forwarding artificial neural networks, most commonly applied to analyzing visual imagery. CNN uses different convolutional layers for classification. The first layer is the input layer where the dataset is fed to the network. The middle layers are called hidden layers, which are responsible for classification. The last layer is the output layer. CNN usually consists of the convolutional, pooling, full-connected layers.

I am using the architecture proposed by 
```
1] Hassner, T. (2018). Age and Gender Classification Using Convolutional Neural Networks. [online] Available at: https://talhassner.github.io/home/publication/2015_CVPR 
[2] Ekmekji, A. (2018). Convolutional Neural Networks for Age and Gender Classification. [online] Cs231n.stanford.edu. Available at: http://cs231n.stanford.edu/reports/2016/pdfs/003_Report.pdf
[3] https://github.com/dpressel/rude-carnie
 
```
In the approach provided in [1], the researcher has used CNN for age and gender classification.  In this approach, the dataset used was the Adience benchmark for age and gender classification of unfiltered images. They showed that despite the images were having many different alignments, different sizes, different resolutions and variety of people from different background, the model developed was able to classify age and gender by using Convolutional Neural Network (CNN). They provided the basic CNN architecture for age classification.

#### Dataset
![alt text](https://talhassner.github.io/home/projects/Adience/adience_ageandgender.png)

The dataset used is Adience Dataset which contains unfiltered faces for age classification. The sources of the images included are Flick albums, assembled by automatic upload from iPhone 5(or later) smart phone devices, and released by their authors to the general public under the Creative Common (CC) license 
The entire Adience collection includes images around 19k of 2,284 subjects. Each image was having age range associated with it. The dataset was having images of younger generation. Its bit obvious as the social network is been used more by younger generation. For simplicity we reinitialize the dataset with labels to each age range by using Excel operations. Using the below formula, we were able to refine the dataset with the single labels. [Ref](https://talhassner.github.io/home/projects/Adience/Adience-data.html)

### Preprocessing (TFRecord)

TFRecord file format is a simple record-oriented binary format developed by the Google that many TensorFlow applications use for training different models. We first focused on converting the dataset into TFRecord format. It was crucial and challenging task to convert the dataset to the TFRecord format. We developed the python script to stored the dataset (Audience Benchmark Dataset) in TFRecord format and stored images in byte format and labels in int64 format.
For any successful implementation its very necessary that the dataset is divided into training, testing and evaluation. It is the important step in the machine learning. So we divided that dataset into 60% training 20% testing and 20% validation.
```
Writing to TFRecord:

Data -> FeatureSet -> Example -> Serialized Example -> TFRecord.


Reading from TFRecord

TFRecord -> SerializedExample -> Example -> FeatureSet -> Data

```
### Model


## Prerequisites

1) Python 2.7 or above

    ```
    python3 --version
    
   ```
2) Conda

### Installing

A step by step series of examples that tell you how to get a development env running

```
1) Create virtual enviornment in conda
	conda create -n TensorFlow pip python=3.6 
2) Activate environment
	activate TensorFlow
3) Install TensorFlow
	pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.12.0-cp36-cp36m-linux_x86_64.whl

```
To check if installed successfully, in terminal run 

```
python import tf as tensorflow

```

If everything worked fine, then you will get no error message.

## Run

1) Prepare TfRecord File by executng build_input.py

```
python build_input.py
```

2) Build model by executing model.py

```
python model.py
```

## Results


## Authors

Akshay Kokane
akshaykokane.com

## Acknowledgments

* [Thanks for open-sourcing the architecture](https://talhassner.github.io/home/publication/2015_CVPR)

