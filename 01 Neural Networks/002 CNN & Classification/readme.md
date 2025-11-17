# Neural Networks

## Covolutional Neural Networks (CNNs) & Classfication Models
CNN is well-known to have good performance in extracing spatial information from the given data. Therefore, it is important to know how to utilize CNN models when dealing with data such as images, LiDAR data, or other sensory data which may contain spatial information. 

In this course, we are not going to design the CNN architecture, but we are going to use one of the well-designed CNN models developed by other researchers. Instead, we will modify the later part of the neural network (fully-connected layers) to classify CIFAR-10 dataset. To minimize training time and for better results, we will use the pretrained weights trained with the Imagenet dataset as the initial condition to the neural network. 

Please refer to the example codes (train.py and cnn_network.py) and the following website [Transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/). Use CIFAR-10 dataset with [Keras library](https://keras.io/api/datasets/cifar10/) and train the CNN network. You may select any CNN architecture, but for your convinience, select one from the following keras documentation: https://keras.io/api/applications. You may lock (set trainable to false) the CNN layers for faster training.

## Author
Hwanmoo Yong