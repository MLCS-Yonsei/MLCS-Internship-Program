# Neural Networks

## Multi Layer Perceptron (MLP)
Multi Layer Perceptron (MLP) is a simple but important network used by a variety of projects. It consists only of fully connected layers (Dense layers in Keras) consisting of weights, biases and activation functions. The last activation function of a neural network is usually a softmax function which converts the logit to a probability distribution (for the classification).

In this course, we are going to train two MLP networks with MNIST dataset. The original MNIST dataset does not provide the validation dataset, but we will split the given dataset into three as we did in the previous course. Afterwards, we will compare the results from two different neural networks (e.g. deeper vs. shallower neural networks or larger vs. smaller layer size)

Please refer to the example code (mlp.py) and build and train two different models with the MNIST dataset divided by the same rules. It doesn't matter how big or small the networks are, but make sure if there is a different between the two. Note that when training, only train and validation datasets are utilized.

## Author
Hwanmoo Yong