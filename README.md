# Tensorflow2 Classification CNN Training Template #
This code is a training template of single-label classification CNN's with TensorFlow2. It should contain 3 different approaches which can be fine tuned your 
personal use and preferences. While *Keras* is the recommended way to train by Google, it has several major disadvantages.
First being the recurring compatibility bug with *CUDA* and the convolutional layer. Also the changed signatures between
tf-keras and native keras. It is not possible to mix the two due to mismatch in types. However, tf-keras lacks some 
implementations that are available in the native *Keras* implementation. It is easy and frustrating to get caught in 
the internal dependency hell.  
To avoid all this I did change to plain tf layers. Full control of session and graph make it easier to fine tune
and modify models for deployment (like adding an extra image loading section in front of your feature network). However,
it should be noted that multiple tf layers are deprecated now while Google suggests to use keras layers instead.

### Hint ###
Attention! Due to a TensorFlow2 import bug, this project is currently on hold
https://github.com/tensorflow/tensorflow/issues/32574

### Training with TensorFlow layers ###
Since my old templates were all written without eager mode I also wanted to keep it that way for the first draft. 
Data-loading is part of another tutorial you can find on 
[Medium](https://medium.com/ri-rewe-digital/manage-your-image-dataset-with-googles-automl-ac5a45c23c9d). 
The training loop is an adoption of a ML class from 
[Stanford University](https://github.com/cs230-stanford/cs230-code-examples/tree/master/tensorflow/vision).
Train, eval and test are separated into different sections which share weights from the feature extractor and classification section.

### Training with Keras layers ###
Implementing models with Keras did not change much going from tf1 to tf2. Therefore, the code does not contain much detail.
Basically tf1-keras tutorials still apply what is one of the major benefits of Keras.

### Training with TensorFlow Estimator ###
Google's tf-estimator is not recommended in their official documentation anymore. Instead it suggests to use Keras instead since
tf-estimator itself also uses this API.
Hence, there is currently no implementation of tf-estimator included at the moment.

### References ###
https://github.com/Tencent/tencent-ml-images  
https://github.com/cs230-stanford/cs230-code-examples/tree/master/tensorflow/vision  
https://github.com/ri-rewe-digital/automl-training  
https://www.tensorflow.org/beta/guide/effective_tf2  
