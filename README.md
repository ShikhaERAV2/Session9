## CIFAR-10 Image Classification using PyTorch
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. The dataset is divided into 391 training batches and 71 test batch, each with 128 images. The test batch contains exactly randomly-selected images from each class. The training batches contain the remaining images in random order. 
The various classes are ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck') 

Here are the classes in the dataset, as well as 10 random images from each: 
![image](https://github.com/ShikhaERAV2/Session8/assets/160948226/92783e22-2d42-4bfd-8c31-b26040152108)

## Model Description
This is a Multiple convolution layers in Convolutional Neural Network for Image identification trained on CIFAR10 dataset.Basic model structure 

C1 Convolution Block->C2 Convolution Block->C3 Convolution Block->C4 Convolution Block

## Code Structure
- S9_Final.ipynb: The main Jupyter Notebook contains the code to load data in train and test datasets -> transform data-> load model (defined in model.py)-> train model -> test the model -> Check the accuracy of the model thus trained. This model uses Batch Normalization.
- model.py: This file contains the definition of the model. Basic architecture of the model is defined with multiple convolution layers and fully connected layers.
- utils.py: This file contains the utility functions like display of the sample data images and plotting the accuracy and loss during training.

## Requirements
 - Pytorch
 - Matplotlib

## Model
Model Name : Net

*Test Accuracy = 73.20% (max)

*Train Accuracy = 68.34%

*Total params: 2,034,040

Analysis:

- Model is Overfitting.
- Number of parameter is high though but model can perform better with more number of Epoch.
- Batch Normailzation with image augmentation helps improve the model performance.
- Depthwise Separable Convolution imporved the model performance - Test Accuracy of 65% Epoch 20
- Dilated Convolution improved the model performance - Test Accuracy of 73% Epoch 20
- Albumentation required model to train for longer time to achive the desiered accuracy.

Model Performance:
![image](https://github.com/ShikhaERAV2/Session8/assets/160948226/7a876aa4-02f4-4e53-9929-6707636b0492)


Mis-Classified Images:
![image](https://github.com/ShikhaERAV2/Session8/assets/160948226/a9c7d1ff-e78e-4431-ac9a-94265978fb4a)

