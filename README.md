# Convolutional Neural Networks - Waste Segregation case study for MS - AI/ML
> This is a case study for Convolutional Neural Networks - Waste Segregation case study

## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)

<!-- You can include any other section that is pertinent to your problem -->

## General Information
- Analyse the Data with 7 category Cardboard, Food_Waste, Glass, Metal, Other, Paper, Plastic
- Load the data, data preparation, preprocess image, data visualisation and plot the categories
- Visualize sample images, data splitting to Training, Test and Validation
- Build the Model, Train the model and Evaluate the model
- Plot the Training vs Validation Accuracy and Loss graph
- Refine the model as expected to prevent overfitting, augmentation, dropout, and learning rate
- Model compile with Optimizer Adaptive moment estimation
- Kares callbacks monitor the training process and auto-adjust based on the performance on validation data
- Evaluation of validation data and prediction
- Save the model
- Create the classification report and the confusion matrix
- Show some images, and predictions from generator
  
<!-- You don't have to answer all the questions - just the ones relevant to your project. -->

## Conclusions
Accuracy, Loss and Overfitting 
- Training accuracy improved
- Validation accuracy improved to 64%
- Learning rate reduced, smoothing training & avoiding overshooting
- Accuracies stayed close around 5%
- Overall, the epoch took 200s; it can be optimized with better hardware

Findings 
- Category Plastic has the best metrics, maybe due to more samples or features that is distinctive and stand out better in the training.
- There is model confusion between similar classes, like cardboard vs paper
- The performance is low due to data imbalance, category similarity, like paper vs cardboard
- The model would have improved with MobileNetV2, ResNet along with Batch Normalization, Dropouts, and Regularization, which is used here to reduce the overfitting. 
<!-- You don't have to answer all the questions - just the ones relevant to your project. -->


## Technologies Used
- Python Programming
- NumPy objects to create arrays/metrics to apply DL/ML modelsPython Programming
- Panda for Data Wrangling and Data manipulation purposes
- Seaborn to create visually appealing statistical graphics
- Matplotlib to create a range of plots and visualization
- RE for regular expression
- Scipy is user-firendly and efficient numerical routines
- Shutil is high-level file operations
- Random is built-in Python module that implements pseudo-random number generators
- ZipFile is built-in Python module for ZIP archives
- Pathlib is built-in Python module to interact with files and directories
- Scikit Learn ML (Machine Learning) Library
- Fastai Deep learning library on top of PyTorch
- Torchvision is a library part of PyTorch
- Pillow is the Python Imaging Library
- Keras is a high-level API for building and training NN (Neural Networks) on top of TensorFlow
- TensorFlow is an open-source ML framework for building and developing ML models
- All the versions are the latest as of April 2025

<!-- As the library versions keep on changing, it is recommended to mention the version of the library used in this project -->

## Acknowledgements
I want to credit upGrad for the Master of Science in Machine Learning and Artificial Intelligence (AI/ML) degree alongside IIIT-Bangalore, and LJMU, UK
- This project was inspired by all the Professors who trained us during the Convolutional Neural Networks, namely
  - G.Srinivasaraghavan - Professor, IIT-B

## Contact
Created by [@rajaravisekara] - feel free to contact me, Raja - Sr Architect - AI Cloud


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->
