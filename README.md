# Deep-Food

This project was done as part of my Bachlor's Thesis at the chair of [Connected Mobility](https://www.cm.in.tum.de/en/home/) (former Applied Informatics) at [Technical University of Munich](https://www.tum.de/en/). 
The goal of this thesis was to develop a program for recognizing food objects.

## Abstract
As obesity becomes more and more of a problem in developed countries, food logging is frequently used to help overweight people to balance their energy intake. Unfortunately, food logging is a tedious and inaccurate process. Computer vision and machine learning can help the user with this process. By taking an image of the meal, algorithms are able to make automated food intake assessments by detecting food items and their size. The goal of this thesis is the evaluation and implementation of a proof of concept application that can be used to facilitate and extend future food logging applications. For the task of food classification seven approaches were evaluated including feature classifiers like SIFT and SURF and convolutional neural networks. To enable the classification, segmentation, training and evaluation of classifiers, an extensive application was implemented using Python. The application supports data preprocessing and is designed so that it can be extended with additional image recognition concepts. 
With the aforementioned algorithms it was shown that algorithms are able to achieve a classification accuracy of 75% on 50 different food item classes if the algorithm suggests five possible candidates.
