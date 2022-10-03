# MODULE 13 CHALLENGE : Venture Funding With Deep Learning

For this challenge our role is to act as a risk management associate at a venture capital firm. We are tasked with creating a model that predicts whether applicants will be successfu if funded by the firm. The start of this analysis comes from a csv file that contains more than 34,000 organizations that have recieved funding from the firm, and it contains information that includes whether or not it became successful. We then use our machine learning knowledge to create a binary classifier model that will predict whethe an applicant will become a successful business. This challenge consists of three technical deliverables, we are to preprocess data for a neural network model, use the model-fit-predict pattern to compile and evaluate a binary classification model, and attempt to optimize the model two times.


The changes I made to optimize the model include:

Attempt #1
I increased the number of neurons in the output layer to 3.
I increased the number of hidden layers.
I increased the number of epochs.
    
This attempt increased the model's accuracy but only by .05%
    
Attempt #2
I increased the number of neurons in the output layer to 5.
I increased the number of nodes per layer by 10 in each hidden layer.
I decreased the number of epochs.
    
This attempt did not increase the model's accuracy, it ended up decreasing the accuracy by almost .2%

---

## Technologies

This project leverages python 3.7 with the following packages:

* [Pandas](https://github.com/google/pandas) - Pandas is a powerfull tool for data analysis and manipulation. Pandas provides a plethora of useful functions that make it easy to express, analyze, and manipulate data.

* [scikit-learn](https://scikit-learn.org/stable/) - This is a machine learning library for the python programming language. This library allows for the use of multiple machine learning models, tools, and algorithms.

* [TensorFlow](https://www.tensorflow.org/) - TensorFlow is an end-to-end open source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries, and community resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML-powered applications.


---

## Installation Guide

Before running the application first install the following dependencies.

```
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder

```

---

## Usage

To use the deep neural file simply clone the repository and open the venture_funding_with_deep_learning.ipynb file in jupyter notebook.

Upon opening the file you will have the option to run the whole note book and that will provide you with all of the calculations, evaluations, and visualizations for the analysis of the neural network data. Some screenshots of that in action can be seen below via this link below.

* [SCREENSHOTS](https://github.com/AdamCooke22/module_13/tree/main/screenshots) 

## Contributors

Completed by Adam Cooke

---

## License

MIT
