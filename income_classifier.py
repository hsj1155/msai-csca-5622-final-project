# import libraries
from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.base import clone 

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# fetch dataset 
adult = fetch_ucirepo(id=2)


# data (as pandas dataframes) 
X = adult.data.features 
y = adult.data.targets 
