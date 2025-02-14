#%% Load dataset and preprocessing 

import numpy as np
import pandas as pd 

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import  StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

import warnings
warnings.filterwarnings("ignore")

#load dataset
california = fetch_california_housing()




# %% linear, ridge, lasso regressions models, train tuning

#%% elastic net traning and hyperparameter tuning

#%% Model Comparision 



