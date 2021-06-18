

#Import required libraries 
from tensorflow import keras #library for neural network
import pandas as pd #loading data in table form  
import seaborn as sns #visualisation 
import matplotlib.pyplot as plt #visualisation
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import normalize #machine learning algorithm library

#Reading data 
data = pd.read_csv("./data/Iris.csv")
print("Describing the data: ", data.describe())
print("Info of the data:", data.info())

print("10 first samples of the dataset:",data.head(10))
print("10 last samples of the dataset:",data.tail(10))

print(data["species"].unique())
data.loc[data["species"]=="Iris-setosa","species"]=0
data.loc[data["species"]=="Iris-versicolor","species"]=1
data.loc[data["species"]=="Iris-virginica","species"]=2
print(data.head())

data=data.iloc[np.random.permutation(len(data))]
print(data.head())

