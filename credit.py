import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

import sklearn
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score 
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import average_precision_score, precision_recall_curve

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
 #from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

credit_card_data=pd.read_csv("creditcard.csv")
# tail and head function is for print the last and first five rows
print(credit_card_data.head())
print(credit_card_data.tail())

# some more information about dataset
print(credit_card_data.info())

# give us sum of missing value in column
print(credit_card_data.isnull().sum())

# 0 repersent the normal transaction and 1 repersent the fraud transaction which 
# is shown in class column
# now we will check the how may legit transaction and how many fraud transaction
print(credit_card_data['Class'].value_counts())
# this detaset is heighly unbalnced
# now sepreate the data for analysis

legit=credit_card_data[credit_card_data.Class==0]
fraud=credit_card_data[credit_card_data.Class==1]

# gives us number of fraud row and column 
print(legit.shape)
print(fraud.shape)

# capston8 it will shows the quantity of fraud and legit transaction in pie chart
print((credit_card_data.groupby('Class')['Class'].count()/credit_card_data['Class'].count()) *100)
((credit_card_data.groupby('Class')['Class'].count()/credit_card_data['Class'].count()) *100).plot.pie()


#static measurement of dataset
# see at 19:44 give us count,mean,std,min,and %of transaction below at some price
print(legit.Amount.describe())

# gives us count,mean,std,min,and % of transaction below at some price offraud trancation

print(fraud.Amount.describe())

# compare the value of both transaction
print(credit_card_data.groupby('Class').mean())

# under sampling
# build a sample dataset containing distribution of normal transaction and fraudulent trasactions
#  it will take random 492 transaction from the dataset 
# number of fraudlent trasactions-> 492
legit_sample = legit.sample(n=492)

# concatinate of two datafram
#axis 0 means the data is combined row wise first legit then fraud
new_dataset = pd.concat([legit_sample,fraud],axis=0)

print(new_dataset.head())
print(new_dataset.tail())

print(new_dataset['Class'].value_counts())

# spilitting the data into features and target(target is 0 0r 1)
print(new_dataset.groupby('Class').mean())

X=new_dataset.drop(columns='Class',axis=1)
Y=new_dataset['Class']
print(X)
# it will contain only label 0 or 1
print(Y)

# splitting the data into trainig data and testing data
# 0.2 means 20 percent of data
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

print(X.shape, X_train.shape,X_test.shape)

# now trained a meachine learning model and check the model
# model training 
# logistic regration

model=LogisticRegression()

# training the Logistic Regression model with traing data
model.fit(X_train,Y_train)

# Model evalution
# accuracy on trainig data
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction, Y_train)
print("accuracy on trainig data :", training_data_accuracy)

# accuracy score on test data

X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print("accuracy score on test data",test_data_accuracy)

# checking the correlation in heatmap
corr = credit_card_data.corr()
plt.figure(figsize=(24,18))

sns.heatmap(corr, cmap="coolwarm", annot=True)
plt.show()

# Create a bar plot for the number and percentage of fraudulent vs non-fraudulent transcations
plt.figure(figsize=(7,5))
sns.countplot(credit_card_data['Class'])
plt.title("Class Count", fontsize=18)
plt.xlabel("Record counts by class", fontsize=15)
plt.ylabel("Count", fontsize=15)
plt.show()

# now we will check the fraud and legit transaction column wise
# Accumulating all the column names under one variable
cols = list(X.columns.values)

normal_records = credit_card_data.Class == 0
fraud_records = credit_card_data.Class == 1
# here red line indecate the fraud transaction and green line indecate the legit transaction 
# in column wise

plt.figure(figsize=(20, 60))
for n, col in enumerate(cols):
  plt.subplot(10,3,n+1)
  sns.distplot(X[col][normal_records], color='green')
  sns.distplot(X[col][fraud_records], color='red')
  plt.title(col, fontsize=17)
plt.show()






