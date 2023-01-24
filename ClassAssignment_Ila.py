#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#CLASS ASSIGNMENT #ILA SHARMA #C0852428


# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.stats.mstats import winsorize
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD


# In[4]:


data_train = pd.read_csv("ML-MATT-CompetitionQT1920_train.csv",encoding='windows-1252')
data_test = pd.read_csv("ML-MATT-CompetitionQT1920_test.csv", encoding='windows-1252')

print(data_train.shape)
print(data_test.shape)

data_train.head(10)

data_test.head(10)

data_train.describe()

data_test.describe()



data_train = data_train.drop(columns=['CellName','Time'], axis=1)
data_test = data_test.drop(columns=['CellName','Time'], axis=1)


# In[5]:


"""We have an unusual value in **maxUE_UL+DL** column of our dataframe , so we need to handle this by removing and replacing values"""

data_train[data_train.eq('#¡VALOR!').any(1)]

data_train['maxUE_UL+DL'].value_counts()['#¡VALOR!']

data_test[data_test.eq('#¡VALOR!').any(1)]


# In[6]:


"""**Now we understand that the value '#¡VALOR!' needs to be replaced by a numeric value**"""

data_train['maxUE_UL+DL'].unique()


"""**We will replace the value '#¡VALOR!' by 0**"""

data_train['maxUE_UL+DL'] = data_train['maxUE_UL+DL'].replace('#¡VALOR!',0)

data_train['maxUE_UL+DL']

data_train[data_train.eq('#¡VALOR!').any(1)]


# In[7]:


"""**We have finally replaced the value and now we will proceed with Normalization**

**Normalization of training dataset**
"""

from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(data_train)

df_train = pd.DataFrame(np_scaled, columns = data_train.columns)
df_train


# In[8]:


"""**Normalization of test Dataset**"""

np = min_max_scaler.fit_transform(data_test)

df_test = pd.DataFrame(np, columns = data_test.columns)
df_test


# In[9]:


"""**Missing values**"""

a = data_train.isnull().sum()
b = data_test.isnull().sum()
missing_value_feature_all_data = []

def find_missing(df):
  for i in df.index:
    
    if df[i] != 0:
      missing_value_feature_all_data.append(i)

  print(missing_value_feature_all_data)

find_missing(a)

find_missing(b)

data_train.isnull().sum() * 100 / len(data_train)

data_test.isnull().sum() * 100 / len(data_test)


# In[10]:


"""**Handling missing values**"""

data_train = data_train.fillna(method='ffill').fillna(method='bfill')
data_test = data_test.fillna(method='ffill').fillna(method='bfill')

x = data_train.drop(columns = ['Unusual'], axis = 1)
y = data_train['Unusual']

x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=0.2,random_state = 1)



X = data_train.drop(columns = ['Unusual'], axis = 1)
y = data_train['Unusual']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        
    elif train==False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")



from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression(solver='liblinear')
lr_clf.fit(X_train, y_train)

print_score(lr_clf, X_train, y_train, X_test, y_test, train=True)
print_score(lr_clf, X_train, y_train, X_test, y_test, train=False)




# In[11]:


"""**Checking imbalance in target variable**"""

sns.histplot(data_train['Unusual'])



def over_sample_train_test(x,y):
    ros=RandomOverSampler(random_state=10)
    ros.fit(x,y)
    x_res,y_res=ros.fit_resample(x,y)
    x_train,x_val,y_train,y_val=train_test_split(x_res,y_res,test_size=0.3,random_state = 1)
#     x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=0.2,random_state = 1) # if without oversampling
    return x_train,x_val,y_train,y_val

x = data_train.drop(columns = ['Unusual'], axis = 1)
y = data_train.Unusual
x_train,x_val,y_train,y_val = over_sample_train_test(x, y)

f, (ax1, ax2) = plt.subplots(1, 2, figsize = (10,4))
sns.histplot(y_train, ax = ax1)
sns.histplot(y_val, ax = ax2)


# In[16]:


"""**Logistic Regression**"""
import seaborn as sns
def plot_confusion_matrix(model, X, y):
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True,fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# lr = LogisticRegression(solver='liblinear', penalty = 'l1', C = 0.05, random_state = 42, max_iter = 1000)
regulation_parameter = 0.005
lr = LogisticRegression(solver='liblinear', C = regulation_parameter, random_state = 42, max_iter = 1000)

def apply_model(model,x_train,x_val,y_train,y_val):
    print('Logistic Regression')
    model.fit(x_train,y_train)
    y_pred = model.predict(x_val)
    print('')
    print('Train Score:  ',model.score(x_train,y_train))
    print('Validation Score:   ',model.score(x_val,y_val))
    print('')
    plot_confusion_matrix(model, x_val, y_val)
    print(classification_report(y_val,y_pred))

apply_model(lr, x_train,x_val,y_train,y_val)

lasso = LogisticRegression(solver='liblinear', penalty = 'l1', C = 0.05, random_state = 42, max_iter = 1000)


def apply_model(model,x_train,x_val,y_train,y_val):
    print('Logistic Regression')
    model.fit(x_train,y_train)
    y_pred = model.predict(x_val)
    print('')
    print('Train Score:  ',model.score(x_train,y_train))
    print('Validation Score:   ',model.score(x_val,y_val))
    print('')
    plot_confusion_matrix(model, x_val, y_val)
    print(classification_report(y_val,y_pred))

apply_model(lasso, x_train,x_val,y_train,y_val)






# In[ ]:




