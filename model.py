#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import pickle
import requests
import json


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,ConfusionMatrixDisplay


# In[3]:


from numpy import asmatrix,asarray
import PIL
from PIL import Image


# In[4]:


df = pd.read_csv('mnist_train.csv')


# In[5]:


X = df.drop('label',axis = 1)
y = df['label']


# In[6]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 101)


# In[7]:


model = SVC()
model.fit(X_train,y_train)


# In[8]:


predictions = model.predict(X_test)


# In[9]:


print(accuracy_score(y_test,predictions)*100,' %')


# In[10]:


with open('digit_recognizer.pkl', 'wb') as file:
    pickle.dump(model, file)

