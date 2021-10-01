#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.model_selection import train_test_split


# In[2]:


b=load_iris()


# In[3]:


dir(b)


# In[4]:


df=pd.DataFrame(b.data,columns=b.feature_names)


# In[5]:


df.head()
df["target"]=b.target


# In[6]:


x=df.drop("target",axis="columns")
y=df["target"]


# In[11]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


# In[12]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# In[13]:


model1 = SVC()
model1.fit(x_train,y_train)
model1.score(x_test,y_test)


# In[14]:


model2=RandomForestClassifier()
model2.fit(x_train,y_train)
model2.score(x_test,y_test)


# In[15]:


model3=LogisticRegression()
model3.fit(x_train,y_train)
model3.score(x_test,y_test)


# # KFOLIND

# In[17]:


from sklearn.model_selection import cross_val_score


# In[18]:


df = cross_val_score(SVC(),x,y,cv=4)


# In[19]:


import numpy as np


# In[20]:


np.average(df)


# In[21]:


df = cross_val_score(SVC(C=10),x,y,cv=4)


# In[22]:


df=cross_val_score(SVC(C=20,kernel="linear",gamma="scale"),x,y,cv=4)


# In[23]:


np.average(df)


# In[24]:


df = cross_val_score(RandomForestClassifier(n_estimators=10),x,y,cv=4)


# In[25]:


np.average(df)


# In[26]:


df = cross_val_score(RandomForestClassifier(n_estimators=20),x,y,cv=4)


# In[27]:


np.average(df)


# In[28]:


df = cross_val_score(RandomForestClassifier(n_estimators=30),x,y,cv=4)


# In[29]:


np.average(df)


# In[30]:


df = cross_val_score(RandomForestClassifier(n_estimators=50),x,y,cv=4)


# In[31]:


np.average(df)


# In[35]:


df=cross_val_score(LogisticRegression(),x,y,cv=4)


# In[36]:


np.average(df)


# In[ ]:




