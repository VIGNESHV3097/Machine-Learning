#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


os.chdir("C:\\Users\\vignesh\\Desktop\\DATA SCIENCE\\CLUSTERING")


# In[4]:


data = pd.read_csv("HousingData.csv")


# In[5]:


data.info


# In[6]:


data.head()


# In[9]:


data.isnull().sum()


# In[13]:


x = data[["SellingPrice000s","HouseSize00Sqft"]].values
x


# In[17]:


from sklearn.cluster import KMeans
wcss=[]
for i in range(1,7):
    kmeans= KMeans(n_clusters=i, init="k-means++")
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


# In[18]:


plt.plot(range(1,7),wcss)


# In[20]:


wcss


# In[22]:


kmeans = KMeans(n_clusters=3, init= "k-means++")
y_kmeans= kmeans.fit_predict(x)


# In[23]:


y_kmeans


# In[26]:


pd.concat([data, pd.DataFrame(y_kmeans)], axis=1)


# In[35]:


plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1], c= "red", label= "cluster1")
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1], c="blue", label="cluster2")
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1], c="yellow", label="cluster3")
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1], c="black", label="cluster4")
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1], c="green", label="cluster5")


# In[37]:


import scipy.cluster.hierarchy as sch


# In[39]:


dendrogram= sch.dendrogram(sch.linkage(x,method="ward"))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




