# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 09:33:20 2021

@author: mmuno
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
from sklearn import preprocessing
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from sklearn.cluster import SpectralBiclustering, SpectralCoclustering
from numpy import argsort
from matplotlib.pyplot import show, imshow, figure, subplot, suptitle, tight_layout, xticks, xlabel
from sklearn.datasets import make_biclusters
from sklearn.metrics import consensus_score
import igraph as ig
from igraph import*
from scipy import sparse
import networkx as nx

path = "./"
np.random.seed(1234567)

df_ = pd.read_csv(path +"online_retail_II.csv")
df = df_.copy()

# Data exploration
df.head()
df.info()
df.nunique()
df.isnull().sum()
(df['Price']<0).sum()
(df['Quantity']<0).sum()
df.boxplot(column=['Price','Quantity'])

### DATA PREPROCESSING

# Dropping out null values
df.dropna(subset=['Customer ID'], inplace=True)
df.isnull().sum()

# Invoices starting with "C" are either cancelled or returned goods so we are dropping out the item from the data set.
df["Invoice"].str.contains("C", na=False).sum()
df=df[~df["Invoice"].str.contains("C", na=False)]

# Clear any items Quantity less than 1 and outliers
df= df[(df["Quantity"]>0)&(df["Quantity"]<30000)]
df.boxplot(column=['Price','Quantity'])

df.describe().T

# removing StockCode that are not products
df = df[~((df['StockCode']== 'ADJUST')|(df['StockCode']== 'ADJUST2')|(df['StockCode']=='BANK CHARGES')|(df['StockCode']=='D')|(df['StockCode']=='DOT')|(df['StockCode']=='PADS')|(df['StockCode']=='TEST001')|(df['StockCode']=='TEST002'))]
df = df[~(df['StockCode']== 'M')]
df = df[~(df['StockCode']== 'POST')]


### RFM MODEL
df["TotalPrice"]=df["Quantity"]*df["Price"]

# Recency
df["InvoiceDate"].max()
today_date = dt.datetime(2011, 12, 11)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

#Data Frame for Customer ID with the variables of InvoiceDate (will be changed to Rececny in next step), Invoice number (will be changed to "Frequency") and Total sum of transactions (will be changed to "Monetary")
rfm = df.groupby('Customer ID').agg({'InvoiceDate': lambda date: (today_date - date.max()).days,'Invoice': lambda num: num.nunique(),'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

# We reassing the column names in previous cell. 
# Recency: We subtracted today's date from the last transation date to calculate recency in weeks
# Frequency: We counted invoiced per each customer. Since each invoice refers to one transaction we can use them as frequency. 
# Monetary: We sum up all transations per each customer as their monetary value.

rfm.columns = ['Recency', 'Frequency', 'Monetary']
# To avoid miscalculation we run a acheck querry to eleminate 0 values in Monetary and Frequency
rfm = rfm[(rfm["Monetary"]) > 0 & (rfm["Frequency"] > 0)]
# Converting Recency in weeks
rfm['Recency'] = rfm['Recency']/7

rfm.describe().T

# function for plotting
def rfmplot2d(rfm):
    rfm.plot.scatter('Recency', 'Frequency')
    rfm.plot.scatter('Recency', 'Monetary')
    rfm.plot.scatter('Frequency', 'Monetary')

def rfmplot3d(rfm):
    threedee = plt.figure().gca(projection='3d')
    threedee.scatter(rfm['Recency'], rfm['Frequency'], rfm['Monetary'])
    threedee.set_xlabel('Recency')
    threedee.set_ylabel('Frequency')
    threedee.set_zlabel('Monetary')
    plt.show()

rfmplot2d(rfm)
rfmplot3d(rfm)

# Normalizing (Log transformation)
rfm_log = rfm.copy()
rfm_log = rfm_log.applymap(lambda x: np.log(x+1))
rfmplot2d(rfm_log)
rfmplot3d(rfm_log)

def rfmplotclusters(rfm, y):
    type_colors_map = dict(a="red", b="blue", c="green", d="yellow", e= "purple", f="black",g="orange",h="pink")
    colors = list(type_colors_map.values())
    
    rfm.plot.scatter('Recency', 'Frequency', c = [ colors[k] for k in y])
    rfm.plot.scatter('Recency', 'Monetary', c = [ colors[k] for k in y])
    rfm.plot.scatter('Monetary', 'Frequency', c = [ colors[k] for k in y])
    
    threedee = plt.figure().gca(projection='3d')
    threedee.scatter(rfm['Recency'], rfm['Frequency'], rfm['Monetary'], c = [ colors[k] for k in y])
    threedee.set_xlabel('Recency')
    threedee.set_ylabel('Frequency')
    threedee.set_zlabel('Monetary')
    plt.show()


### CLUSTERING
X = rfm[['Recency', 'Frequency', 'Monetary']]
X_log = np.array(rfm_log[['Recency', 'Frequency', 'Monetary']])

# Gaussian Mixture Model
n_comp = np.arange(1,11)
models = [GaussianMixture(n, covariance_type = 'full', random_state=0).fit(X_log) for n in n_comp]
plt.plot(n_comp,[m.bic(X_log) for m in models],label = 'BIC')
plt.plot(n_comp,[m.aic(X_log) for m in models],label = 'AIC')
plt.legend(loc='best')
plt.xlabel('n_comp')

gm3 = GaussianMixture(n_components = 3, covariance_type = 'full', random_state=0).fit_predict(X_log)
rfmplotclusters(rfm_log, gm3)

gm4 = GaussianMixture(n_components = 4, covariance_type = 'full', random_state=0).fit_predict(X_log)
rfmplotclusters(rfm_log, gm4)

gm5 = GaussianMixture(n_components = 5, covariance_type = 'full', random_state=0).fit_predict(X_log)
rfmplotclusters(rfm_log, gm5)

rfm['Average value'] = rfm['Monetary']/rfm['Frequency']
rfm['Cluster'] = gm4
profile = rfm.groupby('Cluster').mean()
profile['Numb_cust'] = rfm.groupby('Cluster')['Monetary'].count()


### Customer - Product Matrix
df_net = df.groupby(['Customer ID', 'StockCode'])['Quantity'].sum().unstack().fillna(0)
df_net_bi = df.groupby(['Customer ID', 'StockCode'])['Quantity'].sum().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)

df_array = sparse.csr_matrix(np.array(df_net_bi))
df_array_T = sparse.csr_matrix(np.array(df_net_bi.T))

net_cus = df_array@df_array_T
net_pro = df_array_T@df_array

net_cus = net_cus.toarray()
net_pro = net_pro.toarray()

np.fill_diagonal(net_cus, 0)

a = np.all(net_cus==0,axis=0) # removing customer that are not connected with others
net_cus = net_cus[~a]
net_cus = net_cus.T[~a]

# Community Detection
test = net_cus[:100,:100] # trying with a smaller matrix first
i = ig.Graph.from_networkx(nx.from_numpy_matrix(test))

x = i.community_fastgreedy() # fast greedy algorithm
clusters = x.as_clustering()
clusters.membership
clusters.modularity
pal =ig.drawing.colors.ClusterColoringPalette(len(clusters))
i.vs['color'] = pal.get_many(clusters.membership)
ig.plot(i)

g = nx.from_numpy_matrix(net_cus)
h = ig.Graph.from_networkx(g)

y = h.community_fastgreedy()
clusters = y.as_clustering()
clusters.membership
clusters.modularity
pal =ig.drawing.colors.ClusterColoringPalette(len(clusters))
h.vs['color'] = pal.get_many(clusters.membership)
ig.plot(h)
