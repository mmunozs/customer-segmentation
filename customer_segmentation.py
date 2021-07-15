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
#customers.index = customers['CustomerID']

# Data exploration
df.head()
df.info()
df.nunique()
df.isnull().sum()
(df['Price']<0).sum()
(df['Quantity']<0).sum()
df.boxplot(column=['Price','Quantity'])
# df.boxplot(column=['Price','Quantity'], by='Country')

cols = ['Price','Quantity']
Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)
IQR = Q3 - Q1
print (IQR)
pos_out = ((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1).sum()
df_out = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
# df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]

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

Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)
IQR = Q3 - Q1
print (IQR)
pos_out = ((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1).sum()
df_out = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
df_out.boxplot(column=['Price','Quantity'])

df.describe().T
df_out.describe().T
df.to_csv("online_retail_II_cleaned.csv")

'''
df[(df['StockCode']== 'ADJUST')][['Quantity','Description']] # remove
df[(df['StockCode']== 'ADJUST2')][['Quantity','Description']] # remove
df[(df['StockCode']=='BANK CHARGES')][['Quantity','Description']] # remove
df[(df['StockCode']=='C2')][['Quantity','Description','Price']] # check if quantity is more than 1, 253 rows
df[(df['StockCode']=='D')][['Quantity','Description']] # remove
df[(df['StockCode']=='DOT')][['Quantity','Description']] # remove
df[(df['StockCode']=='M')][['Quantity','Description','Price']] # remove possibly, 716 rows
df[(df['StockCode']=='PADS')][['Quantity','Description','Price']] # remove
df[(df['StockCode']=='POST')][['Quantity','Description','Price']] # not sure what it is, 1838 rows
df[(df['StockCode']=='SP1002')][['Quantity','Description','Price']]# it is fine
df[(df['StockCode']=='TEST001')][['Quantity','Description']] # remove
df[(df['StockCode']=='TEST002')][['Quantity','Description']] # remove
'''
# removing StockCode
df = df[~((df['StockCode']== 'ADJUST')|(df['StockCode']== 'ADJUST2')|(df['StockCode']=='BANK CHARGES')|(df['StockCode']=='D')|(df['StockCode']=='DOT')|(df['StockCode']=='PADS')|(df['StockCode']=='TEST001')|(df['StockCode']=='TEST002'))]
df = df[~(df['StockCode']== 'M')]
df = df[~(df['StockCode']== 'POST')]

df.plot.scatter(x='Price',y='Quantity')
df = df[df['Price']<50]
df = df[df['Quantity']<7500]


### RFM MODEL
df["TotalPrice"]=df["Quantity"]*df["Price"]

# Recency
df["InvoiceDate"].max()
today_date = dt.datetime(2011, 12, 11)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

df = df[df['Country']=='United Kingdom']
#Data Frame for Customer ID with the variables of InvoiceDate (will be changed to Rececny in next step), Invoice number (will be changed to "Frequency") and Total sum of transactions (will be changed to "Monetary")
rfm = df.groupby('Customer ID').agg({'InvoiceDate': lambda date: (today_date - date.max()).days,'Invoice': lambda num: num.nunique(),'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

# We reassing the column names in previous cell. 
# Recency: We subtracted today's date from the last transation date to calculate recency in weeks
# Frequency: We counted invoiced per each customer. Since each invoice refers to one transaction we can use them as frequency. 
# Monetary: We sum up all transations per each customer as their monetary value.

rfm.columns = ['Recency', 'Frequency', 'Monetary']
# To avoid miscalculation we run a acheck querry to eleminate 0 values in Monetary and Frequency
rfm = rfm[(rfm["Monetary"]) > 0 & (rfm["Frequency"] > 0)]
rfm['Recency'] = rfm['Recency']/7
rfm.describe().T
rfmv = rfm.copy()
rfmv['AverageValue'] = rfmv['Monetary']/rfmv['Frequency']


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

# Scaling
rfm1 = rfm.copy()
scaler = preprocessing.MinMaxScaler()
rfm1[:] = scaler.fit_transform(rfm1)
rfmplot2d(rfm1)
rfmplot3d(rfm1)

# Standardization
rfm2 = rfm.copy()
z_scaler = preprocessing.StandardScaler()
rfm2[:] = z_scaler.fit_transform(rfm2)
rfmplot2d(rfm2)
rfmplot3d(rfm2)

# Normalising
rfm3 = rfm.copy()
L2_norm = preprocessing.Normalizer()
rfm3[:] = L2_norm.fit_transform(rfm3)
rfmplot2d(rfm3)
rfmplot3d(rfm3)

# Log transformation
rfm_log = rfm.copy()
rfm_log = rfm_log.applymap(lambda x: np.log(x+1))
rfmplot2d(rfm_log)
rfmplot3d(rfm_log)



# rfm.plot.scatter('Recency', 'Frequency', c = df['Country'].map(type_colors_map))
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
X3 = rfm3[['Recency', 'Frequency', 'Monetary']] 
X_log = np.array(rfm_log[['Recency', 'Frequency', 'Monetary']])
# K-means
kmeans3 = KMeans(n_clusters=3)
kmeans4 = KMeans(n_clusters=4)
kmeans5 = KMeans(n_clusters=5)

y = kmeans3.fit_predict(X_log)
rfmplotclusters(rfm_log, y)
y = kmeans4.fit_predict(X_log)
rfmplotclusters(rfm_log, y)
y = kmeans5.fit_predict(X_log)
rfmplotclusters(rfm_log, y)


# DBSCAN
y = DBSCAN(eps=0.005, min_samples=3).fit_predict(X_log)
rfmplotclusters(rfm_log, y)

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
gm4.means_

gm5 = GaussianMixture(n_components = 5, covariance_type = 'full', random_state=0).fit_predict(X_log)
rfmplotclusters(rfm_log, gm5)

gmm10 = GaussianMixture(n_components=10, covariance_type="full", random_state=0)
gmm10.fit_predict(arr)

np.unique(gm4)
np.unique(gm5)

rfm['Average value'] = rfm['Monetary']/rfm['Frequency']
rfm['Cluster'] = gm4
profile = rfm.groupby('Cluster').mean()
profile['Numb_cust'] = rfm.groupby('Cluster')['Monetary'].count()
### Customer - Product Matrix
df_net = df.groupby(['Customer ID', 'StockCode'])['Quantity'].sum().unstack().fillna(0)

df_net_bi = df.groupby(['Customer ID', 'StockCode'])['Quantity'].sum().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)

df_array = sparse.csr_matrix(np.array(df_net_bi))
df_array_T = sparse.csr_matrix(np.array(df_net_bi.T))

df_net.sum().plot.box()
df_net.sum(axis=1).plot.box()

net_cus = df_array@df_array_T
net_pro = df_array_T@df_array

net_cus = net_cus.toarray()
net_pro = net_pro.toarray()
np.fill_diagonal(net_cus, 0)


net_cus = net_cus[~a]
net_cus = net_cus.T[~a]

g = nx.from_numpy_matrix(net_cus)
h = ig.Graph.from_networkx(g)

prueba = net_cus[:100,:100]
i = ig.Graph.from_networkx(nx.from_numpy_matrix(prueba))

x = i.community_fastgreedy()
clusters = x.as_clustering()
clusters.membership
clusters.modularity
pal3 =ig.drawing.colors.ClusterColoringPalette(len(clusters))
i.vs['color'] = pal3.get_many(clusters.membership)
ig.plot(i)

y = h.community_fastgreedy()
clusters = y.as_clustering()
clusters.membership
clusters.modularity
pal3 =ig.drawing.colors.ClusterColoringPalette(len(clusters))
h.vs['color'] = pal3.get_many(clusters.membership)
ig.plot(h)


g1 = igraph.Graph.Weighted_Adjacency(net_cus)

v = g1.igraph_community_to_membership()

def pipeline(n, input_df):
    
    cocluster = SpectralCoclustering(n_clusters = n)
    cocluster.fit(input_df.values)
    cocluster_fit_data = input_df.values[argsort(cocluster.row_labels_)]
    cocluster_fit_data = cocluster_fit_data[:, argsort(cocluster.column_labels_)]
    
    bicluster = SpectralBiclustering(n_clusters = n)
    bicluster.fit(input_df.values)
    bicluster_fit_data = input_df.values[argsort(bicluster.row_labels_)]
    bicluster_fit_data = bicluster_fit_data[:, argsort(bicluster.column_labels_)]

    figure(figsize=(16,16))
    #suptitle("Mushrooms\n" + "n_clusters = " + str(n),fontsize=32, fontweight='bold')
    
    left_plot = subplot(121)
    ax = imshow(bicluster_fit_data, aspect='auto', cmap="bone")
    #xticks(range(0,len(input_df.columns)), list(input_df.columns[bicluster.column_labels_]),rotation='vertical')
    xlabel("Biclustering")
    
    right_plot = subplot(122, sharey=left_plot)
    ax = imshow(cocluster_fit_data, aspect='auto', cmap="bone")
    labels = list(input_df.columns[cocluster.column_labels_])
    #xticks(range(0,len(input_df.columns)), labels, rotation='vertical')
    xlabel("Coclustering")
    
    tight_layout()
    
    show()

pipeline(6,df_net_bi)
pipeline(6,best_prod)

model = SpectralBiclustering(n_clusters=5, random_state=0)
model.fit(best_prod)
# score = consensus_score(model.biclusters_,(rows[:, row_idx], columns[:, col_idx]))

# print("consensus score: {:.3f}".format(score))

fit_data = best_prod.values[np.argsort(model.row_labels_)]
fit_data = fit_data[:, np.argsort(model.column_labels_)]

plt.matshow(fit_data, cmap=plt.cm.Blues)
plt.title("After biclustering; rearranged to show biclusters")

plt.show()

g = Graph()
print(g)
g.add_vertices(3)
g.add_edges([(0,1), (1,2)])
g.add_edges([(2, 0)])
g.add_vertices(3)
g.add_edges([(2, 3), (3, 4), (4, 5), (5, 3)])
print(g)
g.get_eid(2, 3)
g.delete_edges(3)
summary(g)
g = Graph.Tree(127, 2)
summary(g)
g = Graph.GRG(100, 0.2)
summary(g)

a = pd.DataFrame([['a',0,1,1],['b',1,0,0],['c',0,1,0]],columns=['index','a','b','c'])
a.set_index('index')
b=np.array([[0,1,1],[1,0,0],[0,1,0]])
g = Graph.Adjacency(b)

d= np.array([[0,1,1,0,0,],[0,0,1,1,0],[0,1,0,0,0,],[0,0,1,1,0],[0,1,1,0,0]])
e = sparse.csr_matrix(d)@sparse.csr_matrix(d.T)
e.toarray()
