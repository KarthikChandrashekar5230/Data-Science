import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch # for creating dendrogram
from sklearn.cluster import AgglomerativeClustering

wines = pd.read_csv("C:\\Users\\kp\\Pictures\\Assignments\\PCA\\wine.csv",header=0)

# Considering only numerical data
features_wines = wines.iloc[:,1:]
# Normalizing the numerical data
features_wines_normal = scale(features_wines)

#Taking First Three Principal Components for our further problem solving
pca = PCA(n_components = 3)
pca_wines = pca.fit_transform(features_wines_normal)

print("PCA Component Values: \n",pca.components_)

# The amount of variance that each PCA explains is
info = pca.explained_variance_ratio_
print("Information retained after PCA in terms of Component-Wise: ",info)
#Each component is of n-dimensional vector
#pca.components_[0]

# Cumulative variance
cumulative_info = np.cumsum(np.round(info,decimals = 4)*100)
print("Total Information retained after PCA: ",cumulative_info)

#Transformed PCA Component Values
pca_features=pd.DataFrame(data = pca_wines, columns = ['PC1','PC2','PC3'])
pca_df = pd.concat([pca_features, wines[['Type']]], axis = 1)

# Variance plot for PCA components obtained
plt.plot(cumulative_info,color="red")
plt.show()

# Plot between PCA1 and PCA2
fig = plt.figure(figsize = (20,15))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal_Component_1')
ax.set_ylabel('Principal_Component_2')
ax.set_title('Two Dimensional Data Visualization')
types = [1,2,3]
colors = ['r', 'g', 'b']
for type,color in zip(types,colors):
    indicesToKeep = pca_df['Type'] == type
    ax.scatter(pca_df.loc[indicesToKeep, 'PC1'],pca_df.loc[indicesToKeep, 'PC2'],c = color,s = 50)
ax.legend(types)
ax.grid()
plt.show()


###### screw plot or elbow curve ############
kclusters = list(range(2,11))
TWSS = [] # variable for storing total within sum of squares for each kmeans

for i in kclusters:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(pca_features)
    WSS = [] # variable for storing within sum of squares for each cluster
    for j in range(i):
        WSS.append(sum(cdist(pca_features.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,pca_features.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))


plt.figure(figsize=(20,15))
plt.plot(kclusters,TWSS, 'ro-');
plt.xlabel("Number of Clusters");
plt.ylabel("Total Within Sum of Squares");
plt.xticks(kclusters)
plt.title('Screw Plot or Elbow Curve for Determining appropriate Clusters')

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters
kmeans_model=KMeans(n_clusters=3)
kmeans_model.fit(pca_features)

# getting the labels of clusters assigned to each row
#kmeans_model.labels_

class_labels=pd.Series(kmeans_model.labels_)  # converting numpy array into pandas series object
pca_df['Predictions']=class_labels  # creating a  new column and assigning it to new column


#########################################################Hierarchical Clustering-Dendogram-Agglomerative Clustering##############################################

#Dendogram Plot using Complete Linkage as Distance Metric
dist_complete = linkage(pca_features, method="complete",metric="euclidean",optimal_ordering=True)

plt.figure(figsize=(20, 20))
plt.title('Hierarchical Clustering Dendrogram using Complete Linkage Metric')
plt.xlabel('Observations Present in the Dataset')
plt.ylabel('Distance Values')
sch.dendrogram(dist_complete,leaf_rotation=0.0,leaf_font_size=8.0)
plt.show()

#Dendogram Plot using Average Linkage as Distance Metric
dist_average = linkage(pca_features, method="average",metric="euclidean",optimal_ordering=True)

plt.figure(figsize=(20, 20))
plt.title('Hierarchical Clustering Dendrogram using average Linkage Metric')
plt.xlabel('Observations Present in the Dataset')
plt.ylabel('Distance Values')
sch.dendrogram(dist_average,leaf_rotation=0.0,leaf_font_size=8.0)
plt.show()

#It is evident from the Dendogram Plots that the Average Linkage Distance Metric will yield us optimal Number of Clusters
#Matching with the Number of Clusters provided in the Original Dataset rather than using other Distance Linkage metrics like
#Complete,single

# Applying AgglomerativeClustering by choosing 3 as clusters from the dendrogram
hierarchical_average = AgglomerativeClustering(n_clusters=3, linkage='average',affinity = "euclidean").fit(pca_features)

pca_df_hierarchical= pca_features

pca_df_hierarchical['Predictions']=pd.Series(hierarchical_average.labels_)

# creating a csv file
pca_df_hierarchical.to_csv("Wines_HierarchicalClustering.csv",encoding="utf-8")