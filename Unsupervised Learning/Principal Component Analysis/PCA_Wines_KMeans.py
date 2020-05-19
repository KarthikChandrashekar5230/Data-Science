import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

wines = pd.read_csv("C:\\Users\\kp\\Pictures\\Assignments\\PCA\\wine.csv",header=0)

# Considering only numerical data
features_wines = wines.iloc[:,1:]
# Normalizing the numerical data
features_wines_normal = scale(features_wines)
pca = PCA(n_components = 10)
pca_wines = pca.fit_transform(features_wines_normal)

# The amount of variance that each PCA explains is
info = pca.explained_variance_ratio_
#Each component is of n-dimensional vector
#pca.components_[0]

# Cumulative variance
cumulative_info = np.cumsum(np.round(info,decimals = 4)*100)

pca_features=pd.DataFrame(data = pca_wines, columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10'])
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
plt.plot(kclusters,TWSS, 'ro-')
plt.xlabel("Number of Clusters")
plt.ylabel("Total Within Sum of Squares")
plt.xticks(kclusters)
plt.title('Screw Plot or Elbow Curve for Determining appropriate Clusters')
plt.show()