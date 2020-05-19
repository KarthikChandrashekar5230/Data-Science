import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch # for creating dendrogram
from sklearn.cluster import AgglomerativeClustering

initial_airlines_df = pd.read_csv("C:\\Users\\kp\\Pictures\\Assignments\\Clustering\\EastWestAirlines.csv",header=0)
airlines=initial_airlines_df.drop(['ID#'],axis=1)
# Normalizing the numerical data
features_airlines_normal = pd.DataFrame(data = scale(airlines), columns = ['Balance', 'Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles','Bonus_miles', 'Bonus_trans', 'Flight_miles_12mo', 'Flight_trans_12', 'Days_since_enroll', 'Award'])

###### screw plot or elbow curve ############
kclusters = list(range(2,20))
TWSS = [] # variable for storing total within sum of squares for each kmeans

for i in kclusters:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(features_airlines_normal)
    WSS = [] # variable for storing within sum of squares for each cluster
    for j in range(i):
        WSS.append(sum(cdist(features_airlines_normal.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,features_airlines_normal.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))

plt.figure(figsize=(20,15))
plt.plot(kclusters,TWSS, 'ro-')
plt.xlabel("Number of Clusters")
plt.ylabel("Total Within Sum of Squares")
plt.xticks(kclusters)
plt.title('Screw Plot or Elbow Curve for Determining appropriate Clusters')
plt.show()

# Selecting 8 clusters from the above scree plot which is the optimum number of clusters
kmeans_model=KMeans(n_clusters=8)
kmeans_model.fit(features_airlines_normal)

# getting the labels of clusters assigned to each row
#kmeans_model.labels_

class_labels=pd.Series(kmeans_model.labels_)  # converting numpy array into pandas series object
features_airlines_normal['Clusters']=class_labels  # creating a  new column and assigning it to new column

#########################################################Hierarchical Clustering-Dendogram-Agglomerative Clustering##############################################

#features_airlines_hierarchical=features_airlines_normal.iloc[:, 0:11].values
features_airlines_hierarchical=np.array(features_airlines_normal.drop(["Clusters"],axis=1))

dist = linkage(features_airlines_hierarchical, method="complete",metric="euclidean")

plt.figure(figsize=(20, 20))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Observations Present in the Dataset')
plt.ylabel('Distance Values')
sch.dendrogram(dist,leaf_rotation=0.0,leaf_font_size=8.0)
plt.show()

# Now applying AgglomerativeClustering choosing 6 as clusters from the dendrogram
hierarchical_complete = AgglomerativeClustering(n_clusters=6, linkage='complete',affinity = "euclidean").fit(features_airlines_hierarchical)

df_features_airlines_hierarchical= features_airlines_normal.drop(["Clusters"],axis=1)

df_features_airlines_hierarchical['Clusters']=pd.Series(hierarchical_complete.labels_)

# getting aggregate mean of each cluster
#df_features_airlines_hierarchical.iloc[:,0:11].groupby(df_features_airlines_hierarchical.Clusters).median()

# creating a csv file
df_features_airlines_hierarchical.to_csv("Clustered_Airlines.csv",encoding="utf-8")