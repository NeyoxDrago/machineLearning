import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans
import numpy as np

style.use('ggplot')

X = np.array([[2,1],
              [2,2],
              [2,4],
              [6,2],
              [6,6],
              [6,9],
              [9,9],
              [9,8],
              [3,2],
              [10,11]])

classifier = KMeans(n_clusters=3)
classifier.fit(X)

centroids = classifier.cluster_centers_
labels = classifier.labels_

colors = 10*['g','r','b','k','c']

plt.scatter(X[:,0],X[:,1],s=25 )

print (centroids)

plt.scatter(centroids[:,0],centroids[:,1],marker = 'x' , s=150)
plt.show()
