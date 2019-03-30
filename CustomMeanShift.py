import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
style.use('ggplot')
from sklearn.datasets.samples_generator import make_blobs


X , y = make_blobs(n_samples = 100 , centers = 3 , n_features = 2)

##X = np.array([[1,2],
##              [2,1],
##              [1,3],
##              [9,9],
##              [8,9],
##              [10,7],
##              [5,5],
##              [4,6],
##              [6,7],])


colors = 10*['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

class Mean_Shift:
    def __init__(self,radius =None , radius_step = 25):
        self.radius = radius
        self.radius_step = radius_step

    def fit(self,data):

        if self.radius == None:
            data_centroid = np.average(data,axis =0)
            all_data_norm = np.linalg.norm(data_centroid)
            self.radius = all_data_norm/self.radius_step

        centroids ={}
        print (self.radius)

        for i in range(len(data)):
            centroids[i] =data[i]

        weights = [i for i in range(self.radius_step)[::-1]]

        while True:
            new_centroids = []

            for i in centroids:
                in_radius = []
                centroid = centroids[i]

                for featureset in data:
##                    if np.linalg.norm(featureset - centroid) < self.radius:
##                        in_radius.append(featureset)
                    distance = np.linalg.norm(featureset- centroid)

                    if distance ==0:
                        distsnce = 0.0000000001

                    weight_index = int(distance/self.radius)
                    #print (weight_index)
                    if weight_index > self.radius_step - 1:
                        weight_index = self.radius_step - 1

                    to_add = (weights[weight_index]**2)*[featureset]
                    in_radius += to_add

                new_centroid = np.average(in_radius,axis=0)
                new_centroids.append(tuple(new_centroid))

            uniques = sorted(list(set(new_centroids)))

            to_pop = []
            for i in uniques:
                for ii in uniques:
                    if i == ii :
                        pass
                    elif np.linalg.norm(np.array(i) - np.array(ii)) <= self.radius:
                        to_pop.append(ii)
                        break
            for i in to_pop:
                try:
                    uniques.remove(i)
                except:
                    pass
                
            previous_centroids = dict(centroids)

            centroids ={}

            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])

            optimized =True

            for i in centroids:
                if not np.array_equal(centroids[i],previous_centroids[i]):
                    optimized = False
                if not optimized:
                    break

            if optimized:
                break

        self.centroids = centroids
        self.classifications = {}

        for i in range(len(self.centroids)):
            self.classifications[i] = []
            
        for featureset in data:
            #compare distance to either centroid
            distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
            #print(distances)
            classification = (distances.index(min(distances)))

            # featureset that belongs to that cluster
            self.classifications[classification].append(featureset)

    def predict(self,data):
        #compare distance to either centroid
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = (distances.index(min(distances)))
        return classification


classifier = Mean_Shift()
classifier.fit(X)
centroids = classifier.centroids
print (centroids)
print (len(centroids))

for classification in classifier.classifications:
    color = colors[classification]
    for feature in classifier.classifications[classification]:
        plt.scatter(feature[0],feature[1],marker= '+' , s = 50 , c = color)
for c in centroids:
    color=colors[c]
    plt.scatter(centroids[c][0],centroids[c][1],marker = "*", s = 100 ,c=color)

plt.show()
