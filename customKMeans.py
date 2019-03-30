import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from sklearn import preprocessing , cross_validation
import pandas as pd
import numpy as np



x = np.array([[1,2],
              [2,2],
              [2,3],
              [7,5],
              [8,7],
              [1,5],
              [8,10],
              [9,10]])

colors = 10*['g','r','b','k','c']

class K_Means:
    def __init__(self,k=5,tol=0.000001,max_iter = 100):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol

    def fit(self,data):
        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications={}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                classification =distances.index(min(distances))
                self.classifications[classification].append(featureset)

            previous_centroids =dict(self.centroids)
            #print (self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis =0)

            #print (self.centroids)
            optimized =True
            
            #print(self.classifications)
            for c in self.centroids:
                original_centroid = previous_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid)/original_centroid *100 ) > self.tol:
##                    print (np.sum((current_centroid - original_centroid)/original_centroid *100 ))
                    optimized = False

            if optimized:
                break

    def predict(self,data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification =distances.index(min(distances))
        return classification


def handle_non_numeric_data(df):
    columns = df.columns.values
    text_digit= {}

    for column in columns :
        def convert_to_int(val):
            return text_digit[val]

        if df[column].dtype != np.int64 and df[column].dtype !=np.float64:

            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x=0

            for unique in unique_elements:
                if unique not in text_digit:
                    text_digit[unique] = x
                    x+=1


            df[column] = list(map(convert_to_int,df[column]))
            
    return df



df = pd.read_excel('titanic.xls')
df = handle_non_numeric_data(df)
df.drop(['body','name','home.dest'],1,inplace =True)
df.fillna(0,inplace = True)

X = np.array(df.drop(['survived'],1).astype(float))
X = preprocessing.scale(X)
Y = np.array(df['survived'])

#X_train , X_test ,Y_train , Y_test = cross_validation.train_test_split(X,Y,test_size = 0.8)

cl = K_Means()
cl.fit(x)

correct = 0

##for i in range(len(X)):
##    predict = np.array(X[i].astype(float))
##    predict = predict.reshape(-1,len(predict))
##    prediction = cl.predict(predict)
##
##    if prediction == Y[0]:
##        correct+=1
##
##
##print ("Accuracy over Survived column :: ",correct/len(X)*100)
 
                

for centroid in cl.centroids:
    plt.scatter(cl.centroids[centroid][0],cl.centroids[centroid][1],
                marker = "o", color = colors[centroid], s=10 , linewidths = 5)


for classification in cl.classifications:
    color =colors[classification]
    for feature in cl.classifications[classification]:
        plt.scatter(feature[0],feature[1],marker = "*" , s=150 , color = color)


predict_me = np.array([[6,7],
                       [5,2],
                       [3,9],
                       [8,5],
                       [10,10],
                       [1,7],
                       [6,3],
                       [8,2]])

for i in predict_me:
    classification = cl.predict(i)
    plt.scatter(i[0],i[1],color = colors[classification] , s = 150 , marker = 'P')

plt.show()




