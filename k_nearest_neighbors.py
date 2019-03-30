import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import pandas as pd
import random

style.use('ggplot')

dataset = {'g':[[1,2],[5,2],[6,1],[2,5]],'b':[[6,7],[5,4],[6,4],[8,5]]}
exampleset = [8,9]

def k_nearest_neighbors(Set,predictionset,k=3):

    distanceset=[]
    for group in Set:
        for i in Set[group]:
            euclidean_distance = np.sqrt((np.sum(np.array(i) - np.array(predictionset))**2))
            ##euclidean_distance = np.linalg.norm(np.array(i) - np.array(predictionset))
            distanceset.append([euclidean_distance,group])



    votes = [i[1] for i in sorted(distanceset)[:k]]
    result = Counter(votes).most_common(1)[0][0]


    return result


print (k_nearest_neighbors(dataset,exampleset,k=2))

##
##[[plt.scatter(i[0],i[1] , color = k , s = 100) for i in dataset[k]] for k in dataset]
##plt.scatter(exampleset[0],exampleset[1],color = 'r' , s=100)
##plt.show()


file = pd.read_csv('breast-cancer-wisconsin.data.txt')
file.replace('?',-99999,inplace = True)
file.drop(['id'],1,inplace = True)
file = file.astype(float).values.tolist()
accuracies =[]
for i in range(10):
    random.shuffle(file)

    test_size = 0.4
    train_data = file[:-int(test_size*(len(file)))]
    test_data =  file[-int(test_size*(len(file))):]
    train_set = {2:[],4:[]}
    test_set = {2:[],4:[]}



    for i in train_data:
        train_set[i[-1]].append(i[:-1])

    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    correct = 0
    total =0


    for group in test_set:
        for feature in test_set[group]:
            vote = k_nearest_neighbors(train_set,feature,k=5)
            if group == vote:
                correct+=1
            total += 1


    #print ('Accuracy ::' , correct/total *100)
    accuracies.append(correct/total *100)



print ("Average Accuracy :: ", sum(accuracies)/len(accuracies))
