import numpy as np
from sklearn import preprocessing , cross_validation , neighbors
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-99999,inplace = True)
df.drop(['id'],1,inplace = True)
x= np.array(df.drop(['Class'],1))
##x = preprocessing.scale(x)
##applied preprocessing just to check the change in output
y = np.array(df['Class'])

x_train, x_test , y_train , y_test = cross_validation.train_test_split(x,y,test_size=0.5)

classifier = neighbors.KNeighborsClassifier()
classifier.fit(x_train,y_train)

accuracy = classifier.score(x_test,y_test)

print (accuracy*100)

example_predict = np.array([[1,1,1,2,1,1,1,5,1]])

prediction = classifier.predict(example_predict)
print (prediction)
