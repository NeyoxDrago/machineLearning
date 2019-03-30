import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from sklearn import preprocessing , cross_validation


df = pd.read_excel('titanic.xls')
df.drop(['body','name'],1,inplace =True)
df.fillna(0,inplace = True)

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

df = handle_non_numeric_data(df)
print (df.head())

X = np.array(df.drop(['sex'],1).astype(float))
X = preprocessing.scale(X)
Y = np.array(df['sex'])

classifier = KMeans(n_clusters = 2)

classifier.fit(X)
correct = 0

for i in range(len(X)):
    predict = np.array(X[i].astype(float))
    predict = predict.reshape(-1,len(predict))
    prediction = classifier.predict(predict)

    if prediction == Y[0]:
        correct+=1


print ("Accuracy over sex Column ::: ",correct/len(X)*100)


X = np.array(df.drop(['survived'],1).astype(float))
X = preprocessing.scale(X)
Y = np.array(df['survived'])

classifier = KMeans(n_clusters = 2)

classifier.fit(X)
correct = 0

for i in range(len(X)):
    predict = np.array(X[i].astype(float))
    predict = predict.reshape(-1,len(predict))
    prediction = classifier.predict(predict)

    if prediction == Y[0]:
        correct+=1


print ("Accuracy over Survived column :: ",correct/len(X)*100)




















