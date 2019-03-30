import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import pandas as pd
from sklearn.cluster import MeanShift
import numpy as np
from sklearn import preprocessing , cross_validation


df = pd.read_excel('titanic.xls')
copydf = pd.DataFrame.copy(df)

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
#print (df.head())

X = np.array(df.drop(['sex'],1).astype(float))
X = preprocessing.scale(X)
Y = np.array(df['sex'])

classifier = MeanShift()

classifier.fit(X)

labels = classifier.labels_
print (labels)
cluster_centers = classifier.cluster_centers_
print (cluster_centers)
n_clusters_ = len(np.unique(labels))
print (n_clusters_)
print (set(labels))

df['cluster_group'] = np.nan

for i in range(len(X)):
    df['cluster_group'].iloc[i] = labels[i]
    
survival_rates ={}

for i in range(n_clusters_):
    temp_df = df[ (df['cluster_group'] == float(i)) ]
    survival_cluster = temp_df[ (temp_df['survived'] == 1 ) ]
    survival_rate = len(survival_cluster)/ len(temp_df)
    survival_rates[i] = survival_rate


print (survival_rates)






