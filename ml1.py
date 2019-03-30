import pandas as p
import quandl as q
import math , datetime
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing , cross_validation
from matplotlib import style
import matplotlib.pyplot as pylt
import pickle


style.use('ggplot')
q.ApiConfig.api_key = 'fS55xSRy_yKebyurKKNc'
df = q.get('WIKI/GOOGL')
#df = df[['Open','High','Low','Close','Volume','Split Ratio','Adj. Open','Adj. Close','Adj. High','Adj. Low','Adj. Volume']]

df['hl_percent'] = (df['Adj. High']- df['Adj. Close'])/ df['Adj. Close']*100
df= df[['Adj. High', 'Adj. Close', 'hl_percent']]

prediction_close = 'Adj. Close'
df.fillna(100,inplace = True)
print (len(df))
predict = int (math.ceil(0.1*len(df)))
##predict = 100
print (predict)

df['prediction'] = df[prediction_close].shift(-predict)

##Above is the code to simply just shift the cursor for a position to another

X = np.array(df.drop(['prediction'],1))
X = preprocessing.scale(X)
X = X[:-predict]
X_lately = X[-predict:]
df.dropna(inplace=True)
y = np.array(df['prediction'])

x_train, x_test , y_train , y_test = cross_validation.train_test_split(X,y,test_size=0.3)

classifier = LinearRegression()
classifier.fit(x_train,y_train)

with open('stocks.pickle','wb') as file:
    pickle.dump(classifier,file) 


accuracy = classifier.score(x_test , y_test)

print (df)
print (X)
print (y)
print (x_train)
print (x_test)
print (y_train)
print (y_test)
print (accuracy*100)


set_prediction = classifier.predict(X_lately)

print (set_prediction)

df['Final_Prediction_Close'] = np.nan


last_date = df.iloc[-1].name
last_datetime= last_date.timestamp()
one_day= 86400
next_datetime = last_datetime + one_day


for i in set_prediction:
    next_date = datetime.datetime.fromtimestamp(next_datetime)
    next_datetime += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Final_Prediction_Close'].plot()
pylt.legend(loc=2)
pylt.xlabel('date')
pylt.ylabel('price')


print (df)
pylt.show()




























