from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

x = np.array([1,2,3,5,6,2])
y = np.array([2,2,8,6,8,5])


def best_fit_slope_and_intercept(x,y):
    numerator = ( (mean(x)*mean(y)) - mean(x*y) )
    denominator = (((mean(x))**2) -  mean(x**2)  )
    m= numerator / denominator
    b= mean(y) - m*mean(x)
    return m,b

def squarred_error(a,b):
    return sum((b-a)**2)

def coefficientofdetermination(yoriginal,ys):
    mean_Y = [mean(yoriginal) for y in yoriginal]
    squarred_error_regression = squarred_error(yoriginal,ys)
    squarrred_error_mean_y = squarred_error(yoriginal,mean_Y)

    return 1 - (squarred_error_regression/squarrred_error_mean_y)

m,b = best_fit_slope_and_intercept(x,y)

line = [(m*c) + b for c in x]



print (coefficientofdetermination(y,line))


plt.scatter(x,y)
plt.plot(x,line)
plt.show()

