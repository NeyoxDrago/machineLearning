from matplotlib import style
import matplotlib.pyplot as plt
import numpy as np

style.use('fivethirtyeight')

class Support_vector_machine:
    def __init__(self,visualization=True):
        self.visualization =visualization
        self.colors = {1:'r',-1:'b'}

        if self.visualization:
            self.fig = plt.figure()
            self.AX = self.fig.add_subplot(1,1,1)


    def fit(self,data):
        self.data = data
        opt_dict = {} ## { ||w|| = [w,b] }

        transform = [[2,2],
                     [2,-2],
                     [-2,2],
                     [-2,-2]]

        alldata =[]

        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    alldata.append(feature)

        self.maxl = max(alldata)
        self.minl = min(alldata)
        alldata = None

        #now we have got the max and the min of the data which we will give as a parameter
        #now creating the step sizes for the convex optimization
        step_sizes = [self.maxl*0.1,
                      self.maxl*0.01,
                      self.maxl*0.001]

        #########################################################
        b_range_multiple = 5
        b_multiple =5
        latest_optimum = self.maxl*10
        #########################################################


        for step in step_sizes:
            w= np.array([latest_optimum,latest_optimum])

            optimized = False

            while not optimized:
                for b in np.arange(-1*(self.maxl*b_range_multiple),
                                   1*(self.maxl*b_range_multiple),
                                   step*b_multiple):
                    for trans in transform:
                        w_transformed = w*trans
                        found = True


                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i

                                if not yi*(np.dot(xi,w_transformed)+b) >= 1:
                                    found = False
                                    #print(xi,':',yi*(np.dot(w_transformed,xi)+b))

                        if found:
                            opt_dict[np.linalg.norm(w_transformed)] = [w_transformed,b]
                            
                if w[0] < 0:
                    optimized = True
                    print ('Optimized a Step')
                else:
                    w=w-step

            norm = sorted([n for n in opt_dict])

            opt_choice = opt_dict[norm[0]]

            self.w =opt_choice[0]
            self.b =opt_choice[1]

            latest_optimum = opt_choice[0][0]+ step*2

        for i in self.data:
            for xi in self.data[i]:
                yi=i
                print (xi,":" , yi*(np.dot(self.w,xi)+self.b))
        print("completed fit method")


    def predict(self,features):
        classification  = np.sign(np.dot(np.array(features),self.w)+self.b)

        if classification != 0 and self.visualization:
            self.AX.scatter(features[0],features[1],s=200,marker ="*",c = 'k')

        return classification

    def visualize(self):
        [[self.AX.scatter(x[0],x[1],s=100,color = self.colors[i]) for x in data_dict[i]] for i in data_dict]


        def hyperplane(x,w,b,v):
            return (-w[0]*x-b+v) / w[1]

        datarange = (self.minl*0.9,self.maxl*0.9)

        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        #positive support vector hyperplane

        Psv1 = hyperplane(hyp_x_min,self.w,self.b,1)
        Psv2 = hyperplane(hyp_x_max,self.w,self.b,1)
        self.AX.plot([hyp_x_min,hyp_x_max],[Psv1,Psv2],'k')
        print ('first')

        #negative support vector hyperplane

        nsv1 = hyperplane(hyp_x_min,self.w,self.b,-1)
        nsv2 = hyperplane(hyp_x_max,self.w,self.b,-1)
        self.AX.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2],'k')
        print ('second')

        #discision boundary support vector hyperplane
    
        dsv1 = hyperplane(hyp_x_min,self.w,self.b,0)
        dsv2 = hyperplane(hyp_x_max,self.w,self.b,0)
        self.AX.plot([hyp_x_min,hyp_x_max],[dsv1,dsv2],'k')
        print ('third')

        plt.show()


data = {-1:np.array([[1,2],
                    [3,6],]),
        1:np.array([[2,5],
                    [5,4],])}

data_dict = {-1:np.array([[1,7],
                          [2,8],
                          [3,8],]),
             
             1:np.array([[5,1],
                         [6,-1],
                         [7,3],])}

svm = Support_vector_machine()
svm.fit(data=data_dict)

predict_us = [[0,10],
              [1,3],
              [3,4],
              [3,5],
              [5,5],
              [5,6],
              [6,-5],
              [5,8]]

for p in predict_us:
    svm.predict(p)

svm.visualize()

    
