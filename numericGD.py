from math import exp,log
import sys
import numpy as np
import random


def optimise_basic(step_size=0.4,max_epochs=30):
    """
    Optimizes f(x) = (x-3)^2 (strictly convex)
    """    
    x = 0
    for e in range(max_epochs):
        x  -= step_size * 2*(x-3) #f'(x) = 2(x-3)
        obj = (x-3)**2
        print(x,obj)

 

def make_dataset(istream):
    """
    Adds a bias dummy variable
    """
    header = istream.readline()

    dataset = []
    for line in istream:
        fields = line.split()
        (y,X) = float(fields[-1]),np.array([1.0,float(fields[0]),float(fields[1])])
        dataset.append((y,X))
    return dataset


class LogisticModel:

    def __init__(self):
        self.weights = np.zeros(1)

    def predict(self,x):
        """
        Prob of success
        """
        #this code attempts to avoid overflows/underflows
        score = np.dot(self.weights,x)
        if score >= 0:
            return 1/(1+exp(-score))
        else:
            x = exp(score)
            return x/(1+x) 
    
    def train(self,dataset,step_size=1.0,max_epochs=500,epsilon=0.0001):

        D = len(dataset[0][1])
        self.weights = np.zeros(D)

        objective_history = [0.0]*max_epochs
        
        for e in range(max_epochs):
            loglik = 0.0
            grad = np.zeros(D)
            for (y,X) in dataset:
                ysucc = self.predict(X)
                grad += X * (y - ysucc)
                loglik += log(ysucc+sys.float_info.epsilon) if y else log(1+sys.float_info.epsilon-ysucc)
            #self.weights += step_size*grad
            self.weights += (step_size/(e+1))*grad
            print('LogLikelihood = ', loglik)
            objective_history[e] = loglik
        return objective_history
            
    def trainSGD(self,dataset,step_size=1.0,max_epochs=500,epsilon=0.0001):

        D = len(dataset[0][1])
        self.weights = np.zeros(D)
        objective_history = [0.0]*max_epochs
        for e in range(max_epochs):
            loglik = 0.0
            random.shuffle(dataset)
            for (y,X) in dataset:
                ysucc = self.predict(X)
                grad = X* (y - ysucc)
                self.weights += (step_size/(e+1))*grad
                #self.weights += step_size*grad
                loglik += log(ysucc+sys.float_info.epsilon) if y else log(1+sys.float_info.epsilon-ysucc)
            print('LogLikelihood = ', loglik)
            objective_history[e] = loglik
        return objective_history

            
    def test(self,dataset):
        c = 0
        for (y,X) in dataset:
            yprob = self.predict(X)
            if (yprob >= 0.5 and y == 1) or (yprob < 0.5 and y == 0):
                c+=1
        return c/len(dataset)


if __name__ == '__main__':
    
    #gradient descent : try with step_size: 0.1,0.4; (0.01,2.0)
    optimise_basic(step_size=3)


    #simple logistic 
    istream = open('data/logistic.dat')
    D = make_dataset(istream)
    istream.close()
    m = LogisticModel()
    objSGD   = m.trainSGD(D,step_size=0.03)
    objBatch = m.train(D,step_size=0.03)
    print('Internal test accurracy :',m.test(D))

    #plotting(uncomment if pandas and pyplot are installed)    
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.DataFrame({'Batch':objBatch,'SGD':objSGD})
    df.plot()
    plt.show()
