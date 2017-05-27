from math import exp,log
import sys
import numpy as np
import random


def optimise_univariate(step_size=0.4,max_epochs=30):
    """
    Optimizes f(x) = (x-3)^2 (strictly convex)
    """    
    x = 0
    for e in range(max_epochs):
        x  -= step_size * 2*(x-3) #f'(x) = 2(x-3)
        obj = (x-3)**2
        print(x,obj)
    return x

def optimise_bivariate(step_size=0.4,max_epochs=30):
    """
    Optimizes f(x) = x^2+y^2+2x+8y (strictly convex)
    """    
    (x,y) = (0,0)
    for e in range(max_epochs):
        obj = x**2+y**2+2*x+8*y
        print((x,y),obj)
        x,y = (x-step_size*(2*x+2),y-step_size*(2*y+8))

    obj = x**2+y**2+2*x+8*y
    print((x,y),obj)
    return (x,y)


def make_dataset(istream,add_bias=True):
    """
    Also adds a bias dummy variable
    """
    header = istream.readline()

    dataset = []
    for line in istream:
        fields = line.split()
        if add_bias:
            (y,X) = float(fields[-1]),np.array([1.0,float(fields[0]),float(fields[1])])
        else:
            (y,X) = float(fields[-1]),np.array([float(fields[0]),float(fields[1])])            
        dataset.append((y,X))
    return dataset


class LogisticModel:

    
    def __init__(self):
        self.weights = np.zeros(1)

    def predict(self,x,weights=None):
        """
        Prob of success
        """
        if weights is None:
            weights = self.weights
            
        #this coding attempts to avoid overflows/underflows
        score = np.dot(weights,x)
        if score >= 0:
            return 1/(1+exp(-score))
        else:
            x = exp(score)
            return x/(1+x) 
    
    def train(self,dataset,step_size=1.0,max_epochs=500,epsilon=0.0001):

        D = len(dataset[0][1])
        weights = np.zeros(D)

        objective_history = [0.0]*max_epochs
                 
        for e in range(max_epochs):
            weights -= (step_size/(e+1))*self.batch_gradient(weights,dataset)
            loglik = -self.loglikelihood(weights,dataset)
            #print('LogLikelihood = ', loglik)
            objective_history[e] = loglik
        self.weights = weights
        print(self.weights)
        print('LogLikelihood = ', loglik)
        return objective_history

    def loglikelihood(self,weights,dataset):
        """
        Computes the negative loglikelihood (objective function) on a dataset
        """
        loglik = 0
        for (y,X) in dataset:
            ysucc = self.predict(X,weights)
            loglik += log(ysucc+sys.float_info.epsilon) if y else log(1-ysucc+sys.float_info.epsilon)
        return -loglik

    def batch_gradient(self,weights,dataset):
        """
        Returns a negative gradient (suited for gradient descent)
        """
        D = len(dataset[0][1])
        grad = np.zeros(D)
        for (y,X) in dataset:
            ysucc = self.predict(X,weights)
            grad += X * (y - ysucc)
        return -grad
    
    def trainBFGS(self,dataset):
        """
        Trains the model with the BFGS method
        """
        from scipy.optimize import minimize
        D = len(dataset[0][1])

        res = minimize(self.loglikelihood,np.zeros(D),args=(dataset,),jac=self.batch_gradient,method="BFGS")
        self.weights = np.array(res.x)
        print('Estimate:',res.x)
        print('LogLikelihood:',res.fun)
        
    def trainSGD(self,dataset,step_size=1.0,max_epochs=500,epsilon=0.0001):
        """
        Trains the model with the SGD method
        """
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
                loglik += log(ysucc+sys.float_info.epsilon) if y else log(1-ysucc+sys.float_info.epsilon)
            #print('LogLikelihood = ', loglik)
            objective_history[e] = loglik
        print(self.weights)
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
    optimise_univariate(step_size=0.4)
    optimise_bivariate(step_size=0.1)

    #istream = open('data/logistic.dat')
    #D = make_dataset(istream,add_bias=True)
    #istream.close()
    #m = LogisticModel()
    #m.plot_bivariate_loglikelihood(D)
    #sys.exit(0)
    #simple logistic 
    istream = open('data/logistic.dat')
    D = make_dataset(istream,add_bias=True)
    istream.close()
    m = LogisticModel()
    objSGD   = m.trainSGD(D,step_size=5)
    print('Internal test accurracy :',m.test(D))
    objBatch = m.train(D,step_size=5.0)
    print('Internal test accurracy :',m.test(D))
    objBFGS = m.trainBFGS(D)
    print('Internal test accurracy :',m.test(D))
    
    #plotting(uncomment if pandas and pyplot are installed)    
    #import pandas as pd
    #import matplotlib.pyplot as plt
    #df = pd.DataFrame({'Batch':objBatch,'SGD':objSGD})
    #df.plot()
    #plt.show()
