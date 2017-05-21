from SparseWeightVector import SparseWeightVector
from math import exp,log


def make_dataset(text):
    """
    @param text: a list of strings of the form : Le/D chat/N mange/V la/D souris/N ./PONCT
    @return    : an n-gram style dataset
    """
    BOL = '@@@'
    EOL = '$$$'

    dataset = []
    for line in text:
        line         = list([ tuple(w.split('/'))  for w in line.split()])
        tokens       = [BOL] + list([tok for(tok,pos) in line]) + [EOL]
        pos          = list([pos for(tok,pos) in line]) 
        tok_trigrams = list(zip(tokens,tokens[1:],tokens[2:]))
        tok_bigramsL = list(zip(tokens,tokens[1:]))
        tok_bigramsR = list(zip(tokens[1:],tokens))
        
        dataset.extend(zip(pos,zip(tok_trigrams,tok_bigramsL,tok_bigramsR)))
                
    return dataset


class MultiClassPerceptron:

    def __init__(self):
        
        self.model   = SparseWeightVector()
        self.Y       = [] #classes

    def train(self,dataset,step_size=0.1,max_epochs=50):

        self.Y = list(set([y for (y,x) in dataset]))

        for e in range(max_epochs):
            
            loss = 0.0            
            for y,x in dataset:
                ypred = self.tag(x)
                if y != ypred:
                    loss += 1.0
                    delta_ref  = SparseWeightVector.code_phi(x,y)
                    delta_pred = SparseWeightVector.code_phi(x,ypred)
                    self.model += step_size*(delta_ref-delta_pred)
            print ("Loss (#errors) = ",loss)
            if loss == 0.0:
                return
                     
    def predict(self,dataline):
        return list([self.model.dot(dataline,c) for c in self.Y])
    
    def tag(self,dataline):

        scores = self.predict(dataline)
        imax   = scores.index(max(scores)) 
        return self.Y[ imax ]

    def test(self,dataset):

        result = list([ (y == self.tag(x)) for y,x in dataset ])
        return sum(result) / len(result)

        
class MultinomialLogistic:

    def __init__(self):
        
        self.model   = SparseWeightVector()
        self.Y       = [] #classes

        
    def train(self,dataset,step_size=0.1,max_epochs=100):
        #Maximizes log likelihood
        
        self.Y = list(set([y for (y,x) in dataset]))

        for e in range(max_epochs): #Batch gradient ascent 
            
            delta_ref  = SparseWeightVector()
            delta_pred = SparseWeightVector()

            loss = 0.0            
            for y,x in dataset:
                 
                delta_ref += SparseWeightVector.code_phi(x,y)
                
                preds = self.predict(x)
                
                for idx,c in enumerate(self.Y):
                    delta_pred += SparseWeightVector.code_phi(x,c) * preds[idx]

                loss += log(preds[self.Y.index(y)]) 
                    
            self.model += step_size*(delta_ref-delta_pred)
            print('Loss (log likelihood) = ',loss)
        
    def predict(self,dataline):
        
        probs = list([exp(self.model.dot(dataline,c)) for c in self.Y])
        Z = sum(probs)
        probs = list([p/Z for p in probs])
        return probs
    
    def tag(self,dataline):

        probs = self.predict(dataline)
        imax  = probs.index(max(probs)) 
        return self.Y[ imax ]

    def test(self,dataset):

        result = list([ (y == self.tag(x)) for y,x in dataset ])
        return sum(result) / len(result)

    
        
corpus = ['Le/D chat/N mange/V la/D souris/N ./PONCT','La/D souris/N danse/V ./PONCT','Il/Pro la/Pro voit/V dans/P la/D cour/N ./PONCT','Le/D chat/N la/Pro mange/V ./PONCT',"Le/D chat/V la/Pro mange/V"]
D = make_dataset(corpus)
print(D)


maxent = MultinomialLogistic()
maxent.train(D,step_size=1.0)
print(maxent.test(D))

perc = MultiClassPerceptron()
perc.train(D,step_size=1.0)
print(perc.test(D))
