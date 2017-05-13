from SparseWeightVector import SparseWeightVector
from math import inf
import numpy as np


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
        dataset.append((pos,list(zip(tok_trigrams,tok_bigramsL,tok_bigramsR))))
                    
    return dataset


class StructuredPerceptron:

    def __init__(self):
        self.model      = SparseWeightVector()
        self.Y          = []    #classes
        self.source_tag = "@@@"

    def tag(self,sentence):
        """
        Viterbi + backtrace
        """
        N = len(sentence)
        K = len(self.Y)
        
        viterbi = np.zeros((N,K))
        history = np.zeros((N,K))
        
        #init
        for j in range(K):
            viterbi[0,j] = self.score(self.source_tag,self.Y[j],sentence[0])

        #Recurrence
        for i in range(1,N):
            for j in range(K):
                smax,amax = -inf,-inf
                for pred in range(K):
                    score =  viterbi[i-1,pred] + self.score(self.Y[pred],self.Y[j],sentence[i])
                    if score > smax:
                        smax,amax = score,pred
                viterbi[i,j],history[i,j] = smax,amax
    
        #End state
        smax,amax = -inf,-inf
        for pred in range(K):
            score = viterbi[N-1,pred]
            if score > smax:
                smax,amax = score,pred
                
        #Backtrace
        rev_tag_sequence = [] 
        for i in range(N-1,-1,-1):
            rev_tag_sequence.append(self.Y[ amax ])
            amax = int(history[i,amax])
            
        return list(reversed(rev_tag_sequence))

    def score(self,y_pred,y,word_repr):
        """
        Scores a structured perceptron clique
        @param y_pred : prev tag
        @param y  : current tag
        @word_repr : a word data representation (a list of hashable symbols)
        @return a real value
        """
        return self.model.dot(word_repr,(y_pred,y))

    
    def train(self,dataset,step_size=1.0,max_epochs=100):
        """
        @param dataset: a list of couples (y_tags,x_words)
        """
        self.Y = list(set([y for (ytags,xwords) in dataset for y in ytags]))

        N = len(dataset)
        for e in range(max_epochs):
            
            loss = 0.0
            for ytags,xwords in dataset:

                ypreds = self.tag(xwords)
                
                if ypreds != ytags:
                    loss += 1.0 

                    ytags_bigrams = list(zip([self.source_tag]+ytags,ytags))
                    ypreds_bigrams= list(zip([self.source_tag]+ypreds,ypreds))

                    delta_pred = SparseWeightVector()
                    for y,x in zip(ypreds_bigrams,xwords):
                        delta_pred += SparseWeightVector.code_phi(x,y)

                    delta_ref = SparseWeightVector()
                    for y,x in zip(ytags_bigrams,xwords):
                        delta_ref += SparseWeightVector.code_phi(x,y)
                    
                    self.model += step_size*(delta_ref-delta_pred)
                                    
            print('Loss = ',loss, "Sequence accurracy = ",(N-loss)/N)
            if loss == 0:
                return
            
    def test(self,dataset):
        N       = 0.0
        correct = 0.0
        for ytags,xwords in dataset:
            N += len(ytags)
            ypreds = self.tag(xwords)
            correct += sum([ref == pred for ref,pred in zip(ytags,ypreds)])
        return correct / N

if __name__ == '__main__':           
    corpus = ['Le/D chat/N mange/V la/D souris/N ./PONCT','La/D souris/N danse/V ./PONCT','Il/Pro la/Pro voit/V dans/P la/D cour/N ./PONCT','Le/D chat/N la/Pro mange/V ./PONCT',"Le/D chat/N la/Pro mange/V",'Il/Pro est/V grand/A ./PONCT',"Il/Pro se/Pro dirige/V vers/P l'/D est/N ./PONCT"]
    D = make_dataset(corpus)

    p = StructuredPerceptron()
    p.train(D)
    print(p.test(D))

