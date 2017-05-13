from SparseWeightVector import SparseWeightVector
from math import exp,log

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
        dataset.append(tuple(pos,zip(tok_trigrams,tok_bigramsL,tok_bigramsR)))
                    
    return dataset


class LinearCRF:


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
                smax,amax = 0,0
                for pred in range(K):
                    score =  viterbi[i-1,pred] + self.score(self.Y[pred],self.Y[j],sentence[i])
                    if score > smax:
                        smax,amax = score,pred
                viterbi[i,j],history[i,j] = smax,amax
    
        #End state
        smax,amax = 0,0
        for pred in range(K):
            if score > smax:
                smax,amax = score,pred
                
        #Backtrace
        rev_tag_sequence = [self.Y[amax]]
        for i in range(N-1,2,-1):
            amax = history[i,amax]
            rev_tag_sequence.append( self.Y[ history[i-1,amax] ] )
            
        return list(reversed(rev_tag_sequence))

    
    def score(self,y_pred,y,word_repr):
        """
        Scores a CRF clique (psi value for y-1,y,x)
        @param y_pred : prev tag
        @param y  : current tag
        @word_repr : a word data representation (a list of hashable symbols)
        @return a psi (potential) positive value
        """
        return exp(self.model.dot(word_repr,(y_pred,y)))

    
    def forward(self,sentence):
        """
        @param sentence: a list of xwords
        @return a forward matrix and Z (norm constant)
        """
        N = len(sentence)
        K = len(self.Y)
        forward = np.zeros((N,K))  
        #init
        for j in range(K):
            forward[0,j] = self.score(self.source_tag,self.Y[j],sentence[0]) 
        #recurrence
        for i in range(1,N):
            for j in range(K):
                for pred in range(K):
                    forward[i,j] += forward[i-1,pred] * self.score(self.Y[pred],self.Y[j],sentence[i])  
        return (forward,forward[N-1,:].sum())

    def backward(self,sentence):
        """
        @param sentence: a list of xwords
        @return a backward matrix and Z (norm constant)
        """
        N = len(sentence)
        K = len(self.Y)
        
        backward = np.zeros((N,K))
        backward[N-1,:] = 1.0

        #recurrence
        for i in range(N-1,0,-1):
            for j in range(K):
                for succ in range(K):
                    backward[i,j] += backward[i+1,succ] * self.score(self.Y[j],self.Y[succ],sentence[i+1])
        
        return (backward,backward[0,:].sum())
    
    def train(self,dataset,step_size=0.1,max_epochs=100):
        """
        @param dataset: a list of couples (y_tags,x_words)
        """
        self.Y = list(set([y for y in ytags for (ytags,xwords) in dataset]))

        #pre-computes delta_ref (first term of the gradient is constant)
        delta_ref  = SparseWeightVector()
        for ytags,xwords in dataset:
            ytags         = [self.source_tag] + ytags
            ytags_bigrams = list(zip(ytags,ytags[1:]))
            for x,y in zip(xwords,ytags_bigrams):
                delta_ref += SparseWeightVector.code_phi(x,y)
                            
        for e in range(max_epochs):
            
            loss = 0.0
            delta_pred = SparseWeightVector()

            for ytags,xwords in dataset:
                N = len(sentence)
                K = len(self.Y)
                alphas,Z1  = self.forward(xwords)
                betas, Z2  = self.backward(xwords) 
                assert(Z1==Z2)
                Z = Z1
                #init forward-backward
                for ytag in range(K):
                    prob = (self.score(self.source_tag,self.Y[ytag],xwords[0]) * beta[0,ytag]) / Z
                    delta_pred += prob * SparseWeightVector.code_phi(xwords[0],(self.source_tag,self.Y[ytag]))
                #forward-backward loop
                for i in range(1,N):
                    for yprev in range(K):
                        for ytag in range(K):
                            prob = (alpha[i-1,yprev] * self.score(self.Y[yprev],self.Y[ytag],xwords[i]) * beta[i,ytag]) / Z
                            delta_pred += prob * SparseWeightVector.code_phi(xwords[i],(self.Y[yprev],self.Y[ytag]))
            self.model += step_size*(delta_ref-delta_pred)
            print('Loss (log likelihood) = ',loss)

corpus = ['Le/D chat/N mange/V la/D souris/N ./PONCT','La/D souris/N danse/V ./PONCT','Il/Pro la/Pro voit/V dans/P la/D cour/N ./PONCT','Le/D chat/N la/Pro mange/V ./PONCT',"Le/D chat/V la/Pro mange/V"]
D = make_dataset(corpus)
print(D)

crf = LinearCRF()
