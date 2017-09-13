from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding,Flatten

import numpy as np

def make_dataset(text):
    """
    @param text: a list of strings of the form : Le/D chat/N mange/V la/D souris/N ./PONCT
    @return    : a list of couples (xtrigrams,ytags)
    """
    #extracts a trigram dataset 
    dataset = []
    for line in text:
        line         = list([ tuple(w.split('/'))  for w in line.split()])
        tokens       = [FeedForwardTagger.SOURCE_token] + list([tok for(tok,pos) in line]) + [FeedForwardTagger.END_token]
        pos          = list([pos for(tok,pos) in line])
        tok_trigrams = list(zip(tokens,tokens[1:],tokens[2:]))
        dataset.extend(zip(tok_trigrams,pos))
    return dataset
        
class FeedForwardTagger:

    SOURCE_token = "@@@"
    END_token    = '$$$'
    UNK_token    = '__UNK__'

    
    def __init__(self):
        """
        This tagger also trains word embeddings as a byproduct
        @param word_codes: a dictionary of wordforms to idxes
        @param pos_codes: a dictionary of posforms to idxes
        """
        self.word_codes = None
        self.pos_codes  = None
        self.model      = None
        
    def tag(self,tokens,add_dummy_tags=True):
        """
        Unigram tagging of a sequence.
        @param takes as input a list of tokens with source and end
        tags appended
        @param add_dummy_tags:if true adds dummy start and end tokens
        @return the tagged sequence
        """
        if add_dummy_tags:
            tokens = [FeedForwardTagger.SOURCE_token] + tokens + [FeedForwardTagger.END_token]

        tok_trigrams = list(zip(tokens,tokens[1:],tokens[2:]))
        tag_sequence = []  
        for trigram in tok_trigrams:
            xcodes = np.array([[self.word_codes[elt] for elt in trigram]])
            yscores = self.model.predict(xcodes,batch_size=1)
            yhat = self.reverseY[np.argmax(yscores)]
            tag_sequence.append(yhat)
            
        return list(zip(tokens[1:-1],tag_sequence))

        
    def __code_dataset(self,dataset):
        """
        Codes a dataset on integers to make it suitable for training purposes
        Creates coding dictionaries as a side effect.
        
        @param dataset : a list of couples (trigrams of words,ref_tag)
        @return xdata,ydata : a dataset suitable for use with Keras with xdata coded as embeddings input.
        """
        #create dictionaries
        word_lexicon = set([FeedForwardTagger.SOURCE_token,FeedForwardTagger.END_token,FeedForwardTagger.UNK_token])
        pos_lexicon  = set([])
        
        for xtrigram,ytag in dataset:
            word_lexicon.update(xtrigram)
            pos_lexicon.add(ytag)

        self.word_codes = dict([(word,idx) for idx,word in enumerate(word_lexicon)])
        self.pos_codes  = dict([(pos,idx) for idx,pos in enumerate(pos_lexicon)])            
        self.reverseY   = dict([(idx,pos) for (pos,idx) in self.pos_codes.items()])
        self.Nwords     = len(self.word_codes)
        self.Npos       = len(self.pos_codes)

        #codes the dataset
        xdata = []
        ydata = []
        for xtrigram,ytag in dataset:
            xdata.append(list([self.word_codes[elt] for elt in xtrigram]))
            ycode = [0.0]*self.Npos
            ycode[self.pos_codes[ytag]] = 1.0
            ydata.append(np.array(ycode))
        ydata = np.array(ydata)
        #print(self.word_codes)
        #print(self.pos_codes)
        #print(xdata)
        #print(ydata)
        return (xdata,ydata)
            
    def train(self,dataset,
              max_epochs=100,
              embedding_size=4,   #change these values when working at real scale ! 
              hidden_layer_size=16):
        """
        @param dataset: a list of couples (y_tags,x_words)
        """
        xdata, ydata = self.__code_dataset(dataset)
        
        self.model = Sequential()
        self.model.add(Embedding(self.Nwords,embedding_size,input_length=3))
        self.model.add(Flatten())        #concatenates the embeddings layers
        self.model.add(Dense(hidden_layer_size))
        self.model.add(Activation('relu'))
        self.model.add(Dense(self.Npos))
        self.model.add(Activation('softmax'))
        
        self.model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
        self.model.fit(xdata, ydata, batch_size=1, epochs=max_epochs)
        #print(self.model.layers[0].get_weights()) #displays the embeddings
        
    def test(self,dataset):
        """
        @param dataset: a list of (xtrigrams,ytags) as textual values
        @return an accurracy score 
        """
        N       = len(dataset)
        correct = 0.0
        for xwords,ytag in dataset: #it is way more efficient to do the prediction for the whole dataset at once !
            xcodes = np.array([[self.word_codes[elt] for elt in xwords]])
            yscores = self.model.predict(xcodes,batch_size=1)
            yhat = self.reverseY[np.argmax(yscores)]
            if ytag == yhat:
                correct += 1
        return correct / N

if __name__ == '__main__':
    
    corpus = ['Le/D chat/N mange/V la/D souris/N ./PONCT','La/D souris/N danse/V ./PONCT','Il/Pro la/Pro voit/V dans/P la/D cour/N ./PONCT','Le/D chat/N la/Pro mange/V ./PONCT',"Le/D chat/N la/Pro mange/V",'Il/Pro est/V grand/A ./PONCT',"Il/Pro se/Pro dirige/V vers/P l'/D est/N ./PONCT"]
    D = make_dataset(corpus)
    p = FeedForwardTagger()
    p.train(D)
    print(p.test(D))

    s ="Le chat mange la souris .".split()
    print(p.tag(s))
