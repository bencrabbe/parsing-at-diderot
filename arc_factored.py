#!/usr/bin/env python3
import io
from SparseWeightVector import SparseWeightVector
from math import inf
import numpy as np

"""
This is the 1st order projective spanning tree algorithm of Eisner
1996 weighted by a perceptron.
"""

#DATA REPRESENTATION
class DependencyTree:

    def __init__(self,tokens=None, edges=None):
        self.edges  = [] if edges is None else edges                       # couples (gov_idx,dep_idx)
        self.tokens = [('$ROOT$','$ROOT$')] if tokens is None else tokens #couples (wordform,postag)

    
    def __str__(self):
        gdict = dict([(d,g) for (g,d) in self.edges])
        return '\n'.join(['\t'.join([str(idx+1),tok[0],tok[1],str(gdict[idx+1])]) for idx,tok in enumerate(self.tokens[1:])])
                     
    def __make_ngrams(self):
        """
        Makes word representations suitable for feature extraction
        """
        BOL = '@@@'
        EOL = '$$$'
        wordlist = [BOL] + list([w for w,t in self.tokens]) + [EOL]
        taglist = [BOL] + list([t for w,t in self.tokens]) + [EOL]
        word_trigrams = list(zip(wordlist,wordlist[1:],wordlist[2:]))
        tag_trigrams = list(zip(taglist,taglist[1:],taglist[2:]))
        self.tokens = list(zip(wordlist[1:-1],taglist[1:-1],word_trigrams,tag_trigrams))
        
    @staticmethod
    def read_tree(istream):
        """
        Reads a tree from an input stream
        @param istream: the stream where to read from
        @return: a DependencyTree instance 
        """
        deptree = DependencyTree()
        bfr = istream.readline()
        while True:
            if (bfr.isspace() or bfr == ''):
                if deptree.N() > 1:
                    deptree.__make_ngrams()
                    return deptree
                bfr = istream.readline()
            else:
                idx, word, tag, governor_idx = bfr.split()
                deptree.tokens.append((word,tag))
                deptree.edges.append((int(governor_idx),int(idx)))
                bfr = istream.readline()
        deptree.__make_ngrams()
        return deptree

    def accurracy(self,other):
        """
        Compares this dep tree with another by computing their UAS.
        @param other: other dep tree
        @return : the UAS as a float
        """
        assert(len(self.edges) == len(other.edges))
        S1 = set(self.edges)
        S2 = set(other.edges)
        return len(S1.intersection(S2)) / len(S1)
    
    def N(self):
        """
        Returns the length of the input
        """
        return len(self.tokens)
    
    def __getitem__(self,idx):
        """
        Returns the token at index idx
        """
        return self.tokens[idx]

    
class ArcFactoredParser:
        
    LEFTARC  = "L"
    RIGHTARC = "R"
    
    def __init__(self):
        
        self.model = SparseWeightVector()


    def _argmax(self,x,y,argmax,argvalue):
        """
        computes m = max(x,y) and updates prev argmax by argvalue if max(x,y) = y
        @param x: number (current max)
        @param y: number
        @param argmax: the current argmax
        @param argvalue: the potential argmax
        @return (max,argmax) a couple with the current max and argmax
        """
        if y > x:
            return (y,argvalue)
        return (x,argmax)
        
        
    def parse_one(self,sentence):
        """
        @param sentence: a list of tokens as encoded in a dependency
        tree (first token is a dummy root token)
        @return : a DependencyTree Object 
        """
        COMPLETE,INCOMPLETE  = 1,0
        LEFTARROW,RIGHTARROW = 0,1
        
        N = len(sentence)
        chart   = np.zeros((N,N,2,2))
        history = {}

        #recurrence
        for span_length in range(1,N):
            for i in range(N-span_length):
                j = i + span_length
                #incomplete
                max_left,max_right = -inf,-inf
                amax_left,amax_right = i,i
                for k in range(i,j):
                    tmp_score = chart[i][k][RIGHTARROW][COMPLETE] \
                              + chart[k+1][j][LEFTARROW][COMPLETE] \
                              + self.score(j,i,sentence) 
                    max_left,amax_left  = self._argmax(max_left,tmp_score,amax_left,k)
                    tmp_score = chart[i][k][RIGHTARROW][COMPLETE] \
                              + chart[k+1][j][LEFTARROW][COMPLETE] \
                              + self.score(i,j,sentence)
                    max_right,amax_right = self._argmax(max_right,tmp_score,amax_right,k)                    
                chart[i][j][LEFTARROW][INCOMPLETE]  = max_left
                chart[i][j][RIGHTARROW][INCOMPLETE] = max_right
                history[(i,j,LEFTARROW,INCOMPLETE)] = [(i,amax_left,RIGHTARROW,COMPLETE),(amax_left+1,j,LEFTARROW,COMPLETE)]
                history[(i,j,RIGHTARROW,INCOMPLETE)]= [(i,amax_right,RIGHTARROW,COMPLETE),(amax_right+1,j,LEFTARROW,COMPLETE)]

                #complete
                max_left,max_right = -inf,-inf
                amax_left,amax_right = i,i
                for k in range(i,j):
                    max_left,amax_left   = self._argmax(max_left,chart[i][k][LEFTARROW][COMPLETE] + chart[k][j][LEFTARROW][INCOMPLETE],amax_left,k)
                for k in range(i+1,j+1):
                    max_right,amax_right = self._argmax(max_right,chart[i][k][RIGHTARROW][INCOMPLETE] + chart[k][j][RIGHTARROW][COMPLETE],amax_right,k)
                chart[i][j][LEFTARROW][COMPLETE]   = max_left
                chart[i][j][RIGHTARROW][COMPLETE]  = max_right
                history[(i,j,LEFTARROW,COMPLETE)]  = [(i,amax_left,LEFTARROW,COMPLETE),(amax_left,j,LEFTARROW,INCOMPLETE)]
                history[(i,j,RIGHTARROW,COMPLETE)] = [(i,amax_right,RIGHTARROW,INCOMPLETE),(amax_right,j,RIGHTARROW,COMPLETE)]

        #backtrace (collects edges of the dependency tree)
        edges  = []
        agenda = [(0,N-1,RIGHTARROW,COMPLETE)]
        while agenda:
            current_item = agenda.pop()
            (i,j,direction,c) = current_item
            if c == INCOMPLETE and i != j:
                if direction == LEFTARROW:
                    edges.append((j,i))
                elif direction == RIGHTARROW:
                    edges.append((i,j))
            if current_item in history:
                agenda.extend(history[current_item])
        
        return DependencyTree(tokens=sentence,edges=edges)
                
    def score(self,gov_idx,dep_idx,toklist):
        """
        @param gov_idx,dep_idx : the indexes of the governor and
        dependant in the sentence
        @toklist: the list of tokens of the sentence
        @return : a float (score)
        """
        dep_repr = self.__make_arc_representation(gov_idx,dep_idx,toklist)
        ylabel   =  ArcFactoredParser.RIGHTARC if gov_idx < dep_idx else ArcFactoredParser.LEFTARC
        return self.model.dot(dep_repr,ylabel)

    
    def __make_arc_representation(self,gov_idx,dep_idx,toklist):
        """
        Creates a list of values from which to code a dependency arc as binary features 
        Inserts the interactions between the words for coding the dependency in the representation.
        
        @param gov_idx,dep_idx : the indexes of the governor and dependant in the sentence
        @toklist: the list of tokens of the sentence
        @return : a list of tuples ready to be binarized and scored.
        """
        interaction1 = (toklist[gov_idx][1],toklist[dep_idx][1],)
        interaction2 = (toklist[gov_idx][0],toklist[dep_idx][0],)
        #add more interactions here to improve the parser

        return toklist[gov_idx] + toklist[dep_idx] + (interaction1,) + (interaction2,)


    def train(self,dataset,step_size=1.0,max_epochs=100):
        """
        @param dataset : a list of dependency trees
        """
        N = len(dataset)
        for e in range(max_epochs):
            loss = 0.0
            for ref_tree in dataset:
                tokens = ref_tree.tokens
                pred_tree = self.parse_one(tokens)

                if ref_tree.accurracy(pred_tree) != 1.0:
                    loss += 1.0
                    
                    delta_ref = SparseWeightVector()
                    for gov_idx,dep_idx in ref_tree.edges:
                        x_repr = self.__make_arc_representation(gov_idx,dep_idx,tokens)
                        ylabel = ArcFactoredParser.RIGHTARC if gov_idx < dep_idx else ArcFactoredParser.LEFTARC
                        delta_ref += SparseWeightVector.code_phi(x_repr,ylabel)
                    
                    delta_pred = SparseWeightVector()
                    for gov_idx,dep_idx in pred_tree.edges:    
                        x_repr = self.__make_arc_representation(gov_idx,dep_idx,tokens)
                        ylabel = ArcFactoredParser.RIGHTARC if gov_idx < dep_idx else ArcFactoredParser.LEFTARC
                        delta_pred += SparseWeightVector.code_phi(x_repr,ylabel)

                    self.model += step_size*(delta_ref-delta_pred)
                
            print('Loss = ',loss, "Exact match accurracy = ",(N-loss)/N)
            if loss == 0.0:
                return
            
    def test(self,dataset):
        N       = len(dataset)
        sum_acc = 0.0
        for ref_tree in dataset:
            tokens    = ref_tree.tokens
            pred_tree = self.parse_one(tokens)
            print(pred_tree)
            print()
            sum_acc   = ref_tree.accurracy(pred_tree)
        return sum_acc/N
                
test = """
1 le   D     2
2 chat N     3
3 dort V     0
4 .    PONCT 3
"""
test2 = """
1 le    D     2
2 tapis N     3
3 est   V     0
4 rouge A     3  
5 .     PONCT 3
"""

istream = io.StringIO(test)
istream2 =  io.StringIO(test2)
d = DependencyTree.read_tree(istream)
d2 = DependencyTree.read_tree(istream2)

p = ArcFactoredParser()
p.train([d,d2],max_epochs=100)
p.test([d,d2])
