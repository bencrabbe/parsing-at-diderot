import io
from collections import defaultdict
from SparseWeightVector import SparseWeightVector

#DATA REPRESENTATION
class DependencyTree:

    def __init__(self,tokens=None, edges=None):
        self.edges  = [] if edges is None else edges                      #couples (gov_idx,dep_idx)
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
        tag_trigrams  = list(zip(taglist,taglist[1:],taglist[2:]))
        self.tokens   = list(zip(wordlist[1:-1],taglist[1:-1],word_trigrams,tag_trigrams))
        
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

class ArcEagerTransitionParser:

    #actions
    LEFTARC  = "LA"
    RIGHTARC = "RA"
    SHIFT    = "S"
    REDUCE   = "R"
    TERMINATE= "T"
    
    def __init__(self):
        self.model = SparseWeightVector()
        
    @staticmethod
    def static_oracle(configuration,reference_arcs,N):
        """
        @param configuration: a parser configuration
        @param reference arcs: a set of dependency arcs
        @param N: the length of the input sequence
        """
        S,B,A,score = configuration
        all_words   = range(N)
        
        if S and B:
            i,j = S[-1], B[0]
            if i!= 0 and (j,i) in reference_arcs:
                return ArcEagerTransitionParser.LEFTARC
            if  (i,j) in reference_arcs:
                return ArcEagerTransitionParser.RIGHTARC
        if S and any([(k,S[-1]) for k in all_words]):
            return ArcEagerTransitionParser.REDUCE
        if B:
            return ArcEagerTransitionParser.SHIFT
        return ArcEagerTransitionParser.TERMINATE

    
    def static_oracle_derivation(self,ref_parse):
        """
        This generates a static oracle reference derivation from a sentence
        @param ref_parse: a DependencyTree object
        @return : the oracle derivation as a list of (Configuration,action) couples
        """
        sentence = ref_parse.tokens
        edges    = set(ref_parse.edges)
        N        = len(sentence)
        
        C = ((0,),tuple(range(1,len(sentence))),tuple(),0.0)       #A config is a hashable quadruple with score 
        action = ArcEagerTransitionParser.static_oracle(C,edges,N)
        derivation.append((C,action))

        while C[1] and action != ArcEagerTransitionParser.TERMINATE:
                        
            if action ==  ArcEagerTransitionParser.SHIFT:
                C = self.shift(C,sentence)
            elif action == ArcEagerTransitionParser.LEFTARC:
                C = self.leftarc(C,sentence)
            elif action == ArcEagerTransitionParser.RIGHTARC:
                C = self.rightarc(C,sentence)
            elif action == ArcEagerTransitionParser.REDUCE:
                C = self.reduce_config(C,sentence)
            elif action ==  ArcEagerTransitionParser.TERMINATE:
                C = self.terminate(C,sentence)

            action = ArcEagerTransitionParser.static_oracle(C,edges,N)
            derivation.append((C,action))
                            
        return derivation
                
    def shift(self,configuration,tokens):
        """
        Performs the shift action and returns a new configuration
        """
        S,B,A,score = configuration
        w0 = B[0]
        return (S + (w0,),B[1:],A,score+self.score(configuration,ArcEagerTransitionParser.SHIFT,tokens)) 

    def leftarc(self,configuration,tokens):
        """
        Performs the left arc action and returns a new configuration
        """
        S,B,A,score = configuration
        i,j = S[-1],B[0]
        return (S[:-1],B,A + ((j,i),),score+self.score(configuration,ArcEagerTransitionParser.LEFTARC,tokens)) 

    def rightarc(self,configuration,tokens):
        S,B,A,score = configuration
        i,j = S[-1],B[0]
        return (S+[j],B[1:], A + ((i,j),),score+self.score(configuration,ArcEagerTransitionParser.RIGHTARC,tokens)) 

    def reduce_config(self,configuration,tokens):
        S,B,A,score = configuration
        i = S[-1]
        return (S,B,A,score+self.score(configuration,ArcEagerTransitionParser.REDUCE,tokens))
    
    def terminate(self,configuration,tokens):
        S,B,A,score = configuration
        return (S,B,A,score+self.score(configuration,ArcEagerTransitionParser.TERMINATE,tokens))        


    def parse_one(self,sentence):
        """
        Greedy parsing
        @param sentence: a list of tokens
        """
        N = len(sentence)
        C = (tuple(),tuple(range(N)),tuple(),0.0) #A config is a hashable quadruple with score 
        candidates = [C]
        while candidates:
            C = candidates[0][0]
            if  candidates[0][1] == ArcEagerTransitionParser.TERMINATE:
                break
            S,B,A,score = C
            candidates = []
            if B:
                candidates.append((self.shift(C,sentence),ArcEagerTransitionParser.SHIFT))
                j = B[0]
                if not any([(k,j) in A for k in range(N)]):
                    candidates.append((self.rightarc(C,sentence),ArcEagerTransitionParser.RIGHTARC))
            if S:
                i = S[-1]     
                if S and B and i != 0 and not any([(k,i) in A for k in range(N)]): 
                    candidates.append((self.leftarc(C,sentence),ArcEagerTransitionParser.LEFTARC))
                if any([(k,i) in A for k in range(N)]):
                    candidates.append((self.reduce_config(C,sentence),ArcEagerTransitionParser.REDUCE))
            if not B:
                candidates.append((self.terminate(C,sentence),ArcEagerTransitionParser.TERMINATE))
                
            candidates.sort(key=lambda x:x[0][3],reverse=True)
            
        S,B,A,score = C
        #connect to 0 any dummy root
        As = set(A)
        for s in S:
            if not any([(k,s) in As for k in range(N)]):
                As.add((0,s))
        return DependencyTree(tokens=sentence,edges=As)


    def score(self,configuration,action,tokens):
        """
        Computes the prefix score of a derivation
        @param configuration : a quintuple (S,B,A,score,history)
        @param action: an action label in {LEFTARC,RIGHTARC,TERMINATE,SHIFT}
        @param tokens: the x-sequence of tokens to be parsed
        @return a prefix score
        """
        S,B,A,old_score = configuration
        config_repr = self.__make_config_representation(S,B,tokens)
        return old_score + self.model.dot(config_repr,action)

    def __make_config_representation(self,S,B,tokens):
        """
        This gathers the information for coding the configuration as a feature vector.
        @param S: a configuration stack
        @param B  a configuration buffer
        @return an ordered list of tuples 
        """
        #default values for inaccessible positions
        s0w,s1w,s0t,s1t,b0w,b1w,b0t,b1t = "_UNDEF_","_UNDEF_","_UNDEF_","_UNDEF_","_UNDEF_","_UNDEF_","_UNDEF_","_UNDEF_"

        if len(S) > 0:
            s0w,s0t = tokens[S[-1]][0],tokens[S[-1]][1]
        if len(S) > 1:
            s1w,s1t = tokens[S[-2]][0],tokens[S[-2]][1]
        if len(B) > 0:
            b0w,b0t = tokens[B[0]][0],tokens[B[0]][1]
        if len(B) > 1:
            b1w,b1t = tokens[B[1]][0],tokens[B[1]][1]
            
        wordlist = [s0w,s1w,b0w,b1w]
        taglist  = [s0t,s1t,b0t,b1t]
        word_bigrams = list(zip(wordlist,wordlist[1:]))
        tag_bigrams = list(zip(taglist,taglist[1:]))
        word_trigrams = list(zip(wordlist,wordlist[1:],wordlist[2:]))
        tag_trigrams = list(zip(taglist,taglist[1:],taglist[2:]))
        return word_bigrams + tag_bigrams + word_trigrams + tag_trigrams
    
    def test(self,dataset,beam_size=4):
        """
        @param dataset: a list of DependencyTrees
        @param beam_size: size of the beam
        """
        N       = len(dataset)
        sum_acc = 0.0
        for ref_tree in dataset:
            tokens    = ref_tree.tokens
            pred_tree = self.parse_one(tokens,beam_size)
            print(pred_tree)
            print()
            sum_acc   += ref_tree.accurracy(pred_tree)
        return sum_acc/N

        
    def train(self,dataset,step_size=1.0,max_epochs=100,beam_size=4):
        """
        @param dataset : a list of dependency trees
        """
        #TODO
        N = len(dataset)
        sequences = list([ (dtree.tokens,self.oracle_derivation(dtree)) for dtree in dataset])
        
        for e in range(max_epochs):
            loss = 0.0
            for tokens,ref_derivation in sequences:
                pred_beam = self.parse_one(tokens,beam_size,get_beam=True)
                (update, ref_prefix,pred_prefix) = self.early_prefix(ref_derivation,pred_beam)
                #print('R',ref_derivation)
                #print('P',pred_prefix)
                #self.test(dataset,beam_size)

                if update:
                    #print (pred_prefix)
                    loss += 1.0
                    delta_ref = SparseWeightVector()
                    current_config = ref_prefix[0][1]
                    for action,config in ref_prefix:
                        S,B,A,score = current_config
                        x_repr = self.__make_config_representation(S,B,tokens)
                        delta_ref += SparseWeightVector.code_phi(x_repr,action)
                        current_config = config
                        
                    delta_pred = SparseWeightVector()
                    current_config = pred_prefix[0][1]
                    for action,config in pred_prefix:
                        S,B,A,score = current_config
                        x_repr = self.__make_config_representation(S,B,tokens)
                        delta_pred += SparseWeightVector.code_phi(x_repr,action)
                        current_config = config

                    self.model += step_size*(delta_ref-delta_pred)
            print('Loss = ',loss, "%Exact match = ",(N-loss)/N)
            if loss == 0.0:
                return

            
test = """
1 le   D     2
2 chat N     3
3 dort V     0
4 .    PONCT 3
"""
test2 = """
1 le      D     2
2 tapis   N     3
3 est     V     5
4 rouge   A     3
5 et      CC    0
6 le      D     7
7 chat    N     8
8 mange   V     5
9 la      D     10
10 souris N     8
11 .      PONCT 5
"""

istream = io.StringIO(test)
istream2 =  io.StringIO(test2)
d = DependencyTree.read_tree(istream)
d2 = DependencyTree.read_tree(istream2)
p = ArcStandardTransitionParser()
p.train([d,d2],max_epochs=100,beam_size=4)
print(p.test([d,d2],beam_size=4))
