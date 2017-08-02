#! /usr/bin/env python
from SparseWeightVector import SparseWeightVector

"""
Transition based const. parser (includes tagging)
Scored with beam search and perceptron.
"""

class ConsTree:

    def __init__(self,label,children=None):
        self.label = label
        self.children = [] if children is None else children

    def is_leaf(self):
        return self.children == []
    
    def add_child(self,child_node):
        self.children.append(child_node)
        
    def arity(self):
        return len(self.children)
        
    def get_child(self,idx=0):
        """
        returns the ith child of this node
        """
        return self.children[idx]

    def __str__(self):
        """
        pretty prints the tree
        """
        return self.label if self.is_leaf() else '(%s %s)'%(self.label,' '.join([str(child) for child in self.children]))

    def tokens(self,labels=True):
        """
        @param labels: returns a list of strings if true else returns
        a list of ConsTree objects
        @return the list of words at the leaves of the tree
        """
        if self.is_leaf():
                return [self.label] if labels else [self]
        else:
            result = []
            for child in self.children:
                result.extend(child.tokens(labels))
            return result
                
    def index_leaves(self):
        """
        Adds an numeric index to each leaf node
        """
        for idx,elt in enumerate(self.tokens(labels=False)):
            elt.idx = idx
            
    def triples(self):
        """
        Extracts a list of evalb triples from the tree
        (supposes leaves are indexed)
        """
        if self.is_leaf():
            return [(self.idx,self.idx+1,self.label)]
        else:
            subtriples = []
            for child in self.children:
                subtriples.extend(child.triples())
            leftidx  = min([idx for idx,jdx,label in subtriples])
            rightidx = max([jdx for idx,jdx,label in subtriples])
            subtriples.append((leftidx,rightidx,self.label))
            return subtriples

    def compare(self,other):
        """
        Compares this tree to another and computes precision,recall,
        fscore. Assumes self is the reference tree
        @param other: the predicted tree
        @return (precision,recall,fscore)
        """
        self.index_leaves()
        other.index_leaves()
        ref_triples  = set(self.triples())
        pred_triples = set(other.triples())
        intersect = ref_triples.intersection(pred_triples)
        isize = len(intersect)
        P = isize/len(pred_triples)
        R = isize/len(ref_triples)
        F = (2*P*R)/(P+R)
        return (P,R,F)

    
    def close_unaries(self,dummy_annotation='$'):
        """
        In place (destructive) unary closure of unary branches
        """
        if self.arity() == 1:
            current      = self
            unary_labels = []
            while current.arity() == 1 and not current.get_child().is_leaf():
                unary_labels.append(current.label)
                current = current.get_child()
            unary_labels.append(current.label)
            self.label = dummy_annotation.join(unary_labels)
            self.children = current.children
            
        for child in self.children:
            child.close_unaries()

    def expand_unaries(self,dummy_annotation='$'):
        """
        In place (destructive) expansion of unary symbols.
        """
        if dummy_annotation in self.label:
            unary_chain = self.label.split(dummy_annotation)
            self.label  = unary_chain[0]
            backup      = self.children
            current     = self
            for label in unary_chain[1:]:
                c = ConsTree(label)
                current.children = [c] 
                current = c
            current.children = backup
            
        for child in self.children:
            child.expand_unaries()

            
    def left_markovize(self,dummy_annotation=':'):
        """
        In place (destructive) left markovization (order 0)
        """
        if len(self.children) > 2:
            left_sequence = self.children[:-1]
            dummy_label = self.label if self.label[-1] == dummy_annotation else self.label+dummy_annotation
            dummy_tree = ConsTree(dummy_label, left_sequence)
            self.children = [dummy_tree,self.children[-1]]
        for child in self.children:
            child.left_markovize()

    def right_markovize(self,dummy_annotation=':'):
        """
        In place (destructive) right markovization (order 0)
        """
        if len(self.children) > 2:
            right_sequence = self.children[1:]
            dummy_label = self.label if self.label[-1] == dummy_annotation else self.label+dummy_annotation
            dummy_tree = ConsTree(dummy_label, right_sequence)
            self.children = [self.children[0],dummy_tree]
        for child in self.children:
            child.right_markovize()

    def unbinarize(self,dummy_annotation=':'):
        """
        In place (destructive) unbinarization
        """
        newchildren = []
        for child in self.children:
            if child.label[-1] == dummy_annotation:
                child.unbinarize()
                newchildren.extend(child.children)
            else:
                child.unbinarize()
                newchildren.append(child)
        self.children = newchildren

    def collect_nonterminals(self):
        """
        Returns the list of nonterminals found in a tree:
        """
        if not self.is_leaf():
            result =  [self.label]
            for child in self.children:
                result.extend(child.collect_nonterminals())
            return result
        return []

    @staticmethod
    def read_tree(input_str):
        """
        Reads a one line s-expression.
        This is a non robust function to syntax errors
        @param input_str: a s-expr string
        @return a ConsTree object
        """
        tokens = input_str.replace('(',' ( ').replace(')',' ) ').split()
        stack = [ConsTree('dummy')]
        for idx,tok in enumerate(tokens):
            if tok == '(':
                current = ConsTree(tokens[idx+1])
                stack[-1].add_child(current)
                stack.append(current)
            elif tok == ')':
                stack.pop()
            else:
                if tokens[idx-1] != '(':
                    stack[-1].add_child(ConsTree(tok))
        assert(len(stack) == 1)
        return stack[-1].get_child()


class ConstituentTransitionParser:

    SHIFT  = "S"
    REDUCE = "R"
    STOP   = "!"
    
    def __init__(self):
        self.model = SparseWeightVector()
        self.nonterminals = []

    def static_oracle(self,stack,buffer,ref_triples):
        """
        Returns the action to do given a configuration and a ref parse tree
        @param ref_triples : the triples from the reference tree
        @param stack: the config stack
        @param buffer: a list of integers
        @return a couple (parse action, action param)
        """
        if len(stack) >= 2:
            (i,k,X1),(k,j,X2) = stack[-2],stack[-1]
            for X in self.nonterminals:
                if (i,j,X) in ref_triples:
                    return (ConstituentTransitionParser.REDUCE,X)
        if buffer:
            idx = buffer[0]
            for tag in self.nonterminals:
                if(idx,idx+1,tag) in ref_triples:
                    return (ConstituentTransitionParser.SHIFT,tag)
        return (ConstituentTransitionParser.STOP,ConstituentTransitionParser.STOP)

    
    def reference_derivation(self,ref_tree):
        """
        Returns a reference derivation given a reference tree
        @param ref_tree: a ConsTree
        """
        ref_tree.index_leaves()
        ref_triples = set(ref_tree.triples())
        sentence = ref_tree.tokens()
        N = len(sentence)

        action = (None,None)
        c = (tuple(),tuple(range(N)),0.0)
        derivation = [(action,c)]
        
        for t in range(2*N):#because 2N-1+terminate
            S,B,score = c
            action,param = self.static_oracle(S,B,ref_triples)
            if action == ConstituentTransitionParser.REDUCE:
                c = self.reduce(c,param,sentence)
            elif action ==  ConstituentTransitionParser.SHIFT:
                c = self.shift(c,param,sentence)
            else:
                c = self.terminate(c,sentence)
            derivation.append(((action,param),c))
        return derivation


    def build_tree(self,derivation,sentence):
        """
        Builds a ConsTree from a parse derivation
        @param derivation: a parse derivation
        @param sentence: a list of tokens
        @return a ConsTree
        """
        tree_stack = [ ]
        for (action,param) , C in derivation:
            S,B,score = C 
            if action ==  ConstituentTransitionParser.SHIFT:
                i,j,lbl = S[-1]
                tag_node = ConsTree(param)
                leaf_node = ConsTree(sentence[i])
                tag_node.add_child(leaf_node)
                tree_stack.append(tag_node)
            elif action == ConstituentTransitionParser.REDUCE:
                root_node =  ConsTree(param)
                rnode = tree_stack.pop()
                lnode = tree_stack.pop()
                root_node.children = [lnode,rnode]
                tree_stack.append(root_node)
        return tree_stack[-1]
                
    def reduce(self,C,param,sentence):
        """
        Performs a reduction from the current configuration and returns the result
        @param S: a stack
        @param B: a buffer
        @param param: the category for reduction
        @return a configuration
        """
        S,B,score   = C
        i,k,_       = S[-2]
        k,j,_       = S[-1]
        return (S[:-2]+((i,j,param),),B,score+self.score(C,(ConstituentTransitionParser.REDUCE,param),sentence))
    
    def shift(self,C,param,sentence):
        """
        Performs a reduction from the current configuration and returns the result
        @param S: a stack
        @param B: a buffer
        @param param: the category for reduction
        @return a configuration
        """
        S,B,score = C
        idx       = S[-1][1] if S else 0
        return    (S+((idx,idx+1,param),),B[1:],score+self.score(C,(ConstituentTransitionParser.SHIFT,param),sentence))
        
    def terminate(self,C,sentence):
        """
        Performs a stop action returns the result
        """
        S,B,score = C
        return    (S,B,score+self.score(C,(ConstituentTransitionParser.STOP,ConstituentTransitionParser.STOP),sentence))
        

    def score(self,configuration,action,tokens):
        """
        Computes the prefix score of a derivation
        @param configuration : a triple (S,B,score)
        @param action: an action label 
        @param tokens: the x-sequence of tokens to be parsed
        @return a prefix score
        """
        S,B,old_score = configuration
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
        s0cat,s1cat,s0l,s0r,s1l,s1r,b0,b1,b2 = "_UNDEF_","_UNDEF_","_UNDEF_","_UNDEF_","_UNDEF_","_UNDEF_","_UNDEF_","_UNDEF_","_UNDEF_"
        
        if len(S) > 0:
            i,j,lbl = S[-1]
            s0l,s0r,s0cat = tokens[i],tokens[j-1],lbl
        if len(S) > 1:
            i,j,lbl = S[-2]
            s1l,s1r,s1cat = tokens[i],tokens[j-1],lbl
        if len(B) > 0:
            b0 = tokens[B[0]]
        if len(B) > 1:
            b1 = tokens[B[1]]
        if len(B) > 2:
            b2 = tokens[B[2]]

        wordlist = [s0l,s0r,s1l,s1r,b0,b1,b2]
        catlist  = [s0cat,s1cat,b0]
        word_bigrams = list(zip(wordlist,wordlist[1:]))
        word_trigrams = list(zip(wordlist,wordlist[1:],wordlist[2:]))
        cat_bigrams = list(zip(catlist,catlist[1:]))
        
        return word_bigrams + word_trigrams + cat_bigrams


    def transform(self,dataset,left_markov = True):
        """
        In place (destructive) conversion of a treebank to Chomsky Normal Form.
        Builds the list of the parser nonterminals as a side effect
        and indexes references trees.
         
        @param dataset a list of ConsTrees
        @param left_markov: if true -> left markovization else right markovization
        """
        all_nonterminals = set()
        for tree in dataset:
            tree.close_unaries()
            if left_markov:
                tree.left_markovize()
            else:
                tree.right_markovize()
            all_nonterminals.update(tree.collect_nonterminals()) 
        self.nonterminals = list(all_nonterminals)
    
    def parse_one(self,sentence,beam_size=4,get_beam=False,deriv=False,untransform=True):
        """
        @param sentence: a list of strings
        @param beam_size: size of the beam
        @param get_beam : returns the beam instead of tree like structures
        @param deriv: returns the derivation instead of the parse tree
        @param untransform: bool if true unbinarizes the resulting tree.
        """
        
        actions = [ConstituentTransitionParser.SHIFT,\
                   ConstituentTransitionParser.REDUCE,\
                   ConstituentTransitionParser.STOP]
        all_actions = list([(a,p) for a in actions for p in self.nonterminals])
        
        N = len(sentence)
        init = (tuple(),tuple(range(N)),0.0) #A config is a hashable triple with score 
        current_beam = [(-1,(None,None),init)]
        beam = [current_beam]
            
        for i in range(2*N): #because 2*N-1+terminate
            next_beam = []
            for idx, ( _ ,action,config) in enumerate(current_beam):
                S,B,score = config 
                for (a,p) in all_actions:
                    if a ==  ConstituentTransitionParser.SHIFT:
                        if B:
                            newconfig = self.shift(config,p,sentence)
                            next_beam.append((idx,(a,p),newconfig))
                    elif a == ConstituentTransitionParser.REDUCE:
                        if len(S) >= 2:
                            newconfig = self.reduce(config,p,sentence)
                            next_beam.append((idx,(a,p),newconfig))
                    elif a == ConstituentTransitionParser.STOP:
                        if len(S) < 2 and not B:
                            newconfig = self.terminate(config,sentence)
                            next_beam.append((idx,(a,a),newconfig))
            next_beam.sort(key=lambda x:x[2][2],reverse=True)
            next_beam = next_beam[:beam_size]
            beam.append(next_beam)
            current_beam = next_beam
        
        if get_beam:
            return beam
        else:
            #Backtrace for derivation
            idx      = 1
            prev_jdx = 0
            derivation = []
            while prev_jdx != -1:
                current = beam[-idx][prev_jdx]
                prev_jdx,prev_action,C = current
                derivation.append((prev_action,C))
                idx += 1
            derivation.reverse()
            if deriv:
                return derivation
            else:
                result =  self.build_tree(derivation,sentence)
                if untransform:
                    result.unbinarize()
                    result.expand_unaries()
                return result

    def early_prefix(self,ref_parse,beam):
        """
        Finds the prefix for early update, that is the prefix where the ref parse fall off the beam.
        @param ref_parse: a parse derivation
        @param beam: a beam output by the parse_one function
        @return (bool, ref parse prefix, best in beam prefix)
                the bool is True if update required false otherwise
        """
        idx = 0
        for (actionR,configR),(beamCol) in zip(ref_parse,beam):
            found = False
            for source_idx,action,configTarget in beamCol:
                if action == actionR and configTarget[:-1] == configR[:-1]: #-1 -> does not test score equality
                    found = True
                    break
            if not found:
                #backtrace
                jdx = idx
                source_idx = 0
                early_prefix = []
                while jdx >= 0:
                    new_source_idx,action,config = beam[jdx][source_idx]
                    early_prefix.append( (action,config))
                    source_idx = new_source_idx
                    jdx -= 1
                early_prefix.reverse()
                return (True, ref_parse[:idx+1],early_prefix)
            idx+=1
        #if no error found check that the best in beam is the ref parse
        last_ref_action,last_ref_config     = ref_parse[-1]
        _,last_pred_action,last_pred_config =  beam[-1][0]
        if last_pred_config[:-1] == last_ref_config[:-1]:
            return (False,None,None) #returns a no update message
        else:#backtrace
            jdx = len(beam)-1
            source_idx = 0
            early_prefix = []
            while jdx >= 0:
                new_source_idx,action,config = beam[jdx][source_idx]
                early_prefix.append( (action,config) )
                source_idx = new_source_idx
                jdx -= 1
            early_prefix.reverse()
            return (True,ref_parse,early_prefix)
                

    def test(self,treebank,beam_size=4):
        """         
        @param treebank a list of ConsTrees
        @param left_markov: if true -> left markovization else right markovization
        @return the avg f-score
        """
        Fscores = []
        for tree in treebank:
            result = self. parse_one(tree.tokens(),beam_size)
            print(result)
            P,R,F = tree.compare(result)
            Fscores.append(F)
        return sum(Fscores)/len(Fscores)
            
    def train(self,treebank,step_size=1.0,max_epochs=100,beam_size=4,left_markov=True):
        """         
        @param treebank a list of ConsTrees
        @param left_markov: if true -> left markovization else right markovization
        """
        self.transform(treebank,left_markov)
        dataset = list([(tree.tokens(),self.reference_derivation(tree)) for tree in treebank])
        N = len(dataset)
        for e in range(max_epochs):
            loss = 0.0
            for sentence,ref_derivation in dataset:
                pred_beam = (self.parse_one(sentence,get_beam=True))
                (update, ref_prefix,pred_prefix) = self.early_prefix(ref_derivation,pred_beam)
                if update:
                    loss += 1.0
                    delta_ref = SparseWeightVector()
                    current_config = ref_prefix[0][1]
                    for action,config in ref_prefix[1:]:
                        S,B,score = current_config
                        x_repr = self.__make_config_representation(S,B,sentence)
                        delta_ref += SparseWeightVector.code_phi(x_repr,action)
                        current_config = config
                        
                    delta_pred = SparseWeightVector()
                    current_config = pred_prefix[0][1]
                    for action,config in pred_prefix[1:]:
                        S,B,score = current_config
                        x_repr = self.__make_config_representation(S,B,sentence)
                        delta_pred += SparseWeightVector.code_phi(x_repr,action)
                        current_config = config

                    self.model += step_size*(delta_ref-delta_pred)
                    
            print('Loss = ',loss, "%Exact match = ",(N-loss)/N)
            if loss == 0.0:
                return

                    
                
x = ConsTree.read_tree('(S (NP (D le) (N chat)) (VN (V mange)) (NP (D la) (N souris)) (PP (P sur) (NP (D le) (N paillasson))) (PONCT .))')
y = ConsTree.read_tree('(S (NP (D la) (N souris)) (VN (V dort)) (PONCT .))')
z = ConsTree.read_tree('(S (NP (D le) (N cuisinier)) (VN (V mange)) (NP (D une) (N salade) (PP (P avec) (NP (D des) (N cornichons)))) (PONCT .))')

parser = ConstituentTransitionParser()
parser.train([x,y,z])

x = ConsTree.read_tree('(S (NP (D le) (N chat)) (VN (V mange)) (NP (D la) (N souris)) (PP (P sur) (NP (D le) (N paillasson))) (PONCT .))')
y = ConsTree.read_tree('(S (NP (D la) (N souris)) (VN (V dort)) (PONCT .))')
z = ConsTree.read_tree('(S (NP (D le) (N cuisinier)) (VN (V mange)) (NP (D une) (N salade) (PP (P avec) (NP (D des) (N cornichons)))) (PONCT .))')


print(parser.test([x,y,z]))
