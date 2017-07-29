#! /usr/bin/env python
import numpy as np
from numpy import inf
from SparseWeightVector import SparseWeightVector


"""
This is an exact Viterbi perceptron parser
working with spans.
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


    
class ViterbiCKY:
    """
    This implements a CKY parser with perceptron scoring.
    """
    def __init__(self):
        """
        @param weights: a SparseWeightVector
        @param non_terminals: an ordered list of non terminals
        """
        self.model = SparseWeightVector()
        self.nonterminals_decode = [] #maps integers to symbols
        self.nonterminals_code   = {} #maps symbols to integers

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
        self.nonterminals_decode = list(all_nonterminals)
        self.nonterminals_code = dict(zip(self.nonterminals_decode,range(len(self.nonterminals_decode))))
        for tree in dataset:
            tree.index_leaves()

    @staticmethod
    def ngram_tokens(toklist):
        """
        Returns the input with additional fields for ngrams based features
        """
        BOL = '@@@'
        EOL = '$$$'
        wordlist = [BOL] + toklist + [EOL]
        word_trigrams = list(zip(wordlist,wordlist[1:],wordlist[2:]))
        return list(zip(wordlist[1:-1],word_trigrams))

    def score_edge(self,Nroot,Nleft,Nright,i,j,k,sentence):
        """
        Scores an edge.
        @param Nroot,Nleft,Nright : the root node, the left node and
        the right node
        @param i,j,k: the i,j,k indexes of the clique to be scored
        @param sentence: the list of words to be parsed
        @return a perceptron score
        """
        edge_repr = self.__make_edge_representation(i,j,k,sentence)
        return self.model.dot(edge_repr,(Nroot,Nleft,Nright))
        
    def score_unary(self,Nroot,word_idx,sentence):
        """
        Scores an unary edge
        @param Nroot: the pos tag of the word
        @param word_idx: the index of the word in the sentence
        @param sentence: the sentence
        """
        unary_repr = self.__make_unary_representation(word_idx,sentence)
        return self.model.dot(unary_repr,Nroot)

    def __make_edge_representation(self,i,j,k,sentence):
        """
        Builds features for an edge.
        @param i,j,k: the i,j,k indexes of the clique to be scored
        """
        return [sentence[i],sentence[k-1],sentence[k],sentence[j-1]]

    def __make_unary_representation(self,word_idx,sentence):
        """
        Builds features for an unary reduction.
        """
        return [sentence[word_idx]]

    def __build_tree(self,root_vertex,tree_root,history,sentence):
        """
        Builds a parse tree from a chart history
        @param root_vertex: the tuple (i,j,label) encoding a vertex
        @param tree_root: the current ConsTree root
        @param history: the parse forest
        @param sentence: the list of words to be parsed
        @return the root of the tree
        """
        (i,k,labelL),(k,j,labelR) = history[root_vertex]
        left  = ConsTree(self.nonterminals_decode[labelL])
        right = ConsTree(self.nonterminals_decode[labelR])
        tree_root.children = [left,right]
        if k-i > 1: 
            self.__build_tree((i,k,labelL),left,history,sentence)
        else:
            left.add_child(ConsTree(sentence[i][0]))
        if j-k > 1:
            self.__build_tree((k,j,labelR),right,history,sentence)
        else:
            right.add_child(ConsTree(sentence[k][0]))
        return tree_root

    def __build_edges(self,root_vertex,history):
        """
        Extracts the hyperedges of the best tree from the chart and history
        (method useful for running the update while training)
        """
        left,right = history[root_vertex]
        result = [(root_vertex,left,right)]
        
        i,k,lblA = left
        k,j,lblB = right
        if k-i == 1:
            result.append((left,i))
        else:
            result.extend(self.__build_edges(left,history))
        if j-k == 1:
            result.append((right,k))
        else:
            result.extend(self.__build_edges(right,history))
        return result

    def parse_one(self,sentence,untransform=True,edges=False):
        """
        Parses a sentence with the cky viterbi algorithm
        @param sentence: a list of word strings
        @param untransform: if true, untransforms the result
        @param edges: if true returns hyperedges instead of a parse tree
        @return a ConsTree
        """
        N = len(sentence)
        G = len(self.nonterminals_decode) #num nonterminal symbols
        chart = np.empty([N,N+1,G])
        chart.fill(-inf)       # 0 for perceptron
        history = {}
        
        for i in range(N):#init (= part of speech tagging)
            for Nt in range(G):
                chart[i,i+1,Nt] = self.score_unary(Nt,i,sentence)
                                
        for span in range(2,N+1):#recurrence
            for i in range(N+1-span):
                j = i+span
                for Nt in range(G):
                    for k in range(i+1,j):
                        for Nt1 in range(G):
                            for Nt2 in range(G):
                                score = chart[i,k,Nt1]+chart[k,j,Nt2]+self.score_edge(Nt,Nt1,Nt2,i,j,k,sentence)
                                if score > chart[i,j,Nt]:
                                    chart[i,j,Nt] = score
                                    history[(i,j,Nt)] = ((i,k,Nt1),(k,j,Nt2))
                                    
        #Finds the max
        max_succ,argmax_succ = chart[0,N,0],(0,N,0)
        for Nt in range(1,G):
            if chart[0,N,Nt] > max_succ:
                max_succ,argmax_succ = chart[0,N,Nt],(0,N,Nt)
                
        i,j,label = argmax_succ
        if edges:
            return self.__build_edges(argmax_succ,history)
        else:#builds a ConsTree
            result =  self.__build_tree(argmax_succ,ConsTree(self.nonterminals_decode[label]),history,sentence)
            if untransform:
                result.unbinarize()
                result.expand_unaries()
            return result
    
    def tree_as_edges(self,tree_root):
        """
        Returns a list of hyperedges from a *binary* Constree
        Supposes that leaves a indexed by a field 'idx'
        @param tree_root: a constree
        @return : a list of hyperedges
        @see ConsTree.index_leaves(...) 
        """
        assert(len(tree_root.children) <= 2)
        if len(tree_root.children) == 1: #unary edges
            idx = tree_root.get_child().idx
            jdx = idx + 1
            return [((idx,jdx,self.nonterminals_code[tree_root.label]),idx)]

        elif len(tree_root.children) == 2:#binary edges
            left_edges = self.tree_as_edges(tree_root.children[0])
            right_edges = self.tree_as_edges(tree_root.children[1])
            i,k,lblA = left_edges[0][0]
            k,j,lblB = right_edges[0][0]
            result = [((i,j,self.nonterminals_code[tree_root.label]),(i,k,lblA),(k,j,lblB))]
            result.extend(left_edges)
            result.extend(right_edges)
            return result

    def test(self,treebank):
        """
        Tests a model against a treebank
        and returns the average F-score.
        """
        Fscore = []
        for ref_tree in treebank:
            xinput = ViterbiCKY.ngram_tokens(ref_tree.tokens())
            pred_tree = self.parse_one(xinput)
            print(pred_tree)
            P,R,F = ref_tree.compare(pred_tree)
            Fscore.append(F)
        return sum(Fscore)/len(Fscore)

    
    def train(self,treebank,step_size=1.0,max_epochs=100,left_markov=True):
        """
        Trains the parser with a structured perceptron
        @param: treebank a list of ConsTrees
        @param: left_markov :uses left markovization or right markovization 
        """
        
        self.transform(treebank,left_markov) #binarizes the treebank
        #makes a (x,y) pattern for the data set 
        dataset = list( [(ViterbiCKY.ngram_tokens(tree.tokens()),self.tree_as_edges(tree)) for tree in treebank] )
        
        N = len(dataset)
        
        for e in range(max_epochs):
            loss = 0.0
            for sentence,ref_edges in dataset:
                pred_edges = set(self.parse_one(sentence,edges=True))
                ref_edges  = set(ref_edges) 
                if pred_edges != ref_edges: #update
                    loss += 1.0
                    delta_ref = SparseWeightVector()
                    for r_edge in ref_edges:
                        if len(r_edge) == 3:
                            (i,j,Nroot),(i,k,Nl),(k,j,Nr) = r_edge
                            x_repr = self.__make_edge_representation(i,j,k,sentence)
                            delta_ref += SparseWeightVector.code_phi(x_repr,(Nroot,Nl,Nr))
                        elif len(r_edge) == 2:
                            (i,j,pos),widx = r_edge
                            x_repr = self.__make_unary_representation(widx,sentence)
                            delta_ref += SparseWeightVector.code_phi(x_repr,pos)

                    delta_pred = SparseWeightVector()
                    for p_edge in pred_edges:
                        if len(p_edge) == 3:
                            (i,j,Nroot),(i,k,Nl),(k,j,Nr) = p_edge
                            x_repr = self.__make_edge_representation(i,j,k,sentence)
                            delta_pred += SparseWeightVector.code_phi(x_repr,(Nroot,Nl,Nr))
                        elif len(p_edge) == 2:
                            (i,j,pos),widx = p_edge
                            x_repr = self.__make_unary_representation(widx,sentence)
                            delta_pred += SparseWeightVector.code_phi(x_repr,pos)

                    self.model += step_size * (delta_ref-delta_pred)
            print('Loss = ',loss, "%Local accurracy = ",(N-loss)/N)
            if loss == 0.0:
                return
  
            
x = ConsTree.read_tree('(S (NP (D le) (N chat)) (VN (V mange)) (NP (D la) (N souris)) (PP (P sur) (NP (D le) (N paillasson))) (PONCT .))')
y = ConsTree.read_tree('(S (NP (D la) (N souris)) (VN (V dort)) (PONCT .))')
z = ConsTree.read_tree('(S (NP (D le) (N cuisinier)) (VN (V mange)) (NP (D une) (N salade) (PP (P avec) (NP (D des) (N cornichons)))) (PONCT .))')

parser = ViterbiCKY()
parser.train([x,y,z],max_epochs=10,left_markov=False)
x = ConsTree.read_tree('(S (NP (D le) (N chat)) (VN (V mange)) (NP (D la) (N souris)) (PP (P sur) (NP (D le) (N paillasson))) (PONCT .))')
y = ConsTree.read_tree('(S (NP (D la) (N souris)) (VN (V dort)) (PONCT .))')
z = ConsTree.read_tree('(S (NP (D le) (N cuisinier)) (VN (V mange)) (NP (D une) (N salade) (PP (P avec) (NP (D des) (N cornichons)))) (PONCT .))')
print("F score = ",parser.test([x,y,z]))
