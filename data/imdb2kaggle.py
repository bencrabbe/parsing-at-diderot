import os
import os.path
import random
"""
This script turns the raw stanford ImdB data set into a kaggle
compliant format for running as a kaggle competition
"""
def load_dataset(dir_path,ref_label):
    """
    Loads a dataset from a directory path (positive or negative) and 
    returns a list of couples (Counter of Bow_freq,ref_label) one for each text
    """
    dpath    = os.path.abspath(dir_path)
    data_set = [] 
    for f in os.listdir(dpath):
        filepath      = os.path.join(dpath,f)
        file_stream   = open(filepath)
        #normalize spaces and removes tabs
        text          = file_stream.read()
        file_stream.close()
        data_set.append((text,ref_label))
    return data_set


def generate_train_data(filename,datalines):
    ostream = open(filename,'w')
    for idx,line in enumerate(datalines):
        text,label  = line        
        print(','.join([label,text]),file=ostream)
    ostream.close()

def generate_student_test(filename,datalines):
    ostream = open(filename,'w')
    print('idx,text',file=ostream)
    for idx,line in enumerate(test):
        text, label  = line 
        print('%d,%s'%(idx,text),file=ostream)
    ostream.close()

def generate_sample(filename,datalines):
    ostream = open(filename,'w')
    print('idx,sentY',file=ostream)
    for idx,line in enumerate(test):
        text, label  = line 
        print('%d,%s'%(idx,1),file=ostream)
    ostream.close()
    
def generate_kaggle_solution(filename,datalines):
    ostream = open(filename,'w')
    print('idx,sentY',file=ostream)
    for idx,line in enumerate(test):
        text,y1  = line
        print(','.join([str(idx),y1]),file=ostream)
    ostream.close()

if __name__ == '__main__':

    trainpos = load_dataset("aclImdb/train/pos","1")
    trainneg = load_dataset("aclImdb/train/neg","0")
    train = trainpos + trainneg
    random.shuffle(train)
    generate_train_data('sentimentIMDB_train.csv',train)

    testpos = load_dataset("aclImdb/test/pos","1")
    testneg = load_dataset("aclImdb/test/neg","0")
    test = testpos + testneg
    random.shuffle(test)
    
    generate_student_test('sentimentIMDB_test.csv',test)
    generate_kaggle_solution('sentimentIMDB_solution.csv',test)
    generate_sample('sentimentIMDB_sample.csv',test)
