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
        ordinal_value = f.split('.')[0].split('_')[1]
        filepath      = os.path.join(dpath,f)
        file_stream   = open(filepath)
        #normalize spaces and removes tabs
        text          = ' '.join(file_stream.read().split())
        file_stream.close()
        data_set.append((text,ref_label,ordinal_value))
    return data_set

testpos = load_dataset("aclImdb/test/pos","1")
testneg = load_dataset("aclImdb/test/neg","0")
test = testpos + testneg
random.shuffle(test)

ostream = open('sentimentIMDB_test.csv','w')
print('text,bool_sentiment,scaled_sentiment',file=ostream)
for line in test:
    print(','.join(line),file=ostream)
ostream.close()
    
