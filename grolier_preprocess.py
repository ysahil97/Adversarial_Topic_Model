import gensim.corpora as corpora
from gensim.test.utils import common_corpus, common_dictionary
from gensim.models.coherencemodel import CoherenceModel
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utility as util
import lib_plot


torch.manual_seed(1)
use_cuda = True


'''
Important model parameters
'''
DATASET = "20newsgroups" # For now, we just test it on 20newsgroups dataset
NUM_TOPICS = 20
LAMBDA = 10 # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 10 # For WGAN and WGAN-GP, number of critic iters per gen iter
ITERS = 40600 # How many generator iterations to train for
VOCAB_SIZE = 2000# Vocab length of the generator
GENERATOR_PARAM = 100 # Number of neurons in the middle layer of the generator
LEAK_FACTOR = 0.2 # leak parameter used in generator
BATCH_SIZE = 512
A_1 = 0.00015
B_1 = 0
B_2 = 0.9

# Temporary change, needs to be changed later
dataset_path = "/home/sahil/deeplearning/ATM_GANs/ATM/20news/20news-18828/all_docs"
dataset_path_1 = "/home/sahil/deeplearning/ATM_GANs/ATM/20news/20news-18828"
result_dataset = "/home/sahil/deeplearning/ATM_GANs/ATM/datasets/grolier/result_dataset.feat"
train_dataset = "/home/sahil/deeplearning/ATM_GANs/ATM/datasets/grolier/grolier15276.csv"
vocab_file = "/home/sahil/deeplearning/ATM_GANs/ATM/20newsgroups_sakshi/data_20news/data/20news/vocab.new"
MODEL_PATH = "/home/sahil/deeplearning/ATM_GANs/ATM/models/model_1/"

alpha = [1]*20
vocab_text = util.create_vocab(vocab_file)

def preprocess_dataset():
    with open(result_dataset, 'w') as myfile:
        doc_content = ""
        with open(train_dataset, 'r') as readfile:
            for f in readfile.readlines():
                orig_line = f.strip('\n').split(',')
                print(orig_line[0])
                # Removing empty articles in grolier dataset
                if orig_line[1] == '' and orig_line[2] == '':
                    continue
                else:
                    doc_content += str(orig_line[0])
                    for j in range(1,len(orig_line),2):
                        doc_content += ' '+str(orig_line[j])+':'+str(orig_line[j+1])
                doc_content += '\n'
        myfile.write(doc_content)
    
preprocess_dataset()
exit()
#Create the TF-IDF matrix
def get_tfidf():
    result = util.create_dataset(train_dataset)
    return result

test_result = get_tfidf()




