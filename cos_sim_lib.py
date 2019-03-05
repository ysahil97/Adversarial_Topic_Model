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
from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm

def cos_sim(a,b):
    return dot(a,b)/(norm(a)*norm(b))


vocab_file = "/home/sahil/deeplearning/ATM_GANs/ATM/20newsgroups_sakshi/data_20news/data/20news/vocab.new"
MODEL_PATH = "/home/sahil/deeplearning/ATM_GANs/ATM/models/model_3/"
EVAL_GENERATED_FOLDER = "/home/sahil/deeplearning/ATM_GANs/ATM/models/model_3/eval/generated_docs/"
COS_SIM_FOLDER = "/home/sahil/deeplearning/ATM_GANs/ATM/models/model_3/eval/cosine_similarity_analysis/"
device = torch.device("cuda")

'''
Important model parameters
'''
NUM_NOISE = 3
DATASET = "20newsgroups" # For now, we just test it on 20newsgroups dataset
NUM_TOPICS = 4
LAMBDA = 10 # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 10 # For WGAN and WGAN-GP, number of critic iters per gen iter
ITERS = 40600 # How many generator iterations to train for
VOCAB_SIZE = 2000# Vocab length of the generator
GENERATOR_PARAM = 100 # Number of neurons in the middle layer of the generator
LEAK_FACTOR = 0.2 # leak parameter used in generator
BATCH_SIZE = 32
A_1 = 0.00015
B_1 = 0
B_2 = 0.9


torch.manual_seed(1)
use_cuda = True
vocab_text = util.create_vocab(vocab_file)


def top_val(vector,tops):
    t_list = list(enumerate(list(vector)))
    t_list.sort(key = lambda x : x[1])
    t_list.reverse()
    result_vector = [0]*2000
    for i in range(tops):
        result_vector[t_list[i][0]] = t_list[i][1]
    return result_vector


def fill_topic_report(mean_matrix,variance_matrix,num_topics):
    for ia in range(num_topics):
        filename = COS_SIM_FOLDER + "topic_"+str(ia+1)+"_report.txt"

        with open(filename, 'w') as myfile:
            doc_content = ""
            for j in range(len(mean_matrix[ia])):
                doc_content += "Topics "+str(ia+1)+" and "+str(j+1) + ":\nMean: "+str(mean_matrix[ia][j])+" Variance: "+str(variance_matrix[ia][j])+"\n\n"
            myfile.write(doc_content)

def cos_similarity(fakes,num_topics):
    between_topics_mean = list()
    between_topics_variance = list()
    between_topics_vectors = []
    for topic in range(num_topics):
        temp = []
        fake_np = fakes[topic].tolist()
        for i in range(32):
            temp.append(top_val(fake_np[i],50))
        between_topics_vectors.append(temp)

    for i in range(4):
        mean_vector = []
        variance_vector = []
        for j in range(4):
           # if i != j:
             temp = list()
             for k in range(32):
                 for l in range(32):
                     if k != l:
                         x = cos_sim(between_topics_vectors[i][k],between_topics_vectors[j][l])
                         temp.append(x)
             mean_vector.append(np.mean(temp))
             variance_vector.append(np.var(temp))
        between_topics_mean.append(mean_vector)
        between_topics_variance.append(variance_vector)
    x = np.array(between_topics_mean)
    y = np.array(between_topics_variance)
    print(x)
    print(y)
    fill_topic_report(between_topics_mean,between_topics_variance,num_topics)
    #print(x)
    #print(y)

