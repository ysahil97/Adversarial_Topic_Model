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
MODEL_PATH = "/home/sahil/deeplearning/ATM_GANs/ATM/models/model_1/"
EVAL_GENERATED_FOLDER = "/home/sahil/deeplearning/ATM_GANs/ATM/models/model_1/eval/generated_docs/"
COS_SIM_FOLDER = "/home/sahil/deeplearning/ATM_GANs/ATM/models/model_1/eval/cosine_similarity_analysis/"
device = torch.device("cuda")

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
BATCH_SIZE = 32
A_1 = 0.00015
B_1 = 0
B_2 = 0.9

torch.manual_seed(1)
use_cuda = True
vocab_text = util.create_vocab(vocab_file)

'''
Iterators for fake data used in Generator
'''
def inf_data_gen(alpha):
    if DATASET == "20newsgroups":
        while True:
            dataset = []
            for i in range(BATCH_SIZE):
                sample = np.random.dirichlet(alpha)
                dataset.append(sample)
            dataset = np.array(dataset, dtype='float32')
            np.random.shuffle(dataset)
            yield dataset

'''
Generator and Discriminator description of ATM-GAN's
'''
class generator(nn.Module):
    def __init__(self):
        super(generator,self).__init__()
        main = nn.Sequential(
               nn.Linear(NUM_TOPICS,GENERATOR_PARAM),
               nn.LeakyReLU(LEAK_FACTOR,True),
               nn.BatchNorm1d(GENERATOR_PARAM),
               nn.Linear(GENERATOR_PARAM,VOCAB_SIZE),
               nn.Softmax()
               )
        self.main = main

    def forward(self,noise):
        output = self.main(noise)
        return output


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator,self).__init__()
        main = nn.Sequential(
               nn.Linear(VOCAB_SIZE,GENERATOR_PARAM),
               nn.LeakyReLU(LEAK_FACTOR,True),
               nn.Linear(GENERATOR_PARAM,1))
        self.main = main

    def forward(self,inputs):
        output = self.main(inputs)
        return output.view(-1)

def fill_topic_report(mean_matrix,variance_matrix,num_topics):
    for ia in range(num_topics):
        filename = COS_SIM_FOLDER + "topic_"+str(ia+1)+"_report.txt"
        
        with open(filename, 'w') as myfile:
            doc_content = ""
            for j in range(len(mean_matrix[ia])):
                doc_content += "Topics "+str(ia+1)+" and "+str(j+1) + ":\nMean: "+str(mean_matrix[ia][j])+" Variance: "+str(variance_matrix[ia][j])+"\n\n"
            myfile.write(doc_content)

def top_val(vector,tops):
    t_list = list(enumerate(list(vector)))
    t_list.sort(key = lambda x : x[1])
    t_list.reverse()
    result_vector = [0]*2000
    for i in range(tops):
        result_vector[t_list[i][0]] = t_list[i][1]
    return result_vector

def generate_docs(alpha,num_topics,test_G, test_D):

    between_topics_mean = list()
    between_topics_variance = list()
    between_topics_vectors = []
    for ia in range(num_topics):
        cosine_scores = []
        alpha[ia] = 100
        data = inf_data_gen(alpha)
        _data = next(data)
        sampled_data = torch.Tensor(_data)
        if use_cuda:
            sampled_data = sampled_data.cuda()
        sampled_data_v = autograd.Variable(sampled_data)

        fake = test_G(sampled_data_v)
        fake_np = fake.tolist()
        text_data = util.tfidf2doc(fake_np,vocab_text)
        for doc_iter in range(32):
            doc_name = EVAL_GENERATED_FOLDER+"topic_"+str(ia+1)+"/doc_"+str(doc_iter)+".txt"
            with open(doc_name, 'w') as myfile:
                t_list = list(enumerate(list(fake_np[doc_iter])))
                t_list.sort(key = lambda x: x[1])
                t_list.reverse()
                doc_content = ""
                for i in range(len(t_list)):
                    doc_content += vocab_text[t_list[i][0]] + ' '
                myfile.write(doc_content)
        alpha[ia] = 1

        for first in range(31):
            for second in range(first+1,32):
                first_list =top_val(fake_np[first],100)
                second_list = top_val(fake_np[second],100)
                cosine_result = cos_sim(first_list,second_list)
                cosine_scores.append(cosine_result)
        print("Topic "+str(ia+1)+":\n")
        print("Mean of cosine scores: "+str(np.mean(cosine_scores))+"\nVariance of cosine scores: "+str(np.var(cosine_scores))+"\n\n")
        temp = []
        for i in range(32):
            temp.append(top_val(fake_np[i],100)) 
        between_topics_vectors.append(temp)

    for i in range(20):
        mean_vector = []
        variance_vector = []
        for j in range(20):
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
    fill_topic_report(between_topics_mean,between_topics_variance,num_topics)
    
    print(x)
    print(y)
                
test_G = generator()
test_D = discriminator()
test_G.load_state_dict(torch.load(MODEL_PATH+"atm_generator.pt"))
test_D.load_state_dict(torch.load(MODEL_PATH+"atm_discriminator.pt"))
test_G.to(device)
test_D.to(device)

alpha = [1]*20
generate_docs(alpha,20,test_G,test_D)
