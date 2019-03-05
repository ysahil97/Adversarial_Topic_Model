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

def inf_alpha_gen():
    if DATASET == "20newsgroups":
        while True:
            dataset = []
            for i in range(BATCH_SIZE):
                sample = np.random.normal(loc = 0.5,size=NUM_NOISE)
                dataset.append(sample)
            dataset = np.array(dataset, dtype='float32')
            np.random.shuffle(dataset)
            yield dataset


'''
Iterators for fake data used in Generator
'''
def inf_data_gen():
    if DATASET == "20newsgroups":
        while True:
            dataset = []
            for i in range(BATCH_SIZE):
                sample = np.random.normal(loc = 0.5,size=NUM_TOPICS)
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

class alpha_generator(nn.Module):
    def __init__(self):
        super(alpha_generator,self).__init__()
        main = nn.Sequential(
               nn.Linear(NUM_NOISE,10),
               nn.LeakyReLU(LEAK_FACTOR,True),
               nn.BatchNorm1d(10),
               nn.Linear(10,4),
               nn.Softmax()
        )
        self.main = main

    def forward(self,noise):
        output = self.main(noise)
        return output


def assign_topic_to_generator(fakes,topics_20ng):
    '''
    Assuming U_mass score for now
    '''
    doc_multi_name = MODEL_PATH + "results/result.log"
    fake_data = [i.tolist() for i in fakes]
    topic_cnt = 0
    topic_coherence_store = [list() for i in range(4)]
    for i in topics_20ng:
        topic = []
        topic.append(i)
        model_cnt = 0
        for j in fake_data:
            text_data = util.tfidf2doc(j,vocab_text)
            id2word = corpora.Dictionary(text_data)
            corpus_newsgroups = [id2word.doc2bow(text) for text in text_data]
            cm_umass = CoherenceModel(topics=topic, corpus=corpus_newsgroups, dictionary=id2word, texts=text_data, coherence='u_mass')
            coherence_umass = cm_umass.get_coherence()  # get coherence value
            topic_coherence_store[model_cnt].append(tuple((coherence_umass,topic_cnt)))
            model_cnt += 1
        topic_cnt += 1

    with open(doc_multi_name, 'w') as myfile:
        doc_content = ""
        for i in range(len(topic_coherence_store)):
            topic_coherence_store[i].sort()
            topic_number =topic_coherence_store[i][3][1]
            doc_content += "Generator_"+str(i)+": Topic "+str(topic_number)+"\n"
        myfile.write(doc_content)

    return


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

def generate_docs(num_topics,alpha_g, generators):

    between_topics_mean = list()
    between_topics_variance = list()
    between_topics_vectors = []
    for ia in range(num_topics):
        cosine_scores = []
#        alpha[ia] = 100
        data = inf_data_gen()
        _data = next(data)
        sampled_data = torch.Tensor(_data)
        if use_cuda:
            sampled_data = sampled_data.cuda()
        sampled_data_v = autograd.Variable(sampled_data)

        fake = generators[ia](sampled_data_v)
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
 #       alpha[ia] = 1

        for first in range(31):
            for second in range(first+1,32):
                first_list =top_val(fake_np[first],50)
                second_list = top_val(fake_np[second],50)
                cosine_result = cos_sim(first_list,second_list)
                cosine_scores.append(cosine_result)
        print("Topic "+str(ia+1)+":\n")
        print("Mean of cosine scores: "+str(np.mean(cosine_scores))+"\nVariance of cosine scores: "+str(np.var(cosine_scores))+"\n\n")
        temp = []
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
    fill_topic_report(between_topics_mean,between_topics_variance,num_topics)

    print(x)
    print(y)

def cos_similarity(fakes,k):
    between_topics_mean = list()
    between_topics_variance = list()
    between_topics_vectors = []
    for topic in range(k):
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
    fill_topic_report(between_topics_mean,between_topics_variance,k)
    print(x)
    print(y)

'''                
test_G = generator()
test_D = discriminator()
test_G.load_state_dict(torch.load(MODEL_PATH+"atm_generator.pt"))
test_D.load_state_dict(torch.load(MODEL_PATH+"atm_discriminator.pt"))
test_G.to(device)
test_D.to(device)
'''
generators = []
test_G_1 = generator()
test_G_2 = generator()
test_G_3 = generator()
test_G_4 = generator()
# test_D = discriminator()
test_G_1.load_state_dict(torch.load(MODEL_PATH+"atm_generator_0.pt"))
test_G_2.load_state_dict(torch.load(MODEL_PATH+"atm_generator_1.pt"))
test_G_3.load_state_dict(torch.load(MODEL_PATH+"atm_generator_2.pt"))
test_G_4.load_state_dict(torch.load(MODEL_PATH+"atm_generator_3.pt"))
# test_D.load_state_dict(torch.load(MODEL_PATH+"atm_discriminator.pt"))
test_G_1.to(device)
test_G_2.to(device)
test_G_3.to(device)
test_G_4.to(device)
generators.append(test_G_1)
generators.append(test_G_2)
generators.append(test_G_3)
generators.append(test_G_4)
alpha_g = alpha_generator()
alpha_g.load_state_dict(torch.load(MODEL_PATH+"atm_alpha_generator.pt"))
alpha_g.to(device)


#alpha = [1]*20
generate_docs(4,alpha_g,generators)

