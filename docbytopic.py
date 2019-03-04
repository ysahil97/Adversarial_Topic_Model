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

vocab_file = "/home/sahil/deeplearning/ATM_GANs/ATM/20newsgroups_sakshi/data_20news/data/20news/vocab.new"
MODEL_PATH = "/home/sahil/deeplearning/ATM_GANs/ATM/models/model_1/20newsgroups/"
device = torch.device("cuda")

'''
Important model parameters
'''
DATASET = "20newsgroups" # For now, we just test it on 20newsgroups dataset
#NUM_TOPICS = 20
LAMBDA = 10 # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 10 # For WGAN and WGAN-GP, number of critic iters per gen iter
ITERS = 40600 # How many generator iterations to train for
#VOCAB_SIZE = 2000# Vocab length of the generator
GENERATOR_PARAM = 100 # Number of neurons in the middle layer of the generator
LEAK_FACTOR = 0.2 # leak parameter used in generator
BATCH_SIZE = 128
A_1 = 0.00015
B_1 = 0
B_2 = 0.9

# Temporary change, needs to be changed later
dataset_path = "/home/sahil/deeplearning/ATM_GANs/ATM/20news/20news-18828/all_docs"
dataset_path_1 = "/home/sahil/deeplearning/ATM_GANs/ATM/20news/20news-18828"
if DATASET == "20newsgroups":
    train_dataset = "/home/sahil/deeplearning/ATM_GANs/ATM/20newsgroups_sakshi/data_20news/data/20news/combined.feat"
    vocab_file = "/home/sahil/deeplearning/ATM_GANs/ATM/20newsgroups_sakshi/data_20news/data/20news/vocab.new"
    MODEL_PATH = "/home/sahil/deeplearning/ATM_GANs/ATM/models/model_1/20newsgroups/"
    vocab_text = util.create_vocab(vocab_file)
    VOCAB_SIZE = 2000
    NUM_TOPICS = 4
elif DATASET == "grolier":
    MODEL_PATH = "/home/sahil/deeplearning/ATM_GANs/ATM/models/model_1/grolier/"
    train_dataset = "/home/sahil/deeplearning/ATM_GANs/ATM/datasets/grolier/result_dataset.feat"
    vocab_file = "/home/sahil/deeplearning/ATM_GANs/ATM/datasets/grolier/grolier15276_words.txt"
    vocab_text = util.create_vocab_grolier(vocab_file)
    VOCAB_SIZE = 15276
    NUM_TOPICS = 4




'''
topics_20ng = [
    ['alternative','atheism'],
    ['computer', 'graphics'],
    ['computer','os','ms','windows'],
    ['computer','sys','ibm','pc','hardware'],
    ['computer','sys','mac','hardware'],
    ['computer','windows'],
    ['misc','sale'],
    ['rec','auto'],
    ['rec','motorcycle'],
    ['rec','sport','baseball'],
    ['rec','sport','hockey'],
    ['science','crypto'],
    ['science','electronics'],
    ['science','medical'],
    ['science','space'],
    ['society','religion','christian'],
    ['talk','politics','guns'],
    ['talk','politics','middle','east'],
    ['talk','politics','misc'],
    ['talk','religion','misc']
]
'''
topics_20ng = [
    [ 'wire' , 'card' , 'price' , "application" ,'software', 'monitor'],
    [ 'team' , 'play' , 'player' , 'season' ,  'league' ],
    ['kill' , 'bike', 'live' , 'leave' , 'weapon' , 'happen' , 'gun', 'crime' , 'car' , 'hand'],
    # ['computer','windows','os','ms','hardware','file','ibm','machine'],
    ['space','nasa' , 'drive' , 'scsi' , 'orbit' , 'launch' ,'data' ,'control' , 'earth' ,'moon'],
    # ['armenian','people','war','israel','israeli','arab','jew','kill','turkish','attack'],
    # ['car','auto','drivers','bikes','motors','wheels'],
]


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

test_G = generator()
test_D = discriminator()
test_G.load_state_dict(torch.load(MODEL_PATH+"atm_generator.pt"))
test_D.load_state_dict(torch.load(MODEL_PATH+"atm_discriminator.pt"))
test_G.to(device)
test_D.to(device)


alpha = [0.1]*NUM_TOPICS
text_d = []
for ia in range(NUM_TOPICS):
    alpha[ia] = 100.1
    data = inf_data_gen(alpha)
    _data = next(data)
    sampled_data = torch.Tensor(_data)
    # input = input.to(device)
    if use_cuda:
        sampled_data = sampled_data.cuda()
    sampled_data_v = autograd.Variable(sampled_data)

    fake = test_G(sampled_data_v)
    fake_np = fake.tolist()
    text_data = util.tfidf2doc(fake_np,vocab_text)
    text_d += text_data
    for doc_iter in range(128):
        doc_name = "/home/sahil/deeplearning/ATM_GANs/ATM/ATM_testgen_docs/topic"+str(ia+1)+"/doc_"+str(doc_iter)+".txt"
        with open(doc_name, 'w') as myfile:
            t_list = list(enumerate(list(fake_np[doc_iter])))
            t_list.sort(key = lambda x: x[1])
            t_list.reverse()
            doc_content = ""
            for i in range(50):
                    doc_content += vocab_text[t_list[i][0]] + ' '
            myfile.write(doc_content)
    alpha[ia] = 0.1

id2word = corpora.Dictionary(text_d)
corpus_newsgroups = [id2word.doc2bow(text) for text in text_d]
cm_umass = CoherenceModel(topics=topics_20ng, corpus=corpus_newsgroups, dictionary=id2word, texts=text_d, coherence='u_mass')
coherence_umass = cm_umass.get_coherence()  # get coherence value
cm_cv = CoherenceModel(topics=topics_20ng, corpus=corpus_newsgroups, dictionary=id2word, texts=text_d, coherence='c_v')
coherence_cv = cm_cv.get_coherence()  # get coherence value
cm_cuci = CoherenceModel(topics=topics_20ng, corpus=corpus_newsgroups, dictionary=id2word, texts=text_d, coherence='c_uci')
coherence_cuci = cm_cuci.get_coherence()  # get coherence value
cm_npmi = CoherenceModel(topics=topics_20ng, corpus=corpus_newsgroups, dictionary=id2word, texts=text_d, coherence='c_npmi')
coherence_npmi = cm_npmi.get_coherence()  # get coherence value
print("Coherence(U_mass): "+str(coherence_umass))
print("Coherence(C_v):    "+str(coherence_cv))
print("Coherence(C_uci):  "+str(coherence_cuci))
print("Coherence(C_npmi): "+str(coherence_npmi))
