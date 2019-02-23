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
MODEL_PATH = "/home/sahil/deeplearning/ATM_GANs/ATM/models/model_3/"
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
BATCH_SIZE = 128
A_1 = 0.00015
B_1 = 0
B_2 = 0.9

topics_20ng = [
    [ 'sale' , 'driver' , 'wire' , 'card' , 'price' , "application" ,'software', 'monitor'],
    ['game' , 'team' , 'play' , 'player' , 'hockey' , 'season' ,  'league' , 'pittsburgh'],
    ['kill' , 'bike', 'live' , 'leave' , 'weapon' , 'happen' , 'gun', 'crime' , 'car' , 'hand'],
    # ['computer','windows','os','ms','hardware','file','ibm','machine'],
    ['space','nasa' , 'drive' , 'scsi' , 'orbit' , 'launch' ,'data' ,'control' , 'earth' ,'moon'],
    # ['armenian','people','war','israel','israeli','arab','jew','kill','turkish','attack'],
    # ['car','auto','drivers','bikes','motors','wheels'],
]

torch.manual_seed(1)
use_cuda = True
vocab_text = util.create_vocab(vocab_file)
# print(vocab_text)
# exit()

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
    doc_multi_name = MODEL_PATH + "results/result_saved_model_3.log"
    fake_data = [i.tolist() for i in fakes]
    topic_cnt = 0
    topic_coherence_store_umass = [list() for i in range(4)]
    topic_coherence_store_cv = [list() for i in range(4)]
    topic_coherence_store_cuci = [list() for i in range(4)]
    topic_coherence_store_cnpmi = [list() for i in range(4)]
    print(topic_coherence_store_umass)
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
            cm_cv = CoherenceModel(topics=topic, corpus=corpus_newsgroups, dictionary=id2word, texts=text_data, coherence='c_v')
            coherence_cv = cm_cv.get_coherence()  # get coherence value
            cm_cuci = CoherenceModel(topics=topic, corpus=corpus_newsgroups, dictionary=id2word, texts=text_data, coherence='c_uci')
            coherence_cuci = cm_cuci.get_coherence()  # get coherence value
            cm_npmi = CoherenceModel(topics=topic, corpus=corpus_newsgroups, dictionary=id2word, texts=text_data, coherence='c_npmi')
            coherence_npmi = cm_npmi.get_coherence()  # get coherence value
            #print(topic_coherence_store_umass[model_cnt])
            topic_coherence_store_umass[model_cnt].append(tuple((coherence_umass,topic_cnt)))
            topic_coherence_store_cv[model_cnt].append(tuple((coherence_cv,topic_cnt)))
            topic_coherence_store_cuci[model_cnt].append(tuple((coherence_cuci,topic_cnt)))
            topic_coherence_store_cnpmi[model_cnt].append(tuple((coherence_npmi,topic_cnt)))
            model_cnt += 1
        # print(topic_coherence_store_umass)
        topic_cnt += 1
    with open(doc_multi_name, 'w') as myfile:
        doc_content = ""
        doc_content += "U_MASS:\n"
        for i in range(len(topic_coherence_store_umass)):
            topic_coherence_store_umass[i].sort()

            topic_number =topic_coherence_store_umass[i][3][1]
            doc_content += "Generator_"+str(i)+": Topic "+str(topic_number)+"\n"
            for k in topic_coherence_store_umass[i]:
                doc_content += str(k[0])+","
            doc_content += "\n"
        doc_content += "\n"

        doc_content += "C_V:\n"

        for i in range(len(topic_coherence_store_umass)):
            topic_coherence_store_cv[i].sort()
            topic_number =topic_coherence_store_cv[i][3][1]
            doc_content += "Generator_"+str(i)+": Topic "+str(topic_number)+"\n"
            for k in topic_coherence_store_cv[i]:
                doc_content += str(k[0])+","
            doc_content += "\n"
        doc_content += "\n"
        doc_content += "C_UCI:\n"

        for i in range(len(topic_coherence_store_umass)):
            topic_coherence_store_cuci[i].sort()
            topic_number =topic_coherence_store_cuci[i][3][1]
            doc_content += "Generator_"+str(i)+": Topic "+str(topic_number)+"\n"
            for k in topic_coherence_store_cuci[i]:
                doc_content += str(k[0])+","
            doc_content += "\n"
        doc_content += "\n"
        doc_content += "NPMI:\n"

        for i in range(len(topic_coherence_store_umass)):
            topic_coherence_store_cnpmi[i].sort()
            topic_number =topic_coherence_store_cnpmi[i][3][1]
            doc_content += "Generator_"+str(i)+": Topic "+str(topic_number)+"\n"
            for k in topic_coherence_store_cnpmi[i]:
                doc_content += str(k[0])+","
            doc_content += "\n"
        doc_content += "\n"
        print(doc_content)
        myfile.write(doc_content)
    return

def get_top_10(fakes,vocab_text):
    doc_multi_name = MODEL_PATH + "results/result_saved_model_top10words.log"
    fake_data = [i.data for i in fakes]

    doc_content = ""
    for i in range(4):
        result_tensor = torch.einsum('ab->b',fake_data[i])
        x = result_tensor.tolist()
        sorted_tensor = [(y,z) for z, y in enumerate(x)]
        sorted_tensor.sort()
        doc_content += "Generator_"+str(i)+": "
        for k in range(50):
            doc_content += vocab_text[sorted_tensor[k][1]]+", "
        doc_content += '\n'

    with open(doc_multi_name, 'w') as myfile:
        myfile.write(doc_content)
    return

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
# test_D.to(device)


alpha = [0.1]*20
text_d = []


data = inf_data_gen()
alpha_data = inf_alpha_gen()
fakes = []
_data = next(alpha_data)
sampled_data = torch.Tensor(_data)
# print(sampled_data.size())
if use_cuda:
    sampled_data = sampled_data.cuda()
sampled_data_v = autograd.Variable(sampled_data)

result = alpha_g(sampled_data_v).data
#print(result)

for i in range(4):
    _data = next(data)
    sampled_data = torch.Tensor(_data)
    print(sampled_data.size())
    if use_cuda:
        sampled_data = sampled_data.cuda()
    sampled_data_v = autograd.Variable(sampled_data)
    # _data_v.append(sampled_data_v)
    fakes.append(autograd.Variable(generators[i](sampled_data_v).data))

get_top_10(fakes,vocab_text)
assign_topic_to_generator(fakes,topics_20ng)
# id2word = corpora.Dictionary(text_d)
# corpus_newsgroups = [id2word.doc2bow(text) for text in text_d]
# cm_umass = CoherenceModel(topics=topics_20ng, corpus=corpus_newsgroups, dictionary=id2word, texts=text_d, coherence='u_mass')
# coherence_umass = cm_umass.get_coherence()  # get coherence value
# cm_cv = CoherenceModel(topics=topics_20ng, corpus=corpus_newsgroups, dictionary=id2word, texts=text_d, coherence='c_v')
# coherence_cv = cm_cv.get_coherence()  # get coherence value
# cm_cuci = CoherenceModel(topics=topics_20ng, corpus=corpus_newsgroups, dictionary=id2word, texts=text_d, coherence='c_uci')
# coherence_cuci = cm_cuci.get_coherence()  # get coherence value
# cm_npmi = CoherenceModel(topics=topics_20ng, corpus=corpus_newsgroups, dictionary=id2word, texts=text_d, coherence='c_npmi')
# coherence_npmi = cm_npmi.get_coherence()  # get coherence value
# print("Coherence(U_mass): "+str(coherence_umass))
# print("Coherence(C_v):    "+str(coherence_cv))
# print("Coherence(C_uci):  "+str(coherence_cuci))
# print("Coherence(C_npmi): "+str(coherence_npmi))