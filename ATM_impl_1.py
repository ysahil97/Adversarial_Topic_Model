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
use_cuda = False


'''
Important model parameters
'''
DATASET = "20newsgroups" # For now, we just test it on 20newsgroups dataset
NUM_TOPICS = 20
LAMBDA = 10 # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 10 # For WGAN and WGAN-GP, number of critic iters per gen iter
ITERS = 200000 # How many generator iterations to train for
VOCAB_SIZE = 2000# Vocab length of the generator
GENERATOR_PARAM = 100 # Number of neurons in the middle layer of the generator
LEAK_FACTOR = 0.2 # leak parameter used in generator
BATCH_SIZE = 512
A_1 = 0.00015
B_1 = 0
B_2 = 0.9

# Temporary change, needs to be changed later
dataset_path = "/home/ysahil/Academics/Sem_8/ATM_GANs/20news/20news-18828/all_docs"
dataset_path_1 = "/home/ysahil/Academics/Sem_8/ATM_GANs/20news/20news-18828"
train_dataset = "/home/ysahil/Academics/Sem_8/ATM_GANs/20newsgroups_sakshi/data_20news/data/20news/train.feat"
vocab_file = "/home/ysahil/Academics/Sem_8/ATM_GANs/20newsgroups_sakshi/data_20news/data/20news/vocab.new"

# alpha = [np.random.randint(1,11) for i in range(0,20)]
alpha = [0.1]*20
alpha[0] = 18.1
vocab_text = util.create_vocab(vocab_file)
# Topic list for Gensim Topic Coherence Pipeline
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
#Create the TF-IDF matrix
def get_tfidf():
    # vocab = util.create_vocab(dataset_path_1)
    result = util.create_dataset(train_dataset)
    return result

def representation_map(result):
    sam_doc = util.sample_document(result)
    return sam_doc

# Create normalized tfidf matrix (one time)
test_result = get_tfidf()


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

'''
Function to initialize one's weights
'''
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


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
Iterators for real data sampled from corpus
'''
def real_data_sampler(test_result):
    while True:
        dataset = []
        for i in range(BATCH_SIZE):
            sample = np.array(representation_map(test_result))
            dataset.append(sample[0])
        dataset = np.array(dataset, dtype='float32')
        np.random.shuffle(dataset)
        yield dataset


def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty.mean()


ATM_G = generator()
ATM_D = discriminator()
ATM_G.apply(weights_init)
ATM_D.apply(weights_init)
print(ATM_G)
print(ATM_D)


if use_cuda:
    ATM_D = ATM_D.cuda()
    ATM_G = ATM_G.cuda()

optimizerD = optim.Adam(ATM_D.parameters(), lr=A_1, betas=(B_1, B_2))
optimizerG = optim.Adam(ATM_G.parameters(), lr=A_1, betas=(B_1, B_2))

one = torch.FloatTensor([1])
mone = torch.FloatTensor([0])
if use_cuda:
    one = one.cuda()
    mone = mone.cuda()

data = inf_data_gen(alpha)
real_data = real_data_sampler(test_result)

for iteration in range(ITERS):
    ############################
    # (1) Update D network
    ###########################
    for p in ATM_D.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update

    # Flush out the gradients present in discriminator and generator
    optimizerD.zero_grad()
    optimizerG.zero_grad()
    for iter_d in range(CRITIC_ITERS):
        _data = next(data)
        sampled_data = torch.Tensor(_data)
        if use_cuda:
            sampled_data = sampled_data.cuda()
        sampled_data_v = autograd.Variable(sampled_data)

        fake = autograd.Variable(ATM_G(sampled_data_v).data)
        D_fake = ATM_D(fake)
        _realdata = next(real_data)
        sampled_real_data = torch.Tensor(_realdata)
        if use_cuda:
            sampled_real_data = sampled_real_data.cuda()
        sampled_real_data_v = autograd.Variable(sampled_real_data)

        D_real = ATM_D(sampled_real_data_v)

        gradient_penalty = calc_gradient_penalty(ATM_D, sampled_real_data_v.data, fake.data)

        D_cost = D_fake.mean() - D_real.mean() + gradient_penalty
        D_cost.backward()
        Wasserstein_D = D_fake.mean() - D_real.mean()
        optimizerD.step()

    for p in ATM_D.parameters():
        p.requires_grad = False  # to avoid computation
    ATM_G.zero_grad()

    _data = next(data)
    sampled_data = torch.Tensor(_data)
    if use_cuda:
        sampled_data = sampled_data.cuda()
    sampled_data_v = autograd.Variable(sampled_data)

    fake = ATM_G(sampled_data_v)
    G = ATM_D(fake)
    G_cost = -(G.mean())
    G_cost.backward()
    optimizerG.step()
    lib_plot.plot('/home/ysahil/Academics/Sem_8/ATM_GANs/' + DATASET + '/' + 'disc cost', D_cost.cpu().data.numpy())
    lib_plot.plot('/home/ysahil/Academics/Sem_8/ATM_GANs/' + DATASET + '/' + 'wasserstein distance', Wasserstein_D.cpu().data.numpy())
    lib_plot.plot('/home/ysahil/Academics/Sem_8/ATM_GANs/' + DATASET + '/' + 'gen cost', G_cost.cpu().data.numpy())
    if iteration % 100 == 99:
        print("Epoch %s\n" % iteration)
        lib_plot.flush()
    lib_plot.tick()

    if iteration % 500 == 99:
        '''
        Applying Gensim Topic Coherence Pipeline
        on the batch of documents, each having 1000 words
        ranked by their normalized tfidf values
        '''
        doc_name = "/home/ysahil/Academics/Sem_8/ATM_GANs/doc_gen/gen_doc_"+str(iteration)+".txt"
        print(fake.size())
        fake_np = fake.tolist()
        print(type(fake_np))
        text_data = util.tfidf2doc(fake_np,vocab_text)
        id2word = corpora.Dictionary(text_data)
        corpus_newsgroups = [id2word.doc2bow(text) for text in text_data]
        cm_umass = CoherenceModel(topics=topics_20ng, corpus=corpus_newsgroups, dictionary=id2word, texts=text_data, coherence='u_mass')
        coherence_umass = cm_umass.get_coherence()  # get coherence value
        cm_cv = CoherenceModel(topics=topics_20ng, corpus=corpus_newsgroups, dictionary=id2word, texts=text_data, coherence='c_v')
        coherence_cv = cm_cv.get_coherence()  # get coherence value
        cm_cuci = CoherenceModel(topics=topics_20ng, corpus=corpus_newsgroups, dictionary=id2word, texts=text_data, coherence='c_uci')
        coherence_cuci = cm_cuci.get_coherence()  # get coherence value
        cm_npmi = CoherenceModel(topics=topics_20ng, corpus=corpus_newsgroups, dictionary=id2word, texts=text_data, coherence='c_npmi')
        coherence_npmi = cm_npmi.get_coherence()  # get coherence value
        print("Coherence(U_mass): "+str(coherence_umass)+"\n")
        print("Coherence(C_v):    "+str(coherence_cv)+"\n")
        print("Coherence(C_uci):  "+str(coherence_cuci)+"\n")
        print("Coherence(C_npmi): "+str(coherence_npmi)+"\n")
        '''
        Document generation of a sample using top 100 words
        ranked by their normalized tf-idf values
        '''
        with open(doc_name, 'w') as myfile:
            t_list = list(enumerate(list(fake_np[0])))
            t_list.sort(key = lambda x: x[1])
            t_list.reverse()
            doc_content = ""
            for i in range(100):
                    doc_content += vocab_text[t_list[i][0]] + ' '
            myfile.write(doc_content)
