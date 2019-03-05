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
import cos_sim_lib as scos


torch.manual_seed(1)
use_cuda = False

k=4

'''
Important model parameters
'''
DATASET = "20newsgroups" # For now, we just test it on 20newsgroups dataset
NUM_TOPICS = 20
NUM_NOISE = 3
LAMBDA = 10 # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 10 # For WGAN and WGAN-GP, number of critic iters per gen iter
ITERS = 40600 # How many generator iterations to train for
VOCAB_SIZE = 2000# Vocab length of the generator
GENERATOR_PARAM = 100 # Number of neurons in the middle layer of the generator
LEAK_FACTOR = 0.2 # leak parameter used in generator
BATCH_SIZE = 256
A_1 = 0.0015
A_1D = 0.002
B_1 = 0
B_2 = 0.9

# Temporary change, needs to be changed later
dataset_path = "/home/sahil/deeplearning/ATM_GANs/ATM/20news/20news-18828/all_docs"
dataset_path_1 = "/home/sahil/deeplearning/ATM_GANs/ATM/20news/20news-18828"
train_dataset = "/home/sahil/deeplearning/ATM_GANs/ATM/20newsgroups_sakshi/data_20news/data/20news/combined.feat"
vocab_file = "/home/sahil/deeplearning/ATM_GANs/ATM/20newsgroups_sakshi/data_20news/data/20news/vocab.new"
MODEL_PATH = "/home/sahil/deeplearning/ATM_GANs/ATM/models/model_3/"


vocab_text = util.create_vocab(vocab_file)
# Topic list for Gensim Topic Coherence Pipeline
topics_20ng = [
    ['alternative','atheism','religion','god','bible','christian','graphics'],
    ['computer','windows','os','ms','hardware','file','ibm','machine'],
    ['science','crypto','electronics','medical','electronics','medicine','space'],
    ['society','religion','christian','talk','politics','guns','middle','east'],
    # ['car','auto','drivers','bikes','motors','wheels'],
]

#Create the TF-IDF matrix
def get_tfidf():
    # vocab = util.create_vocab(dataset_path_1)
    result = util.create_dataset(train_dataset,"20newsgroups")
    return result

def representation_map(result):
    sam_doc = util.sample_document(result)
    return sam_doc

# Create normalized tfidf matrix (one time)
test_result = get_tfidf()


'''
Generator and Discriminator description of ATM-GAN's
'''

class shared_generator(nn.Module):
    def __init__(self):
        super(shared_generator,self).__init__()
        main = nn.Sequential(
               nn.Linear(NUM_TOPICS,GENERATOR_PARAM),
               nn.LeakyReLU(LEAK_FACTOR,True),
               )
        self.main = main
 
    def forward(self,noise):
        output = self.main(noise)
        return output

class generator(nn.Module):
    def __init__(self):
        super(generator,self).__init__()
        main = nn.Sequential(
#               nn.Linear(NUM_TOPICS,GENERATOR_PARAM),
#               nn.LeakyReLU(LEAK_FACTOR,True),
               nn.BatchNorm1d(GENERATOR_PARAM),
               nn.Linear(GENERATOR_PARAM,VOCAB_SIZE),
               nn.Softmax()
               )
        self.main = main

    def forward(self,noise):
        output = self.main(noise)
        return output

class alpha_generator(nn.Module):
    def __init__(self):
        super(alpha_generator,self).__init__()
        main = nn.Sequential(
               nn.Linear(NUM_TOPICS,10),
               nn.LeakyReLU(LEAK_FACTOR,True),
               nn.BatchNorm1d(10),
               nn.Linear(10,4),
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
                sample = np.random.normal(loc = 0.5,scale = 5.0, size=NUM_TOPICS)
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

def combine_alphas_generators(fakes,alphas):
    # TEMP: Testing alternative approach for linear combination rn
    # Thereby keeping both approaches intact for now
    alpha_slice = []
    alpha_tensor = alphas.data
    alpha_tensor.requires_grad = True
    result_tensor = torch.zeros(BATCH_SIZE,VOCAB_SIZE,requires_grad=True)
    result_t = result_tensor.clone()
    if use_cuda:
        result_tensor = result_tensor.cuda()
    for i in range(len(fakes)):
        x = alpha_tensor[:,i]
        x.unsqueeze_(-1)
        x = x.expand(BATCH_SIZE,VOCAB_SIZE)
        result_t += x*fakes[i]
    return autograd.Variable(result_tensor,requires_grad=True)

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


shared_g = shared_generator()

generators = []
for i in range(4):
    x = generator()
    x.apply(weights_init)
    generators.append(x)

ATM_D = discriminator()
alpha_g = alpha_generator()
ATM_D.apply(weights_init)
alpha_g.apply(weights_init)

if use_cuda:
    shared_g = shared_g.cuda()
    ATM_D = ATM_D.cuda()
    alpha_g = alpha_g.cuda()
    for g in generators:
        g = g.cuda()


optimizerD = optim.Adam(ATM_D.parameters(), lr=A_1D, betas=(B_1, B_2))
optimizerGs = []
optimizer_alpha = optim.Adam(alpha_g.parameters(), lr=A_1, betas=(B_1, B_2))
for i in range(4):
    optimizerGs.append(optim.Adam(generators[i].parameters(), lr=A_1, betas=(B_1, B_2)))

optimizer_shared = optim.Adam(shared_g.parameters(),lr = A_1, betas = (B_1,B_2))

one = torch.FloatTensor([1])
mone = torch.FloatTensor([0])
if use_cuda:
    one = one.cuda()
    mone = mone.cuda()

#alpha_data = inf_alpha_gen()
data = inf_data_gen()
real_data = real_data_sampler(test_result)

print(generators)
print(alpha_g)
print(ATM_D)

for iteration in range(ITERS):
    ############################
    # (1) Update D network
    ###########################
    for p in ATM_D.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update

    # Flush out the gradients present in discriminator and generator
    optimizerD.zero_grad()
    for og in optimizerGs:
        og.zero_grad()
    optimizer_alpha.zero_grad()
    optimizer_shared.zero_grad()
    for iter_d in range(CRITIC_ITERS):

        _alpha_data = next(data)
        _data = []
        for i in range(k):
            data_temp = next(data)
            sampled_data = torch.Tensor(data_temp)
            if use_cuda:
                sampled_data = sampled_data.cuda()
            sampled_data_v = autograd.Variable(sampled_data)
            _data.append(sampled_data_v)
        sampled_alpha_data = torch.Tensor(_alpha_data)
        if use_cuda:
            sampled_alpha_data = sampled_alpha_data.cuda()
        sampled_alpha_data_v = autograd.Variable(sampled_alpha_data)

        sampled_alphas = autograd.Variable(alpha_g(sampled_alpha_data_v).data)
        fakes = []
        for i in range(k):
            shared_output = autograd.Variable(shared_g(_data[i]))
            fakes.append(autograd.Variable(generators[i](shared_output).data))
        fake = combine_alphas_generators(fakes,sampled_alphas)
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


    for og in generators:
        og.zero_grad()
    alpha_g.zero_grad()
    shared_g.zero_grad()
    _data = []
    _alpha_data = next(data) 
    for i in range(k):
        data_temp = next(data)
        sampled_data = torch.Tensor(data_temp)
        if use_cuda:
            sampled_data = sampled_data.cuda()
        sampled_data_v = autograd.Variable(sampled_data)
        _data.append(sampled_data_v)

    sampled_alpha_data = torch.Tensor(_alpha_data)
    if use_cuda:
        sampled_alpha_data = sampled_alpha_data.cuda()
    sampled_alpha_data_v = autograd.Variable(sampled_alpha_data)

    sampled_alphas = autograd.Variable(alpha_g(sampled_alpha_data_v).data)
    fakes = []
    for i in range(k):
        shared_output = autograd.Variable(shared_g(_data[i]))
        fakes.append(autograd.Variable(generators[i](shared_output).data))
    fake = combine_alphas_generators(fakes,sampled_alphas)
    G = ATM_D(fake)
    G_cost = -(G.mean())
    G_cost.backward()
    optimizer_alpha.step()
    for og in optimizerGs:
        og.step()
    optimizer_shared.step()

    '''
    for og in optimizerGs:
        og.zero_grad()
    optimizer_shared.zero_grad()
    m = nn.Sigmoid()
    G_1 = m(ATM_D(fake))
    G_cost_1 = autograd.Variable((-1*G),requires_grad = True).mean()

    G_2 = m(ATM_D(fake))
    #G_cost_2 = autograd.Variable(-1*G_2,requires_grad = True).mean()*autograd.Variable((1-G_1),requires_grad = True).mean()
    G_cost_2 = autograd.Variable(-1*G,requires_grad = True).mean()#*autograd.Variable((1-G_1),requires_grad = True).mean()

    G_3 = m(ATM_D(fake))
    norm_3 = torch.reciprocal(torch.add(torch.reciprocal(1-G_1),torch.reciprocal(1-G_2)))*2
    #G_cost_3 = autograd.Variable(-1*G_3,requires_grad=True).mean()*autograd.Variable(norm_3,requires_grad=True).mean()
    G_cost_3 = autograd.Variable(-1*G,requires_grad=True).mean()#*autograd.Variable(norm_3,requires_grad=True).mean()

    G_4 = m(ATM_D(fake))
    norm_4 = torch.reciprocal(torch.add(torch.add(torch.reciprocal(1-G_1),torch.reciprocal(1-G_2)),torch.reciprocal(1-G_3)))*3
    #G_cost_4 = autograd.Variable(-1*G_4,requires_grad = True).mean()*autograd.Variable(norm_4,requires_grad=True).mean()
    G_cost_4 = autograd.Variable(-1*G,requires_grad = True).mean()#*autograd.Variable(norm_4,requires_grad=True).mean()

    G_cost_1.backward()
    G_cost_2.backward()
    G_cost_3.backward()
    G_cost_4.backward()

    for og in optimizerGs:
        og.step()
    optimizer_shared.step()
    '''
    lib_plot.plot(MODEL_PATH+'plots/disc cost',D_cost.cpu().data.numpy())
    lib_plot.plot(MODEL_PATH+'plots/wasserstein distance',Wasserstein_D.cpu().data.numpy())
    lib_plot.plot(MODEL_PATH+'plots/gen cost',G_cost.cpu().data.numpy())
    #lib_plot.plot(MODEL_PATH+'plots/gen 1 cost',G_cost_1.cpu().data.numpy())
    #lib_plot.plot(MODEL_PATH+'plots/gen 2 cost',G_cost_2.cpu().data.numpy())
    #lib_plot.plot(MODEL_PATH+'plots/gen 3 cost',G_cost_3.cpu().data.numpy())
    #lib_plot.plot(MODEL_PATH+'plots/gen 4 cost',G_cost_4.cpu().data.numpy())
    if iteration % 100 == 99:
        print("Epoch %s\n" % iteration)
        lib_plot.flush()
    lib_plot.tick()

    if iteration % 500 == 99:
        '''
        Map the current state of each generator to the topic label learnt
        '''
        assign_topic_to_generator(fakes,topics_20ng)
        '''
        Applying Gensim Topic Coherence Pipeline
        on the batch of documents, each having 1500 words
        ranked by their normalized tfidf values
        '''
        doc_name = MODEL_PATH + "generated_docs/iteration_" + str(iteration)+".txt"
        print(fake.size())
        fake_np = fake.tolist()
        print(type(fake_np))
        '''
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
        print("Coherence(U_mass): "+str(coherence_umass))
        print("Coherence(C_v):    "+str(coherence_cv))
        print("Coherence(C_uci):  "+str(coherence_cuci))
        print("Coherence(C_npmi): "+str(coherence_npmi))
        '''
        '''
        Document generation of a sample using top 100 words
        ranked by their normalized tf-idf values
        '''
        for i in range(k):
            gen_doc_name = MODEL_PATH+"generated_docs/topic_"+str(i+1)+"/doc_"+str(iteration)+".txt"
            with open(gen_doc_name,'w') as genfile:
                fake_gen_np = fakes[i].tolist()
                t_list = list(enumerate(list(fake_gen_np[0])))
                t_list.sort(key = lambda x: x[1])
                t_list.reverse()
                doc_content = ""
                for i in range(20):
                    doc_content += vocab_text[t_list[i][0]] + ': ' +  str(t_list[i][1]) + '\n'
                genfile.write(doc_content)
        scos.cos_similarity(fakes,k)
        with open(doc_name, 'w') as myfile:
            t_list = list(enumerate(list(fake_np[0])))
            t_list.sort(key = lambda x: x[1])
            t_list.reverse()
            doc_content = ""
            for i in range(100):
                    doc_content += vocab_text[t_list[i][0]] + ' '
            myfile.write(doc_content)
        
        torch.save(alpha_g.state_dict(), MODEL_PATH+"atm_alpha_generator.pt")
        for g in range(len(generators)):
            torch.save(generators[g].state_dict(),MODEL_PATH+"atm_generator_"+str(g)+".pt")
        torch.save(ATM_D.state_dict(), MODEL_PATH+"atm_discriminator.pt")
        torch.save(shared_g.state_dict(),MODEL_PATH+"atm_shared_generator.pt")
