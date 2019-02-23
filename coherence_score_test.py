import gensim.corpora as corpora
from gensim.test.utils import common_corpus, common_dictionary
from gensim.models.coherencemodel import CoherenceModel
import random
import os
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import numpy as np
import os
import pandas as pd
import gzip
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk import FreqDist,ngrams,word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import utility as util
import re

vocab_file = "/home/ysahil/Academics/Sem_8/ATM_GANs/20newsgroups_sakshi/data_20news/data/20news/vocab.new"
train_dataset = "/home/ysahil/Academics/Sem_8/ATM_GANs/20newsgroups_sakshi/data_20news/data/20news/train.feat"

topics_20ng = [
    [ 'sale' , 'driver' , 'wire' , 'card' , 'price' , "application" ,'software', 'monitor'],
    ['game' , 'team' , 'play' , 'player' , 'hockey' , 'season' ,  'league' , 'pittsburgh'],
    ['kill' , 'bike', 'live' , 'leave' , 'weapon' , 'happen' , 'gun', 'crime' , 'car' , 'hand'],
    # ['computer','windows','os','ms','hardware','file','ibm','machine'],
    ['space','nasa' , 'drive' , 'scsi' , 'orbit' , 'launch' ,'data' ,'control' , 'earth' ,'moon'],
    # ['armenian','people','war','israel','israeli','arab','jew','kill','turkish','attack'],
    # ['car','auto','drivers','bikes','motors','wheels'],
]


def create_dataset_coherencetest(data_url,vocab_text):
    vocab_length = 2000
    tf = TfidfTransformer()
    """process data input."""
    data = []
    word_count = []
    fin = open(data_url)
    docs = []
    while True:
        doc_b = []
        line = fin.readline()
        if not line:
            break
        id_freqs = line.split()

        doc = {}
        count = 0
        doc_a = [0 for i in range(0,2000)]
        doc_a = np.array(doc_a)
        for id_freq in id_freqs[1:]:
          items = id_freq.split(':')
          # python starts from 0
          doc_b.append(vocab_text[int(items[0])-1])
          doc_a[int(items[0])-1] = int(items[1])
          count += int(items[1])
        if count > 0:
          data.append(doc_a)
          word_count.append(count)
        docs.append(doc_b)
    return docs
    fin.close()
    np_matrix = np.array(data)
    tfidf = tf.fit_transform(np_matrix)
    n_response = tfidf.toarray()
    row_sum = n_response.sum(axis=1)
    length = len(row_sum)
    n_result = n_response/row_sum.reshape(length,1)
    position_NaNs = np.isnan(n_result)
    n_result[position_NaNs] = 0
    n_c_result = sparse.csr_matrix(n_result)
    return n_c_result


vocab_text = util.create_vocab(vocab_file)
all_docs = create_dataset_coherencetest(train_dataset,vocab_text)

id2word = corpora.Dictionary(all_docs)
corpus_newsgroups = [id2word.doc2bow(text) for text in all_docs]
cm_umass = CoherenceModel(topics=topics_20ng, corpus=corpus_newsgroups, dictionary=id2word, texts=all_docs, coherence='u_mass')
coherence_umass = cm_umass.get_coherence()  # get coherence value
cm_cv = CoherenceModel(topics=topics_20ng, corpus=corpus_newsgroups, dictionary=id2word, texts=all_docs, coherence='c_v')
coherence_cv = cm_cv.get_coherence()  # get coherence value
cm_cuci = CoherenceModel(topics=topics_20ng, corpus=corpus_newsgroups, dictionary=id2word, texts=all_docs, coherence='c_uci')
coherence_cuci = cm_cuci.get_coherence()  # get coherence value
cm_npmi = CoherenceModel(topics=topics_20ng, corpus=corpus_newsgroups, dictionary=id2word, texts=all_docs, coherence='c_npmi')
coherence_npmi = cm_npmi.get_coherence()  # get coherence value

print("Coherence(U_mass): "+str(coherence_umass))
print("Coherence(C_v):    "+str(coherence_cv))
print("Coherence(C_uci):  "+str(coherence_cuci))
print("Coherence(C_npmi): "+str(coherence_npmi))