import utility as util
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

tfidf = TfidfVectorizer(stop_words='english',strip_accents='unicode')

train_dataset = "/home/ysahil/Academics/Sem_8/ATM_GANs/20newsgroups_sakshi/data_20news/data/20news/train.feat"
res, params = util.create_dataset(train_dataset)
print(res)

vocab_file = "/home/ysahil/Academics/Sem_8/ATM_GANs/20newsgroups_sakshi/data_20news/data/20news/vocab.new"
vocab = util.create_vocab(vocab_file)
print(vocab)
exit()
x = res[45,:]
print(params)
tfidf.set_params(params)
vocab = tfidf.inverse_transform(x)
print(vocab)