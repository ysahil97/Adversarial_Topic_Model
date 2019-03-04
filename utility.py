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
import re

def remove_stopwords(data):
    stop_words = Counter(stopwords.words('english'))
    ans = []
    for each in data:
        if(each not in stop_words.keys()):
            ans.append(each)
    return ans

def lemmatizer(data):
    lmtzr = WordNetLemmatizer()
    ans = []
    for each in data:
        ans.append(lmtzr.lemmatize(each))
    return ans

def stemmer(data):
    ps = PorterStemmer()
    ans = []
    for each in data:
        ans.append(ps.stem(each))
    return ans

def cleanData(data):
    data = word_tokenize(data)
    data = lemmatizer(remove_stopwords(data))
    string = ' '.join(data)
    return data, string


def folder_count(path):
    count = 0
    l = []
    for f in os.listdir(path):
        child = os.path.join(path,f)
        if os.path.isdir(child):
            l.append(child)
            count += 1
    return count, l


def create_vocab(dataset_path):
    # vocab_file = os.path.join(dataset_path,"vocabulary.txt")
    data = []
    with open(dataset_path, 'r') as myfile:
        lines = myfile.readlines()
        for i in lines:
            data.append(i.split(' ')[0])#=myfile.read().replace('\n', ' ')
    return data

def create_vocab_grolier(dataset_path):
    data = []
    with open(dataset_path, 'r') as myfile:
        lines = myfile.readlines()
        for i in lines:
            data.append(i.strip('\n'))#=myfile.read().replace('\n', ' ')
    return data

def create_tfidf(dataset_path,vocab):
    list_docs = []
    vectorizer = TfidfVectorizer(stop_words='english',vocabulary=vocab,strip_accents='unicode')
    for f in os.listdir(dataset_path):
        child = os.path.join(dataset_path,f)
        with open(child, 'r', errors='ignore') as myfile:
            data=myfile.read().replace('\n', '')
        _ , final_data = cleanData(data)
        list_docs.append(final_data)
    response = vectorizer.fit_transform(list_docs)
    n_response = response.toarray()
    row_sum = n_response.sum(axis=1)
    length = len(row_sum)
    n_result = n_response/row_sum.reshape(length,1)
    position_NaNs = np.isnan(n_result)
    n_result[position_NaNs] = 0
    n_c_result = sparse.csr_matrix(n_result)
#     return response
    return n_c_result


def create_dataset(data_url,dataset):
    if dataset == "20newsgroups":
        vocab_length = 2000
    elif dataset == "grolier":
        vocab_length = 15276

    tf = TfidfTransformer()
    """process data input."""
    data = []
    word_count = []
    fin = open(data_url)
    while True:
        line = fin.readline()
        if not line:
            break
        id_freqs = line.split()

        doc = {}
        count = 0
        doc_a = [0 for i in range(0,vocab_length)]
        doc_a = np.array(doc_a)
        for id_freq in id_freqs[1:]:
          items = id_freq.split(':')
          # python starts from 0
          doc_a[int(items[0])-1] = int(items[1])
          count += int(items[1])
        if count > 0:
          data.append(doc_a)
          word_count.append(count)
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

def sample_document(tfidf_mat):
    tfidf_mat = tfidf_mat.transpose()
    _,num_docs = tfidf_mat.shape
    sampled_document = random.randint(0,num_docs-1)
    result = tfidf_mat.getcol(sampled_document).toarray().T
    return result

def tfidf2doc(tfidf_vecs,vocab):
    # tfidf_docs_list = list(tfidf_vecs)
    docs = []
    # print(tfidf_vecs)
    for i in tfidf_vecs:
        t_list = list(enumerate(i))
        t_list.sort(key = lambda x: x[1])
        t_list.reverse()
        doc = []
        # print(i)
        for j in range(1500):
            doc.append(vocab[t_list[j][0]])
        docs.append(doc)
    return docs
