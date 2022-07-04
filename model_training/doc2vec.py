from gensim.models import doc2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.model_selection import train_test_split
from Bio import SeqIO
import numpy as np
import os

# break a sequence into k-mers
def seq_to_kmers(seq, k=3, overlap=True):
    N = len(seq)
    if overlap:
        return [[seq[i:i+k] for i in range(N - k + 1)]]
    else:
        return [[seq[i:i+k] for i in range(j, N - k + 1, k)]
                for j in range(k)]

# read fasta to kmers
def read_fasta_to_kmers(fasta_path, input_path=None, k=3, overlap=True):
    r = []
    for record in SeqIO.parse(fasta_path, 'fasta'):
        r += (seq_to_kmers(str(record.seq),k, overlap))
    return r

# pretrain doc2vec
def train_doc2vec(data,model_path):
    tagged_data = [TaggedDocument(words=_d, tags=[str(i)]) for i, _d in enumerate(data)]
    # model setting
    model = Doc2Vec(vector_size=50,window=25, min_count=2, epochs=50, workers=10,dm=0)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(model_path)
    return model

# encode kmers data by pre-trained doc2vecmodel
def encode_and_labels(pos, neg, model_path):
    model= Doc2Vec.load(model_path) 
    input_, answer = {}, {}
    for i in range(len(pos)):
        input_[i] = model.infer_vector(pos[i])
        answer[i] = 1
    for i in range(len(neg)):
        input_[i+len(pos)] = model.infer_vector(neg[i])
        answer[i+len(pos)] = 0
    data = np.array(list(input_.values())) 
    labels = np.array(list(answer.values()))
    return data, labels



