import numpy as np
import random
import math


def build_vocab():
    code = int(0)
    vocab = {}
    vocab['UNKNOWN'] = code
    code += 1
    for filename in ['insuranceQA/train','insuranceQA/test1','insuranceQA/test2','insuranceQA/dev']:
        for line in open(filename):
            items = line.strip().split(' ')
            for i in range(2, 3):
                words = items[i].split('_')
                for word in words:
                    if not word in vocab:
                        vocab[word] = code
                        code += 1
    return vocab



def read_alist():
    alist = []
    for line in open('insuranceQA/train'):
        items = line.strip().split(' ')
        alist.append(items[3])
    print('read_alist done ......')
    return alist



def load_vectors(vocab=None):
    if vocab==None:
        return
    vectors = {}
    for line in open('insuranceQA/vectors.nobin'):
        items = line.strip().split(' ')
        if (len(items) < 101):
            continue
        vec = []
        for i in range(1, 101):
            vec.append(float(items[i]))
        vectors[items[0]] = vec
    embeddings=[]
    for word in vocab:
        if word in vectors.keys():
            embeddings.append(vectors[word])
        else:
            random_embeding=np.random.uniform(-1,1,100).tolist()
            embeddings.append(random_embeding)
    return embeddings




def loadTestSet(filename):
    testList = []
    for line in open('insuranceQA/'+filename):
        testList.append(line.strip())    # lower?
    return testList

def loadCandidateSamples(q,a,candidates,vocab):    
    samples=[]
    
    for neg in candidates:
        samples.append((encode_sent(vocab,q, 200),encode_sent(vocab,a, 200),encode_sent(vocab,neg, 200)))
    return  samples


def read_raw():
    raw = []
    for line in open('insuranceQA/train'):
        items = line.strip().split(' ')
        if items[0] == '1':
            raw.append(items)
    return raw

def encode_sent(vocab, string, size):
    x = []
    words = string.split('_')
    for i in range( size ):
        if words[i] in vocab:
            x.append(vocab[words[i]])
        else:
            x.append(vocab['UNKNOWN'])
    return x

def load_train_random(vocab, alist, raw, size):
    x_train_1 = []
    x_train_2 = []
    x_train_3 = []
    for items in raw:
        items = raw[random.randint(0, len(raw) - 1)]
        nega = np.random.choice(alist)
        x_train_1.append(encode_sent(vocab, items[2], 200))
        x_train_2.append(encode_sent(vocab, items[3], 200))
        x_train_3.append(encode_sent(vocab, nega, 200))
    return np.array(x_train_1), np.array(x_train_2), np.array(x_train_3)
def load_val_batch(testList, vocab, index, batch):
    x_train_1 = []
    x_train_2 = []
    x_train_3 = []
    for i in range(0, batch):
        true_index = index + i
        if (true_index >= len(testList)):
            true_index = len(testList) - 1
        items = testList[true_index].split(' ')
        x_train_1.append(encode_sent(vocab, items[2], 200))
        x_train_2.append(encode_sent(vocab, items[3], 200))
        x_train_3.append(encode_sent(vocab, items[3], 200))
    return np.array(x_train_1), np.array(x_train_2), np.array(x_train_3)

def batch_iter(data, batch_size, num_epochs=1, shuffle=False):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch =int( math.ceil(len(data)/batch_size))
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            # start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[end_index-batch_size:end_index]
    
def main():
    
    vocab = build_vocab()
    embeddings =load_vectors(vocab)
    alist = read_alist()
    raw = read_raw()
    test1List = loadTestSet("test1")
    # test2List= loadTestSet("test2")
    # devList= loadTestSet("dev")
    # testSet=[("test1",test1List),("test2",test2List),("dev",devList)]
    
if __name__ == '__main__':
    main()
   