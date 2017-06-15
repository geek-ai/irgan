#coding=utf-8
#! /usr/bin/env python3.4


import numpy as np
import os
import time
import datetime
import insurance_qa_data_helpers
import operator
from insurance_qa_data_helpers import encode_sent
import random
import pickle

import math
now = int(time.time()) 
        
timeArray = time.localtime(now)
timeStamp = time.strftime("%Y%m%d%H%M%S", timeArray)

#print( tf.__version__)

# Parameters
# ==================================================

import tensorflow as tf
import Discriminator
# Model Hyperparameters
tf.flags.DEFINE_integer("max_sequence_length", 200, "Max sequence length fo sentence (default: 200)")
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 500, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0000001, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 0.1, "learning_rate (default: 0.1)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 500000, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("pools_size", 100, "The sampled set of a positive ample, which is bigger than 500")
tf.flags.DEFINE_integer("sampled_size", 100, " the real selectd set from the The sampled pools")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")



FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


# Data Preparatopn
# ==================================================
# Load data
print(("Loading data..."))

vocab = insurance_qa_data_helpers.build_vocab()
# embeddings =insurance_qa_data_helpers.load_vectors(vocab)
alist = insurance_qa_data_helpers.read_alist()
raw = insurance_qa_data_helpers.read_raw()



test1List = insurance_qa_data_helpers.loadTestSet("test1")
test2List= insurance_qa_data_helpers.loadTestSet("test2")
devList= insurance_qa_data_helpers.loadTestSet("dev")
testSet=[("test1",test1List),("test2",test2List),("dev",devList)]


print("Load done...")

val_file = 'insuranceQA/test1'
precision = 'log/test1.dns'+timeStamp




from functools import wraps
def log_time_delta(func):
        @wraps(func)
        def _deco(*args, **kwargs):
                start = time.time()
                ret = func(*args, **kwargs)
                end = time.time()
                delta = end - start
                print( "%s runed %.2f seconds"% (func.__name__,delta))
                return ret
        return _deco

@log_time_delta
def generate_uniform_pair():
    samples=[]
    for pair in raw:
        q=pair[2]
        a=pair[3]
        index=random.randint(0, len(alist) - 1)
        neg= alist[index]
        
        samples.append([encode_sent(vocab,item, FLAGS.max_sequence_length) for item in [q,a,neg]])
    return samples

 

@log_time_delta    
def generate_dns_pair(sess, model):
    samples=[]
    for _index ,pair in enumerate (raw):
        if _index %100==0:
            print( "have sampled %d pairs" % _index)
        q=pair[2]
        a=pair[3]

        pools=np.random.choice(alist,size=[FLAGS.pools_size])    
    
        canditates=insurance_qa_data_helpers.loadCandidateSamples(q,a,pools,vocab)    
        predicteds=[]
        for batch in insurance_qa_data_helpers.batch_iter(canditates,batch_size=FLAGS.batch_size):                            
            feed_dict = {model.input_x_1: batch[:,0],model.input_x_2: batch[:,1],model.input_x_3: batch[:,2]}         
            predicted=sess.run(model.score13,feed_dict)
            predicteds.extend(predicted)        
        index=np.argmax(predicteds)
        samples.append([encode_sent(vocab,item, FLAGS.max_sequence_length) for item in [q,a,pools[index]]])    

    return samples


def dev_step(sess,cnn,testList,dev_size=100):
    scoreList = []
    if dev_size>len(testList)/500:
        dev_size=len(testList)/500
        print( "have test %d samples" % dev_size)
    for i in range(dev_size):
        batch_scores=[]
        for j in range(int(500/FLAGS.batch_size)):
            x_test_1, x_test_2, x_test_3 = insurance_qa_data_helpers.load_val_batch(testList, vocab, i*500+j*FLAGS.batch_size, FLAGS.batch_size)
            feed_dict = {
                cnn.input_x_1: x_test_1,
                cnn.input_x_2: x_test_2,    #x_test_2 equals x_test_3 for the test case
                cnn.input_x_3: x_test_3
                # cnn.dropout_keep_prob: 1.0
            }
            predicted =sess.run([cnn.score12], feed_dict)     # add the [] means the returned value is a list
         

            batch_scores.extend(predicted[0])
        max_indexs=[]    
        max_score=max(batch_scores)
        for index,s in enumerate(batch_scores):
            if s==max_score:
                max_indexs.append(index)

        index=    random.choice(max_indexs)
        if int(testList[i*500+index].split()[0])==1:
            scoreList.append(1)
        else:
            scoreList.append(0)
    return sum(scoreList) *1.0 /len(scoreList)


def evaluation(sess,model,log,num_epochs=0):
    current_step = tf.train.global_step(sess, model.global_step)
    precision_current=dev_step(sess,model,test1List,1800)
    line="test1: %d epoch: precision %f"%(current_step,precision_current)
    print (line)
    print( model.save_model(sess,precision_current))
    log.write(line+"\n")

    return 

def main():
    with tf.Graph().as_default():
        with tf.device("/gpu:0"):
            session_conf = tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement)
            sess = tf.Session(config=session_conf)
            with sess.as_default() ,open(precision,"w") as log:
                    # DIS_MODEL_FILE="model/Discriminator20170107122042.model"
                    # param = pickle.load(open(DIS_MODEL_FILE))
                    # print( param)
                    param= None
                    DIS_MODEL_FILE="model/pre-trained.model"
                    param = pickle.load(open(DIS_MODEL_FILE,"rb"))
                    discriminator = Discriminator.Discriminator(
                            sequence_length=FLAGS.max_sequence_length,
                            batch_size=FLAGS.batch_size,
                            vocab_size=len(vocab),
                            embedding_size=FLAGS.embedding_dim,
                            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                            num_filters=FLAGS.num_filters,
                            learning_rate=FLAGS.learning_rate,
                            l2_reg_lambda=FLAGS.l2_reg_lambda,
                            embeddings=None,
                            paras=param,
                            loss="pair")

                    saver = tf.train.Saver()    
                    sess.run(tf.global_variables_initializer())
                    # evaluation(sess,discriminator,log,0)

                    for i in range(FLAGS.num_epochs):
                        # x1,x2,x3=generate_dns(sess,discriminator)
                        # samples=generate_dns(sess,discriminator)#generate_uniform_pair() #generate_dns(sess,discriminator) #generate_uniform() #                        
                        samples=generate_dns_pair(sess,discriminator) #generate_uniform() # generate_uniform_pair() #                     
                        for j in range(1):
                            for batch in insurance_qa_data_helpers.batch_iter(samples,batch_size=FLAGS.batch_size,num_epochs=1,shuffle=True):    # try:                        
                                    
                                feed_dict = {
                                        discriminator.input_x_1: batch[:,0],
                                        discriminator.input_x_2: batch[:,1],
                                        discriminator.input_x_3: batch[:,2],
                                     
                                    }
                                
                             
                                _, step,    current_loss,accuracy = sess.run(
                                        [discriminator.train_op, discriminator.global_step, discriminator.loss,discriminator.accuracy],
                                        feed_dict)
                                time_str = datetime.datetime.now().isoformat()
                                print(("%s: DIS step %d, loss %f with acc %f "%(time_str, step, current_loss,accuracy)))
                                
                            evaluation(sess,discriminator,log,i)                        
if __name__ == '__main__':
    main()
    # print (embeddings)
                 
