#coding=utf-8
import tensorflow as tf 
import numpy as np 
import  pickle
import time
class QACNN():
    
    def __init__(self, sequence_length, batch_size,vocab_size, embedding_size,filter_sizes, num_filters, dropout_keep_prob=1.0,l2_reg_lambda=0.0,paras=None,learning_rate=1e-2,embeddings=None,loss="pair",trainable=True):
        self.sequence_length=sequence_length
        self.learning_rate=learning_rate
        self.paras=paras
        self.filter_sizes=filter_sizes
        self.num_filters=num_filters
        self.l2_reg_lambda=l2_reg_lambda
        self.dropout_keep_prob = dropout_keep_prob
        self.embeddings=embeddings


        self.embedding_size=embedding_size
        self.batch_size=batch_size
        self.model_type="base"
        self.num_filters_total=self.num_filters * len(self.filter_sizes)

        self.input_x_1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_1")
        self.input_x_2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_2")
        self.input_x_3 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_3")
        
        self.label=tf.placeholder(tf.float32, [batch_size], name="input_x_3")
        
        # Embedding layer
        self.updated_paras=[]
        with tf.name_scope("embedding"):
            if self.paras==None:
                if self.embeddings ==None:
                    print ("random embedding")
                    self.Embedding_W = tf.Variable(
                        tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                        name="random_W")
                else:
                    self.Embedding_W = tf.Variable(np.array(self.embeddings),name="embedding_W" ,dtype="float32",trainable=trainable)
            else:
                print ("load embeddings")
                self.Embedding_W=tf.Variable(self.paras[0],trainable=trainable,name="embedding_W")
            self.updated_paras.append(self.Embedding_W)

        self.kernels=[]        
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, embedding_size, 1, self.num_filters]
                if self.paras==None:
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="kernel_W")
                    b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="kernel_b")
                    self.kernels.append((W,b))
                else:
                    _W,_b=self.paras[1][i]
                    W=tf.Variable(_W)                
                    b=tf.Variable(_b)
                    self.kernels.append((W,b))   
                self.updated_paras.append(W)
                self.updated_paras.append(b)

        

        self.l2_loss = tf.constant(0.0)
        for para in self.updated_paras:
            self.l2_loss+= tf.nn.l2_loss(para)
        

        with tf.name_scope("output"):
            q  =self.getRepresentation(self.input_x_1)
            pos=self.getRepresentation(self.input_x_2)
            neg=self.getRepresentation(self.input_x_3)

            self.score12 = self.cosine(q,pos)
            self.score13 = self.cosine(q,neg)

            self.positive= tf.reduce_mean(self.score12)
            self.negative= tf.reduce_mean( self.score13)

    def getRepresentation(self,sentence):
        embedded_chars_1 = tf.nn.embedding_lookup(self.Embedding_W, sentence)
        embedded_chars_expanded_1 = tf.expand_dims(embedded_chars_1, -1)
        output=[]
        for i, filter_size in enumerate(self.filter_sizes): 
            conv = tf.nn.conv2d(
                embedded_chars_expanded_1,
                self.kernels[i][0],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="conv-1"
            )
            h = tf.nn.relu(tf.nn.bias_add(conv, self.kernels[i][1]), name="relu-1")
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="poll-1"
            )
            output.append(pooled)
        pooled_reshape = tf.reshape(tf.concat(3, output), [-1, self.num_filters_total]) 
        pooled_flat = tf.nn.dropout(pooled_reshape, self.dropout_keep_prob)
        return pooled_flat
    def cosine(self,q,a):

        pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.mul(q, q), 1)) 
        pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.mul(a, a), 1))

        pooled_mul_12 = tf.reduce_sum(tf.mul(q, a), 1) 
        score = tf.div(pooled_mul_12, tf.mul(pooled_len_1, pooled_len_2)+1e-8, name="scores") 
        return score 
      
    
    
    def save_model(self, sess,precision_current=0):

        now = int(time.time())             
        timeArray = time.localtime(now)
        timeStamp = time.strftime("%Y%m%d%H%M%S", timeArray)
        filename="model/"+self.model_type+str(precision_current)+"-"+timeStamp+".model"

        param = sess.run([self.Embedding_W,self.kernels])
        pickle.dump(param, open(filename, 'w'))
        return filename


