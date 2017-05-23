#coding=utf-8
import tensorflow as tf 
import numpy as np 
import time
import pickle
from QACNN import QACNN           
class Discriminator(QACNN):
  

    def __init__(self, sequence_length, batch_size,vocab_size, embedding_size,filter_sizes, num_filters, dropout_keep_prob=1.0,l2_reg_lambda=0.0,learning_rate=1e-2,paras=None,embeddings=None,loss="pair",trainable=True):
        
        QACNN.__init__(self, sequence_length, batch_size,vocab_size, embedding_size,filter_sizes, num_filters, dropout_keep_prob=dropout_keep_prob,l2_reg_lambda=l2_reg_lambda,paras=paras,learning_rate=learning_rate,embeddings=embeddings,loss=loss,trainable=trainable)
        self.model_type="Dis"

    
        with tf.name_scope("output"):

            self.losses = tf.maximum(0.0, tf.sub(0.05, tf.sub(self.score12, self.score13)))
            self.loss = tf.reduce_sum(self.losses) + self.l2_reg_lambda * self.l2_loss
            
            self.reward = 2.0*(tf.sigmoid(tf.sub(0.05, tf.sub(self.score12, self.score13))) -0.5) # no log
            self.positive= tf.reduce_mean(self.score12)
            self.negative= tf.reduce_mean( self.score13)

            self.correct = tf.equal(0.0, self.losses)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, "float"), name="accuracy")


        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads_and_vars]
        self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)

