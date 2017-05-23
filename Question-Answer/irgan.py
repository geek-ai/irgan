#coding=utf-8
#! /usr/bin/env python3.4
import numpy as np
import os
import time
import datetime
import operator
import random
import tensorflow as tf
import pickle
import copy

import Discriminator
import Generator
from insurance_qa_data_helpers import encode_sent
import insurance_qa_data_helpers
# import dataHelper
# Model Hyperparameters
tf.flags.DEFINE_integer("max_sequence_length", 200, "Max sequence length fo sentence (default: 200)")
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 500, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 0.05, "learning_rate (default: 0.1)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 500000, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 10, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 10, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("pools_size", 100, "The sampled set of a positive ample, which is bigger than 500")
tf.flags.DEFINE_integer("gen_pools_size", 20, "The sampled set of a positive ample, which is bigger than 500")
tf.flags.DEFINE_integer("g_epochs_num", 1, " the num_epochs of generator per epoch")
tf.flags.DEFINE_integer("d_epochs_num", 1, " the num_epochs of discriminator per epoch")
tf.flags.DEFINE_integer("sampled_size", 100, " the real selectd set from the The sampled pools")
tf.flags.DEFINE_integer("sampled_temperature", 20, " the temperature of sampling")
tf.flags.DEFINE_integer("gan_k", 5, "he number of samples of gan")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
# print(("\nParameters:"))
# for attr, value in sorted(FLAGS.__flags.items()):
#		 print(("{}={}".format(attr.upper(), value)))
# print((""))

timeStamp = time.strftime("%Y%m%d%H%M%S", time.localtime(int(time.time())))


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
log_precision = 'log/test1.gan_precision'+timeStamp
loss_precision = 'log/test1.gan_loss'+timeStamp

from functools import wraps
#print( tf.__version__)
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



def generate_gan(sess, model,loss_type="pair",negative_size=3):
	samples=[]
	for _index ,pair in enumerate (raw):
		if _index %100==0:
			print( "have sampled %d pairs" % _index)
		q=pair[2]
		a=pair[3]


		neg_alist_index=[i for i in range(len(alist))] 
		neg_alist_index.remove(_index)                 #remove the positive index
		sampled_index=np.random.choice(neg_alist_index,size=[FLAGS.pools_size],replace= False)
		pools=np.array(alist)[sampled_index]

		canditates=insurance_qa_data_helpers.loadCandidateSamples(q,a,pools,vocab)	
		predicteds=[]
		for batch in insurance_qa_data_helpers.batch_iter(canditates,batch_size=FLAGS.batch_size):							
			feed_dict = {model.input_x_1: batch[:,0],model.input_x_2: batch[:,1],model.input_x_3: batch[:,2]}			
			predicted=sess.run(model.gan_score,feed_dict)
			predicteds.extend(predicted)

		# index=np.argmax(predicteds)
		# samples.append([encode_sent(vocab,item, FLAGS.max_sequence_length) for item in [q,a,pools[index]]])
		exp_rating = np.exp(np.array(predicteds)*FLAGS.sampled_temperature*1.5)
		prob = exp_rating / np.sum(exp_rating)
		neg_samples = np.random.choice(pools, size= negative_size,p=prob,replace=False) 
		for neg in neg_samples:
			samples.append([encode_sent(vocab,item, FLAGS.max_sequence_length) for item in [q,a,neg]])
	return samples


@log_time_delta	 
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
				cnn.input_x_2: x_test_2,	#x_test_2 equals x_test_3 for the test case
				cnn.input_x_3: x_test_3
			}
			predicted =sess.run(cnn.score12, feed_dict)	 
			batch_scores.extend(predicted)

		max_indexs=[]	

		max_score=max(batch_scores)
		for index,s in enumerate(batch_scores):
			if s==max_score:
				max_indexs.append(index)
		if len(max_indexs)==0:
			scoreList.append(0)
			continue
		index=	random.choice(max_indexs)
		if int(testList[i*500+index].split()[0])==1:
			scoreList.append(1)
		else:
			scoreList.append(0)
	return sum(scoreList) *1.0 /len(scoreList)



@log_time_delta	 
def evaluation(sess,model,log,num_epochs=0):
	current_step = tf.train.global_step(sess, model.global_step)
	if isinstance(model,  Discriminator.Discriminator):
		model_type="Dis"
	else:
		model_type="Gen"

	precision_current=dev_step(sess,model,test1List,1800)
	line="test1: %d epoch: precision %f"%(current_step,precision_current)
	print (line)
	print( model.save_model(sess,precision_current))
	log.write(line+"\n")
	log.flush()


	
def main():
	with tf.Graph().as_default():
		with tf.device("/gpu:1"):
			session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,log_device_placement=FLAGS.log_device_placement)
			sess = tf.Session(config=session_conf)

			with sess.as_default() ,open(log_precision,"w") as log,open(loss_precision,"w") as loss_log :

				DIS_MODEL_FILE="model/pre-trained.model"   # overfitted DNS
				param = pickle.load(open(DIS_MODEL_FILE,"rb"))
		
				# param= None
				loss_type="pair"
				discriminator = Discriminator.Discriminator(
						sequence_length=FLAGS.max_sequence_length,
						batch_size=FLAGS.batch_size,
						vocab_size=len(vocab),
						embedding_size=FLAGS.embedding_dim,
						filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
						num_filters=FLAGS.num_filters,
						learning_rate=FLAGS.learning_rate,
						l2_reg_lambda=FLAGS.l2_reg_lambda,
						# embeddings=embeddings,
						embeddings=None,
						paras=param,
						loss=loss_type)

				generator = Generator.Generator(
						sequence_length=FLAGS.max_sequence_length,
						batch_size=FLAGS.batch_size,
						vocab_size=len(vocab),
						embedding_size=FLAGS.embedding_dim,
						filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
						num_filters=FLAGS.num_filters,
						learning_rate=FLAGS.learning_rate*0.1,
						l2_reg_lambda=FLAGS.l2_reg_lambda,
						# embeddings=embeddings,
						embeddings=None,
						paras=param,
						loss=loss_type)

					
				sess.run(tf.global_variables_initializer())
				# evaluation(sess,discriminator,log,0)
				for i in range(FLAGS.num_epochs):
					if i>0:
						samples=generate_gan(sess,generator) 
						# for j in range(FLAGS.d_epochs_num):							
						for _index,batch in enumerate(insurance_qa_data_helpers.batch_iter(samples,num_epochs=FLAGS.d_epochs_num,batch_size=FLAGS.batch_size,shuffle=True)):	# try:						
						
							feed_dict = {discriminator.input_x_1: batch[:,0],discriminator.input_x_2: batch[:,1],discriminator.input_x_3: batch[:,2]}							
							_, step,	current_loss,accuracy = sess.run(
									[discriminator.train_op, discriminator.global_step, discriminator.loss,discriminator.accuracy],
									feed_dict)

							line=("%s: DIS step %d, loss %f with acc %f "%(datetime.datetime.now().isoformat(), step, current_loss,accuracy))
							if _index%10==0:
								print(line)
							loss_log.write(line+"\n")
							loss_log.flush()
						
						evaluation(sess,discriminator,log,i)

					for g_epoch in range(FLAGS.g_epochs_num):	
						for _index,pair in enumerate(raw):

							q=pair[2]
							a=pair[3]	

							neg_alist_index=[item for item in range(len(alist))] 
							sampled_index=np.random.choice(neg_alist_index,size=[FLAGS.pools_size-1],replace= False)
							sampled_index=list(sampled_index)
							sampled_index.append(_index)
							pools=np.array(alist)[sampled_index]

							samples=insurance_qa_data_helpers.loadCandidateSamples(q,a,pools,vocab)
							predicteds=[]
							for batch in insurance_qa_data_helpers.batch_iter(samples,batch_size=FLAGS.batch_size):							
								feed_dict = {generator.input_x_1: batch[:,0],generator.input_x_2: batch[:,1],generator.input_x_3: batch[:,2]}
								
								predicted=sess.run(generator.gan_score,feed_dict)
								predicteds.extend(predicted)														 
							
							exp_rating = np.exp(np.array(predicteds)*FLAGS.sampled_temperature)
							prob = exp_rating / np.sum(exp_rating)

							neg_index = np.random.choice(np.arange(len(pools)) , size=FLAGS.gan_k, p=prob ,replace=False)	# 生成 FLAGS.gan_k个负例

							subsamples=np.array(insurance_qa_data_helpers.loadCandidateSamples(q,a,pools[neg_index],vocab))	
							feed_dict = {discriminator.input_x_1: subsamples[:,0],discriminator.input_x_2: subsamples[:,1],discriminator.input_x_3: subsamples[:,2]}
							reward = sess.run(discriminator.reward,feed_dict)				 # reward= 2 * (tf.sigmoid( score_13 ) - 0.5)

							samples=np.array(samples)
							feed_dict = {
											generator.input_x_1: samples[:,0],
											generator.input_x_2: samples[:,1],
											generator.neg_index: neg_index,
											generator.input_x_3: samples[:,2],
											generator.reward: reward
										}
							_, step,	current_loss,positive,negative = sess.run(																					#应该是全集上的softmax	但是此处做全集的softmax开销太大了
									[generator.gan_updates, generator.global_step, generator.gan_loss, generator.positive,generator.negative],		 #	 self.prob= tf.nn.softmax( self.cos_13)
									feed_dict)																													#self.gan_loss = -tf.reduce_mean(tf.log(self.prob) * self.reward) 

							line=("%s: GEN step %d, loss %f  positive %f negative %f"%(datetime.datetime.now().isoformat(), step, current_loss,positive,negative))
							if _index %100==0:
								print(line)
							loss_log.write(line+"\n")
							loss_log.flush()
							
						
						evaluation(sess,generator,log,i*FLAGS.g_epochs_num + g_epoch)
						log.flush()



										
if __name__ == '__main__':

	main()

