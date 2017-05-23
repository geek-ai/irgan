import tensorflow as tf
from dis_model_pairwise_nn import DIS
import cPickle
import utils as ut
from eval.map import MAP
from eval.mrr import MRR
from eval.precision import precision_at_k
from eval.ndcg import ndcg_at_k


FEATURE_SIZE = 46
HIDDEN_SIZE = 46
BATCH_SIZE = 8
D_WEIGHT_DECAY = 0.001
D_LEARNING_RATE = 0.0001

workdir = 'MQ2008-semi'
GAN_PAIRWISE_MODEL_BEST_FILE = workdir + '/gan/gan_best_nn.model'

query_url_feature =\
    ut.load_all_query_url_feature(workdir + '/Large_norm.txt', FEATURE_SIZE)
query_pos_train = ut.get_query_pos(workdir + '/train.txt')
query_pos_test = ut.get_query_pos(workdir + '/test.txt')


param_best = cPickle.load(open(GAN_PAIRWISE_MODEL_BEST_FILE))
assert param_best is not None
discriminator_best = DIS(FEATURE_SIZE, HIDDEN_SIZE, D_WEIGHT_DECAY, D_LEARNING_RATE, loss='log', param=param_best)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.initialize_all_variables())

p_1_best = precision_at_k(sess, discriminator_best, query_pos_test, query_pos_train, query_url_feature, k=1)
p_3_best = precision_at_k(sess, discriminator_best, query_pos_test, query_pos_train, query_url_feature, k=3)
p_5_best = precision_at_k(sess, discriminator_best, query_pos_test, query_pos_train, query_url_feature, k=5)
p_10_best = precision_at_k(sess, discriminator_best, query_pos_test, query_pos_train, query_url_feature, k=10)

ndcg_1_best = ndcg_at_k(sess, discriminator_best, query_pos_test, query_pos_train, query_url_feature, k=1)
ndcg_3_best = ndcg_at_k(sess, discriminator_best, query_pos_test, query_pos_train, query_url_feature, k=3)
ndcg_5_best = ndcg_at_k(sess, discriminator_best, query_pos_test, query_pos_train, query_url_feature, k=5)
ndcg_10_best = ndcg_at_k(sess, discriminator_best, query_pos_test, query_pos_train, query_url_feature, k=10)

map_best = MAP(sess, discriminator_best, query_pos_test, query_pos_train, query_url_feature)
mrr_best = MRR(sess, discriminator_best, query_pos_test, query_pos_train, query_url_feature)

print("Best ", "p@1 ", p_1_best, "p@3 ", p_3_best, "p@5 ", p_5_best, "p@10 ", p_10_best)
print("Best ", "ndcg@1 ", ndcg_1_best, "ndcg@3 ", ndcg_3_best, "ndcg@5 ", ndcg_5_best, "p@10 ", ndcg_10_best)
print("Best MAP ", map_best)
print("Best MRR ", mrr_best)