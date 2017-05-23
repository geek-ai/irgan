import cPickle
import random
import tensorflow as tf
import numpy as np
from eval.precision import precision_at_k
from eval.ndcg import ndcg_at_k
from eval.map import MAP
from eval.mrr import MRR
import utils as ut
from dis_model_pairwise_nn import DIS


FEATURE_SIZE = 46
HIDDEN_SIZE = 46
BATCH_SIZE = 8
WEIGHT_DECAY = 0.01
D_LEARNING_RATE = 0.0001

DNS_K = 15

workdir = 'MQ2008-semi'
DIS_TRAIN_FILE = workdir + '/run-train-dns.txt'
DNS_MODEL_BEST_FILE = workdir + '/dns_best_nn.model'


query_url_feature, _, _ = ut.load_all_query_url_feature(workdir + '/Large_norm.txt', FEATURE_SIZE)
query_pos_train = ut.get_query_pos(workdir + '/train.txt')
query_pos_test = ut.get_query_pos(workdir + '/test.txt')


def generate_dns(sess, model, filename):
    data = []
    print('dynamic negative sampling ...')
    for query in query_pos_train:
        pos_list = query_pos_train[query]
        candidate_list = list(set(query_url_feature[query].keys()) - set(pos_list))

        if len(candidate_list) <= 0:
            continue

        candidate_list_feature = [query_url_feature[query][url] for url in candidate_list]
        candidate_list_feature = np.asarray(candidate_list_feature)
        candidate_list_score = sess.run(model.pred_score, feed_dict={model.pred_data: candidate_list_feature})

        neg_list = []
        for i in range(len(pos_list)):
            choice_index = np.random.choice(np.arange(len(candidate_list)), size=DNS_K)  # true or false
            choice = np.array(candidate_list)[choice_index]
            choice_score = np.array(candidate_list_score)[choice_index]
            neg_list.append(choice[np.argmax(choice_score)])

        for i in range(len(pos_list)):
            data.append((query, pos_list[i], neg_list[i]))

    random.shuffle(data)
    with open(filename, 'w') as fout:
        for (q, pos, neg) in data:
            fout.write(','.join([str(f) for f in query_url_feature[q][pos]])
                       + '\t'
                       + ','.join([str(f) for f in query_url_feature[q][neg]]) + '\n')
            fout.flush()


def main():
    discriminator = DIS(FEATURE_SIZE, HIDDEN_SIZE, WEIGHT_DECAY, D_LEARNING_RATE, loss='log', param=None)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.initialize_all_variables())

    print('start dynamic negative sampling with log ranking discriminator')
    p_best_val = 0.0
    ndcg_best_val = 0.0

    for epoch in range(200):
        generate_dns(sess, discriminator, DIS_TRAIN_FILE)
        train_size = ut.file_len(DIS_TRAIN_FILE)

        index = 1
        while True:
            if index > train_size:
                break
            if index + BATCH_SIZE <= train_size + 1:
                input_pos, input_neg = ut.get_batch_data(DIS_TRAIN_FILE, index, BATCH_SIZE)
            else:
                input_pos, input_neg = ut.get_batch_data(DIS_TRAIN_FILE, index, train_size - index + 1)
            index += BATCH_SIZE

            input_pos = np.asarray(input_pos)
            input_neg = np.asarray(input_neg)

            _ = sess.run(discriminator.d_updates,
                         feed_dict={discriminator.pos_data: input_pos, discriminator.neg_data: input_neg})

        p_5 = precision_at_k(sess, discriminator, query_pos_test, query_pos_train, query_url_feature, k=5)
        ndcg_5 = ndcg_at_k(sess, discriminator, query_pos_test, query_pos_train, query_url_feature, k=5)

        if p_5 > p_best_val:
            p_best_val = p_5
            discriminator.save_model(sess, DNS_MODEL_BEST_FILE)
            print("Best: ", " p@5 ", p_5, "ndcg@5 ", ndcg_5)
        elif p_5 == p_best_val:
            if ndcg_5 > ndcg_best_val:
                ndcg_best_val = ndcg_5
                discriminator.save_model(sess, DNS_MODEL_BEST_FILE)
                print("Best: ", " p@5 ", p_5, "ndcg@5 ", ndcg_5)


    sess.close()
    param_best = cPickle.load(open(DNS_MODEL_BEST_FILE))
    assert param_best is not None
    discriminator_best = DIS(FEATURE_SIZE, HIDDEN_SIZE, WEIGHT_DECAY, D_LEARNING_RATE, loss='log', param=param_best)

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


if __name__ == '__main__':
    main()