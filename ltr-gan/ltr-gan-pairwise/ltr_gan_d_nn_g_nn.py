import tensorflow as tf
from dis_model_pairwise_nn import DIS
from gen_model_nn import GEN
import cPickle
import utils as ut
import numpy as np
import random
from eval.precision import precision_at_k
from eval.ndcg import ndcg_at_k


FEATURE_SIZE = 46
HIDDEN_SIZE = 46
BATCH_SIZE = 8
D_WEIGHT_DECAY = 0.01
G_WEIGHT_DECAY = 0.01
D_LEARNING_RATE = 0.0001
G_LEARNING_RATE = 0.0005

workdir = 'MQ2008-semi'
DIS_TRAIN_FILE = workdir + '/run-train-gan.txt'
DIS_MODEL_FILE_NN = workdir + '/dns_nn.model'

GAN_MODEL_BEST_FILE = workdir + '/gan_best_nn.model'


query_url_feature = ut.load_all_query_url_feature(workdir + '/Large_norm.txt', FEATURE_SIZE)
query_pos_train = ut.get_query_pos(workdir + '/train.txt')
query_pos_test = ut.get_query_pos(workdir + '/test.txt')


'''
If there are only two levels of relevance and for each "observed"
relevant-irrelevant document pair (d_i, d_j) we sample an unlabelled
document d_k to form the "generated" document pair (d_k, d_j), then it
can be shown that the objective function of the IRGAN-pairwise minimax
game Eq. (7) in the paper is bounded by the mathematical expectation of 
(f_{\phi} (d_i, q) - f_{\phi} (d_k, q)) / 2 which is independent of the
irrelevant document d_j, via a straightforward application of Jensen's
inequality on the logarithm function.
'''
def generate_for_d(sess, model, filename):
    data = []
    print('negative sampling for d using g...')
    for query in query_pos_train:
        pos_list = query_pos_train[query]
        all_list = list(query_url_feature[query].keys())
        # candidate_list = list(set(all_list) - set(pos_list))
        candidate_list = all_list
        pos_set = set(pos_list)

        if len(candidate_list) <= 0:
            continue

        candidate_list_feature = [query_url_feature[query][url] for url in candidate_list]
        candidate_list_feature = np.asarray(candidate_list_feature)

        candidate_list_score = sess.run(model.pred_score, feed_dict={model.pred_data: candidate_list_feature})
        # softmax for candidate
        exp_rating = np.exp(candidate_list_score)
        prob = exp_rating / np.sum(exp_rating)
        neg_list = []
        for i in range(len(pos_list)):
            while True:
                neg = np.random.choice(candidate_list, p=prob)
                if neg not in pos_set:
                    neg_list.append(neg)
                    break

        # neg_index = np.random.choice(np.arange(len(candidate_list)), size=[len(pos_list)], p=prob)
        # neg_list = np.array(candidate_list)[neg_index]

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
    print("load initial model ...")

    param_nn = cPickle.load(open(DIS_MODEL_FILE_NN))
    assert param_nn is not None

    discriminator = DIS(FEATURE_SIZE, HIDDEN_SIZE, D_WEIGHT_DECAY, D_LEARNING_RATE, loss='log', param=param_nn)
    generator = GEN(FEATURE_SIZE, HIDDEN_SIZE, G_WEIGHT_DECAY, G_LEARNING_RATE, param=param_nn)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.initialize_all_variables())

    print('start adversarial training')

    p_best_val = 0.0
    ndcg_best_val = 0.0

    for epoch in range(30):
        if epoch > 0:
            # G generate negative for D, then train D
            print('Training D ...')
            generate_for_d(sess, generator, DIS_TRAIN_FILE)
            train_size = ut.file_len(DIS_TRAIN_FILE)

            for d_epoch in range(30):
                index = 1
                while True:
                    if index > train_size:
                        break
                    if index + BATCH_SIZE <= train_size + 1:
                        input_pos, input_neg = ut.get_batch_data(DIS_TRAIN_FILE, index, BATCH_SIZE)
                    else:
                        input_pos, input_neg = ut.get_batch_data(DIS_TRAIN_FILE, index, train_size - index + 1)
                    index += BATCH_SIZE

                    _ = sess.run(discriminator.d_updates,
                                 feed_dict={discriminator.pos_data: input_pos, discriminator.neg_data: input_neg})

                p_5 = precision_at_k(sess, discriminator, query_pos_test, query_pos_train, query_url_feature, k=5)
                ndcg_5 = ndcg_at_k(sess, discriminator, query_pos_test, query_pos_train, query_url_feature, k=5)

                if p_5 > p_best_val:
                    p_best_val = p_5
                    ndcg_best_val = ndcg_5
                    discriminator.save_model(sess, GAN_MODEL_BEST_FILE)
                    print("Best: ", "dis p@5 ", p_5, "dis ndcg@5 ", ndcg_5)
                elif p_5 == p_best_val:
                    if ndcg_5 > ndcg_best_val:
                        ndcg_best_val = ndcg_5
                        discriminator.save_model(sess, GAN_MODEL_BEST_FILE)
                        print("Best: ", "dis p@5 ", p_5, "dis ndcg@5 ", ndcg_5)

        # Train G
        print('Training G ...')
        for g_epoch in range(50):  # 50
            for query in query_pos_train.keys():
                pos_list = query_pos_train[query]
                # candidate_list = list(set(query_url_feature[query].keys()) - set(pos_list))
                candidate_list = list(query_url_feature[query].keys())

                if len(candidate_list) <= 0:
                    continue

                candidate_list_feature = [query_url_feature[query][url] for url in candidate_list]
                candidate_list_feature = np.asarray(candidate_list_feature)
                candidate_list_score = sess.run(generator.pred_score, {generator.pred_data: candidate_list_feature})

                # softmax for all
                exp_rating = np.exp(candidate_list_score)
                prob = exp_rating / np.sum(exp_rating)

                neg_index = np.random.choice(np.arange(len(candidate_list)), size=[len(pos_list)], p=prob)
                neg_list = np.array(candidate_list)[neg_index]

                pos_list_feature = [query_url_feature[query][url] for url in pos_list]
                neg_list_feature = [query_url_feature[query][url] for url in neg_list]
                neg_index = np.asarray(neg_index)
                # every negative samples have a reward
                neg_reward = sess.run(discriminator.reward,
                                      feed_dict={discriminator.pos_data: pos_list_feature,
                                                 discriminator.neg_data: neg_list_feature})

                # Method 1: softmax before gather
                _ = sess.run(generator.gan_updates,
                             feed_dict={generator.pred_data: candidate_list_feature,
                                        generator.sample_index: neg_index,
                                        generator.reward: neg_reward})

    print('Best p@5: ', p_best_val, 'Best ndcg@5: ', ndcg_best_val)


if __name__ == '__main__':
    main()