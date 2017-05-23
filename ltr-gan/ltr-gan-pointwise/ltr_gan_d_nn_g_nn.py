import cPickle
import random
import tensorflow as tf
import numpy as np
from eval.precision import precision_at_k
from eval.ndcg import ndcg_at_k
from eval.map import MAP
from eval.mrr import MRR
import utils as ut
from dis_model_pointwise_nn import DIS
from gen_model_nn import GEN


FEATURE_SIZE = 46
HIDDEN_SIZE = 46
BATCH_SIZE = 8
WEIGHT_DECAY = 0.01
D_LEARNING_RATE = 0.001
G_LEARNING_RATE = 0.001
TEMPERATURE = 0.2
LAMBDA = 0.5

workdir = 'MQ2008-semi'
DIS_TRAIN_FILE = workdir + '/run-train-gan.txt'
GAN_MODEL_BEST_FILE = workdir + '/gan_best_nn.model'


query_url_feature, query_url_index, query_index_url =\
    ut.load_all_query_url_feature(workdir + '/Large_norm.txt', FEATURE_SIZE)
query_pos_train = ut.get_query_pos(workdir + '/train.txt')
query_pos_test = ut.get_query_pos(workdir + '/test.txt')


def generate_for_d(sess, model, filename):
    data = []
    print('negative sampling for d using g ...')
    for query in query_pos_train:
        pos_list = query_pos_train[query]
        all_list = query_index_url[query]
        candidate_list = all_list

        candidate_list_feature = [query_url_feature[query][url] for url in candidate_list]
        candidate_list_feature = np.asarray(candidate_list_feature)

        candidate_list_score = sess.run(model.pred_score, feed_dict={model.pred_data: candidate_list_feature})
        # softmax for candidate
        exp_rating = np.exp(candidate_list_score - np.max(candidate_list_score))
        prob = exp_rating / np.sum(exp_rating)

        neg_list = np.random.choice(candidate_list, size=[len(pos_list)], p=prob)

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
    discriminator = DIS(FEATURE_SIZE, HIDDEN_SIZE, WEIGHT_DECAY, D_LEARNING_RATE, param=None)
    generator = GEN(FEATURE_SIZE, HIDDEN_SIZE, WEIGHT_DECAY, G_LEARNING_RATE, temperature=TEMPERATURE, param=None)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.initialize_all_variables())

    print('start adversarial training')

    p_best_val = 0.0
    ndcg_best_val = 0.0

    for epoch in range(30):
        if epoch >= 0:
            # G generate negative for D, then train D
            print('Training D ...')
            for d_epoch in range(100):
                if d_epoch % 30 == 0:
                    generate_for_d(sess, generator, DIS_TRAIN_FILE)
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

                    pred_data = []
                    pred_data.extend(input_pos)
                    pred_data.extend(input_neg)
                    pred_data = np.asarray(pred_data)

                    pred_data_label = [1.0] * len(input_pos)
                    pred_data_label.extend([0.0] * len(input_neg))
                    pred_data_label = np.asarray(pred_data_label)

                    _ = sess.run(discriminator.d_updates,
                                 feed_dict={discriminator.pred_data: pred_data,
                                            discriminator.pred_data_label: pred_data_label})
        # Train G
        print('Training G ...')
        for g_epoch in range(30):
            for query in query_pos_train.keys():
                pos_list = query_pos_train[query]
                pos_set = set(pos_list)
                all_list = query_index_url[query]

                all_list_feature = [query_url_feature[query][url] for url in all_list]
                all_list_feature = np.asarray(all_list_feature)
                all_list_score = sess.run(generator.pred_score, {generator.pred_data: all_list_feature})

                # softmax for all
                exp_rating = np.exp(all_list_score - np.max(all_list_score))
                prob = exp_rating / np.sum(exp_rating)

                prob_IS = prob * (1.0 - LAMBDA)

                for i in range(len(all_list)):
                    if all_list[i] in pos_set:
                        prob_IS[i] += (LAMBDA / (1.0 * len(pos_list)))

                choose_index = np.random.choice(np.arange(len(all_list)), [5 * len(pos_list)], p=prob_IS)
                choose_list = np.array(all_list)[choose_index]
                choose_feature = [query_url_feature[query][url] for url in choose_list]
                choose_IS = np.array(prob)[choose_index] / np.array(prob_IS)[choose_index]

                choose_index = np.asarray(choose_index)
                choose_feature = np.asarray(choose_feature)
                choose_IS = np.asarray(choose_IS)

                choose_reward = sess.run(discriminator.reward, feed_dict={discriminator.pred_data: choose_feature})

                _ = sess.run(generator.g_updates,
                             feed_dict={generator.pred_data: all_list_feature,
                                        generator.sample_index: choose_index,
                                        generator.reward: choose_reward,
                                        generator.important_sampling: choose_IS})


            p_5 = precision_at_k(sess, generator, query_pos_test, query_pos_train, query_url_feature, k=5)
            ndcg_5 = ndcg_at_k(sess, generator, query_pos_test, query_pos_train, query_url_feature, k=5)

            if p_5 > p_best_val:
                p_best_val = p_5
                ndcg_best_val = ndcg_5
                generator.save_model(sess, GAN_MODEL_BEST_FILE)
                print("Best:", "gen p@5 ", p_5, "gen ndcg@5 ", ndcg_5)
            elif p_5 == p_best_val:
                if ndcg_5 > ndcg_best_val:
                    ndcg_best_val = ndcg_5
                    generator.save_model(sess, GAN_MODEL_BEST_FILE)
                    print("Best:", "gen p@5 ", p_5, "gen ndcg@5 ", ndcg_5)


    sess.close()
    param_best = cPickle.load(open(GAN_MODEL_BEST_FILE))
    assert param_best is not None
    generator_best = GEN(FEATURE_SIZE, HIDDEN_SIZE, WEIGHT_DECAY, G_LEARNING_RATE, temperature=TEMPERATURE, param=param_best)
    sess = tf.Session(config=config)
    sess.run(tf.initialize_all_variables())

    p_1_best = precision_at_k(sess, generator_best, query_pos_test, query_pos_train, query_url_feature, k=1)
    p_3_best = precision_at_k(sess, generator_best, query_pos_test, query_pos_train, query_url_feature, k=3)
    p_5_best = precision_at_k(sess, generator_best, query_pos_test, query_pos_train, query_url_feature, k=5)
    p_10_best = precision_at_k(sess, generator_best, query_pos_test, query_pos_train, query_url_feature, k=10)

    ndcg_1_best = ndcg_at_k(sess, generator_best, query_pos_test, query_pos_train, query_url_feature, k=1)
    ndcg_3_best = ndcg_at_k(sess, generator_best, query_pos_test, query_pos_train, query_url_feature, k=3)
    ndcg_5_best = ndcg_at_k(sess, generator_best, query_pos_test, query_pos_train, query_url_feature, k=5)
    ndcg_10_best = ndcg_at_k(sess, generator_best, query_pos_test, query_pos_train, query_url_feature, k=10)

    map_best = MAP(sess, generator_best, query_pos_test, query_pos_train, query_url_feature)
    mrr_best = MRR(sess, generator_best, query_pos_test, query_pos_train, query_url_feature)

    print("Best ", "p@1 ", p_1_best, "p@3 ", p_3_best, "p@5 ", p_5_best, "p@10 ", p_10_best)
    print("Best ", "ndcg@1 ", ndcg_1_best, "ndcg@3 ", ndcg_3_best, "ndcg@5 ", ndcg_5_best, "p@10 ", ndcg_10_best)
    print("Best MAP ", map_best)
    print("Best MRR ", mrr_best)


if __name__ == '__main__':
    main()