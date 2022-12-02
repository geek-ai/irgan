import tensorflow as tf
from dis_model_dns import DIS
from mf_model import MF

import cPickle
import numpy as np
import multiprocessing
import time
import matplotlib.pyplot as plt

cores = multiprocessing.cpu_count()/2

RUN_MF = False
RUN_DIS = True

#########################################################################################
# Hyper-parameters
#########################################################################################
EMB_DIM = 5
DNS_K = 5
workdir = 'ml-100k/'
train_filename = 'train'
test_filename = 'test'

DIS_TRAIN_FILE = workdir + 'dis-train.txt'
DIS_MODEL_FILE = workdir + "model_dns.pkl"
#########################################################################################
# Load data
#########################################################################################
user_pos_train = {}
all_items_ids = []
all_user_ids = []
NUM_RATINGS_TRAIN = 0
with open(workdir + train_filename)as fin:
    for line in fin:
        line = line.split()
        uid = int(line[0])
        iid = int(line[1])
        r = float(line[2])
        if r > 0:
            if uid in user_pos_train:
                user_pos_train[uid].append(iid)
            else:
                user_pos_train[uid] = [iid]
        if iid not in all_items_ids:
            all_items_ids.append(iid)
        if uid not in all_user_ids:
            all_user_ids.append(uid)
        NUM_RATINGS_TRAIN += 1

user_pos_test = {}
NUM_RATINGS_TEST = 0
with open(workdir + test_filename)as fin:
    for line in fin:
        line = line.split()
        uid = int(line[0])
        iid = int(line[1])
        r = float(line[2])
        if r > 0:
            if uid in user_pos_test:
                user_pos_test[uid].append(iid)
            else:
                user_pos_test[uid] = [iid]
        if iid not in all_items_ids:
            all_items_ids.append(iid)
        if uid not in all_user_ids:
            all_user_ids.append(uid)
        NUM_RATINGS_TEST += 1

USER_NUM = len(user_pos_train)
ITEM_NUM = len(all_items_ids)
print(USER_NUM, ITEM_NUM)
all_items = set(range(ITEM_NUM))

all_users = user_pos_train.keys()
all_users.sort()

def generate_dns(sess, model, filename):
    data = []
    for u in user_pos_train:
        pos = user_pos_train[u]
        all_rating = sess.run(model.dns_rating, {model.u: u})
        all_rating = np.array(all_rating)
        neg = []
        candidates = list(all_items - set(pos))

        for _ in range(len(pos)):
            choice = np.random.choice(candidates, DNS_K)
            choice_score = all_rating[choice]
            neg.append(choice[np.argmax(choice_score)])

        for i in range(len(pos)):
            data.append(str(u) + '\t' + str(pos[i]) + '\t' + str(neg[i]))

    with open(filename, 'w')as fout:
        fout.write('\n'.join(data))


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))


def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


def simple_test_one_user_test(x):
    rating = x[0]
    u = x[1]

    test_items = list(all_items - set(user_pos_train[u]))
    item_score = []
    for i in test_items:
        item_score.append((i, rating[i]))

    item_score = sorted(item_score, key=lambda x: x[1], reverse=True)
    item_sort = [(x[0], x[1]) for x in item_score]

    r = []
    rmse = 0
    for i, j in item_sort:
        if i in user_pos_test[u]:
            r.append(1)
            rmse += np.square(1-j)
        else:
            r.append(0)

    p_3 = np.mean(r[:3])
    p_5 = np.mean(r[:5])
    p_100 = np.mean(r[:100])

    ndcg_3 = ndcg_at_k(r, 3)
    ndcg_5 = ndcg_at_k(r, 5)
    ndcg_100 = ndcg_at_k(r, 100)

    return np.array([p_3, p_5, p_100, ndcg_3, ndcg_5, ndcg_100, rmse])

def simple_test_one_user_train(x):
    rating = x[0]
    u = x[1]

    test_items = list(all_items)
    item_score = []
    for i in test_items:
        item_score.append((i, rating[i]))

    item_score = sorted(item_score, key=lambda x: x[1], reverse=True)
    item_sort = [(x[0], x[1]) for x in item_score]

    r = []
    rmse = 0
    for i, j in item_sort:
        if i in user_pos_train[u]:
            r.append(1)
            rmse += np.square(1-j)
        else:
            r.append(0)

    p_3 = np.mean(r[:3])
    p_5 = np.mean(r[:5])
    p_100 = np.mean(r[:100])

    ndcg_3 = ndcg_at_k(r, 3)
    ndcg_5 = ndcg_at_k(r, 5)
    ndcg_100 = ndcg_at_k(r, 100)

    return np.array([p_3, p_5, p_100, ndcg_3, ndcg_5, ndcg_100, rmse])

def evaluate(sess, model, which_set = "test"):
    num_ratings = 0
    if which_set == "test":
        which_func = simple_test_one_user_test
        num_ratings = NUM_RATINGS_TEST
    else:
        which_func = simple_test_one_user_train
        num_ratings = NUM_RATINGS_TRAIN
    result = np.array([0.] * 3)
    pool = multiprocessing.Pool(cores)
    batch_size = 128
    test_users = user_pos_test.keys()
    test_user_num = len(test_users)
    index = 0
    while True:
        if index >= test_user_num:
            break
        user_batch = test_users[index:index + batch_size]
        index += batch_size

        user_batch_rating = sess.run(model.all_rating, {model.u: user_batch})
        user_batch_rating_uid = zip(user_batch_rating, user_batch)
        batch_result = pool.map(which_func, user_batch_rating_uid)
        for re in batch_result:
            result += [re[2], re[5], re[6]]
    pool.close()
    ret = (np.array(result.tolist()[:2]) / test_user_num).tolist()
    ret.append((np.array(result.tolist()[2]) / num_ratings).tolist())
    ret = zip(["p_100", "ndcg_100", "rmse"], ret)
    return ret


def generate_uniform(filename):
    data = []
    #print('uniform negative sampling...')
    for u in user_pos_train:
        pos = user_pos_train[u]
        candidates = list(all_items - set(pos))
        neg = np.random.choice(candidates, len(pos))
        pos = np.array(pos)

        for i in range(len(pos)):
            data.append(str(u) + '\t' + str(pos[i]) + '\t' + str(neg[i]))

    with open(filename, 'w')as fout:
        fout.write('\n'.join(data))


def main():
    np.random.seed(70)
    param = None
    mf = MF(ITEM_NUM, USER_NUM, EMB_DIM, lamda=0.1, param=param, initdelta=0.05, learning_rate=0.05)
    discriminator = DIS(ITEM_NUM, USER_NUM, EMB_DIM, lamda=0.1, param=param, initdelta=0.05, learning_rate=0.05)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    epoch = 0
    if RUN_MF:
        result_train_mf = evaluate(sess, mf, "train")
        result_test_mf = evaluate(sess, mf, "test")
    if RUN_DIS:
        result_train_dis = evaluate(sess, discriminator, "train")
        result_test_dis = evaluate(sess, discriminator, "test")

    if RUN_MF: print("epoch MF", epoch, "dis train: ", result_train_mf, "dis test:", result_test_mf)
    if RUN_DIS: print("epoch DIS", epoch, "dis train: ", result_train_dis, "dis test:", result_test_dis)

    generate_uniform(DIS_TRAIN_FILE) # Uniformly sample negative examples

    # creating initial data values
    # of x and y
    x_values = np.array([0])
    if RUN_MF:
        y_values_train_mf =  np.array([result_train_mf[0][1]])
        y_values_test_mf =  np.array([result_test_mf[0][1]])
    if RUN_DIS:
        y_values_train_dis =  np.array([result_train_dis[0][1]])
        y_values_test_dis =  np.array([result_test_dis[0][1]])

    for epoch in range(30):
        with open(DIS_TRAIN_FILE)as fin:
            for line in fin:
                line = line.split()
                u = int(line[0])
                i = int(line[1])
                j = int(line[2])
                #positive:
                if RUN_MF:
                    _ = sess.run(mf.d_updates,
                                 feed_dict={mf.u: [u], mf.pos: [i], mf.real: [1.0]})
                    _ = sess.run(mf.d_updates,
                                 feed_dict={mf.u: [u], mf.pos: [j], mf.real: [0.0]})
                if RUN_DIS:
                    _ = sess.run(discriminator.d_updates,
                                 feed_dict={discriminator.u: [u], discriminator.pos: [i], discriminator.neg: [j]})

        if RUN_MF:
            result_train_mf = evaluate(sess, mf, "train")
            result_test_mf = evaluate(sess, mf, "test")
        if RUN_DIS:
            result_train_dis = evaluate(sess, discriminator, "train")
            result_test_dis = evaluate(sess, discriminator, "test")

        if RUN_MF: print("epoch MF", epoch+1, "dis train: ", result_train_mf, "dis test:", result_test_mf)
        if RUN_DIS: print("epoch DIS", epoch+1, "dis train: ", result_train_dis, "dis test:", result_test_dis)
        x_values = np.append(x_values, epoch+1)
        if RUN_MF:
            y_values_train_mf = np.append(y_values_train_mf, result_train_mf[0][1])
            y_values_test_mf = np.append(y_values_test_mf, result_test_mf[0][1])
        if RUN_DIS:
            y_values_train_dis = np.append(y_values_train_dis, result_train_dis[0][1])
            y_values_test_dis = np.append(y_values_test_dis, result_test_dis[0][1])


    if RUN_MF:
        line1, = plt.plot(x_values, y_values_train_mf, label = "P@100 Train MF")
        line1.set_xdata(x_values)
        line1.set_ydata(y_values_train_mf)
    if RUN_DIS:
        line2, = plt.plot(x_values, y_values_train_dis, label = "P@100 Train DIS")
        line2.set_xdata(x_values)
        line2.set_ydata(y_values_train_dis)

    if RUN_MF:
        line3, = plt.plot(x_values, y_values_test_mf, label = "P@100 Test MF")
        line3.set_xdata(x_values)
        line3.set_ydata(y_values_test_mf)
    if RUN_DIS:
        line4, = plt.plot(x_values, y_values_test_dis, label = "P@100 Test DIS")
        line4.set_xdata(x_values)
        line4.set_ydata(y_values_test_dis)

    plt.title("Model convergence", fontsize=20)

    plt.xlabel("Iteration")
    plt.ylabel("Model Performance")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
