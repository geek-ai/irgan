import tensorflow as tf
from dis_model_dns import DIS
import cPickle
import numpy as np
import multiprocessing
import time
import matplotlib.pyplot as plt

cores = multiprocessing.cpu_count()/2

#########################################################################################
# Hyper-parameters
#########################################################################################
EMB_DIM = 5
DNS_K = 5
workdir= 'SEEK_AU_202109_100_5K/'
train_filename='train'
test_filename='test'
DIS_TRAIN_FILE = workdir + 'dis-train.txt'
DIS_MODEL_FILE = workdir + "model_dns.pkl"
#########################################################################################
# Load data
#########################################################################################
user_pos_train = {}
all_items_ids = []
all_user_ids = []
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

user_pos_test = {}
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
    for i, j in item_sort:
        if i in user_pos_test[u]:
            r.append(1)
        else:
            r.append(0)

    p_3 = np.mean(r[:3])
    p_5 = np.mean(r[:5])
    p_100 = np.mean(r[:100])

    ndcg_3 = ndcg_at_k(r, 3)
    ndcg_5 = ndcg_at_k(r, 5)
    ndcg_100 = ndcg_at_k(r, 100)

    return np.array([p_3, p_5, p_100, ndcg_3, ndcg_5, ndcg_100])

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
    for i, j in item_sort:
        if i in user_pos_train[u]:
            r.append(1)
        else:
            r.append(0)

    p_3 = np.mean(r[:3])
    p_5 = np.mean(r[:5])
    p_100 = np.mean(r[:100])

    ndcg_3 = ndcg_at_k(r, 3)
    ndcg_5 = ndcg_at_k(r, 5)
    ndcg_100 = ndcg_at_k(r, 100)

    return np.array([p_3, p_5, p_100, ndcg_3, ndcg_5, ndcg_100])

def evaluate(sess, model, which_set = "test"):
    if which_set == "test":
        which_func = simple_test_one_user_test
    else:
        which_func = simple_test_one_user_train
    result = np.array([0.] * 2)
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
            result += [re[2], re[5]]
    pool.close()
    ret = result / test_user_num
    ret = zip(["p_100", "ndcg_100"], list(ret))
    return ret


def generate_uniform(filename):
    data = []
    print('uniform negative sampling...')
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
    discriminator = DIS(ITEM_NUM, USER_NUM, EMB_DIM, lamda=0.1, param=param, initdelta=0.05, learning_rate=0.05)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    dis_log = open(workdir + 'dis_log_dns.txt', 'w')

    epoch = 0
    result_train = evaluate(sess, discriminator, "train")
    result_test = evaluate(sess, discriminator, "test")
    print("epoch ", epoch, "dis train: ", result_train, "dis test:", result_test)

    #generate_uniform(DIS_TRAIN_FILE) # Uniformly sample negative examples

    # creating initial data values
    # of x and y
    x_values = np.array([0])
    y_values_train =  np.array([result_train[1][1]])
    y_values_test =  np.array([result_test[1][1]])

    for epoch in range(5):
        generate_dns(sess, discriminator, DIS_TRAIN_FILE)  # dynamic negative sample
        with open(DIS_TRAIN_FILE)as fin:
            for line in fin:
                line = line.split()
                u = int(line[0])
                i = int(line[1])
                j = int(line[2])
                _ = sess.run(discriminator.d_updates,
                             feed_dict={discriminator.u: [u], discriminator.pos: [i],
                                        discriminator.neg: [j]})

        result_train = evaluate(sess, discriminator, "train")
        result_test = evaluate(sess, discriminator, "test")
        print("epoch ", epoch+1, "dis train: ", result_train, "dis test:", result_test)
        x_values = np.append(x_values, epoch+1)
        y_values_train = np.append(y_values_train, result_train[1][1])
        y_values_test = np.append(y_values_test, result_test[1][1])


        buf = '\t'.join([str(x) for x in result_train])
        dis_log.write(str(epoch) + '\t' + buf + '\n')
        dis_log.flush()

    dis_log.close()

    # to run GUI event loop
    plt.ion()

    # here we are creating sub plots
    figure, ax = plt.subplots(figsize=(10, 8))
    line1, = ax.plot(x_values, y_values_train)
    line1.set_xdata(x_values)
    line1.set_ydata(y_values_train)
    line2, = ax.plot(x_values, y_values_test)
    line2.set_xdata(x_values)
    line2.set_ydata(y_values_test)

    # setting title
    plt.title("Model convergence", fontsize=20)

    # setting x-axis label and y-axis label
    plt.xlabel("Iteration")
    plt.ylabel("Model Performance")
    plt.show()

    # drawing updated values
    figure.canvas.draw()

    # This will run the GUI event
    # loop until all UI events
    # currently waiting have been processed
    figure.canvas.flush_events()


if __name__ == '__main__':
    main()
