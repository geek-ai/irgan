import numpy as np

def cal_mrr(r):
    num = 1
    for i in r:
        if i:
            break
        num += 1
    return 1. / num


def MRR(sess, model, query_pos_test, query_pos_train, query_url_feature):
    rs = []
    for query in query_pos_test.keys():
        pos_set = set(query_pos_test[query])
        pred_list = list(set(query_url_feature[query].keys()) - set(query_pos_train.get(query, [])))

        pred_list_feature = [query_url_feature[query][url] for url in pred_list]
        pred_list_feature = np.asarray(pred_list_feature)
        pred_list_score = sess.run(model.pred_score, feed_dict={model.pred_data: pred_list_feature})
        pred_url_score = zip(pred_list, pred_list_score)
        pred_url_score = sorted(pred_url_score, key=lambda x: x[1], reverse=True)

        r = [0.0] * len(pred_list_score)
        for i in range(0, len(pred_list_score)):
            (url, score) = pred_url_score[i]
            if url in pos_set:
                r[i] = 1.0
        rs.append(r)

    return np.mean([cal_mrr(r) for r in rs])


def MRR_user(sess, model, query_pos_test, query_pos_train, query_url_feature):
    rs = []
    query_test_list = sorted(query_pos_test.keys())
    for query in query_test_list:
        pos_set = set(query_pos_test[query])
        pred_list = list(set(query_url_feature[query].keys()) - set(query_pos_train.get(query, [])))

        pred_list_feature = [query_url_feature[query][url] for url in pred_list]
        pred_list_feature = np.asarray(pred_list_feature)
        pred_list_score = sess.run(model.pred_score, feed_dict={model.pred_data: pred_list_feature})
        pred_url_score = zip(pred_list, pred_list_score)
        pred_url_score = sorted(pred_url_score, key=lambda x: x[1], reverse=True)

        r = [0.0] * len(pred_list_score)
        for i in range(0, len(pred_list_score)):
            (url, score) = pred_url_score[i]
            if url in pos_set:
                r[i] = 1.0
        rs.append(r)

    return [cal_mrr(r) for r in rs]