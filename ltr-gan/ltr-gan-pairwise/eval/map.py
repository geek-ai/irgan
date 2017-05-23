import numpy as np

def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)


def average_precision(r):
    r = np.asarray(r)
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def MAP(sess, model, query_pos_test, query_pos_train, query_url_feature):
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

    return np.mean([average_precision(r) for r in rs])


def MAP_user(sess, model, query_pos_test, query_pos_train, query_url_feature):
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

    return [average_precision(r) for r in rs]