import tensorflow as tf
import cPickle


class DIS():
    def __init__(self, feature_size, hidden_size, weight_decay, learning_rate, loss='log', param=None):
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.d_params = []

        self.pos_data = tf.placeholder(tf.float32, shape=[None, self.feature_size], name="pos_data")
        self.neg_data = tf.placeholder(tf.float32, shape=[None, self.feature_size], name="neg_data")
        self.pred_data = tf.placeholder(tf.float32, shape=[None, self.feature_size], name="pred_data")

        with tf.variable_scope('discriminator'):
            if param == None:
                self.W_1 = tf.get_variable('weight_1', [self.feature_size, self.hidden_size],
                                         initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
                self.W_2 = tf.get_variable('weight_2', [self.hidden_size, 1],
                                         initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
                self.b = tf.get_variable('b', [self.hidden_size], initializer=tf.constant_initializer(0.0))
            else:
                self.W_1 = tf.Variable(param[0])
                self.W_2 = tf.Variable(param[1])
                self.b = tf.Variable(param[2])
            self.d_params.append(self.W_1)
            self.d_params.append(self.W_2)
            self.d_params.append(self.b)

        pos_score = tf.matmul(tf.nn.tanh(tf.nn.xw_plus_b(self.pos_data, self.W_1, self.b)), self.W_2)
        neg_score = tf.matmul(tf.nn.tanh(tf.nn.xw_plus_b(self.neg_data, self.W_1, self.b)), self.W_2)

        if loss == 'svm':
            # ranking svm loss
            with tf.name_scope('svm_loss'):
                self.loss = tf.reduce_mean(tf.maximum(0.0, 1.0 - (pos_score - neg_score)))\
                            + self.weight_decay * (tf.nn.l2_loss(self.W_1) + tf.nn.l2_loss(self.W_2) + tf.nn.l2_loss(self.b))
                # For generator
                self.reward = tf.sigmoid(tf.maximum(0.0, 1.0 - (pos_score - neg_score)))
        elif loss == 'log':
            # ranking log loss
            with tf.name_scope('log_loss'):
                self.loss = -tf.reduce_mean(tf.log(tf.sigmoid(pos_score - neg_score)))\
                            + self.weight_decay * (tf.nn.l2_loss(self.W_1) + tf.nn.l2_loss(self.W_2) + tf.nn.l2_loss(self.b))
                # For generator
                self.reward = tf.reshape(tf.log(tf.sigmoid(neg_score - pos_score)), [-1])
        else:
            assert 'You should use svm and log.'

        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        # optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.d_updates = optimizer.minimize(self.loss, var_list=self.d_params)

        # Given batch query-url pairs, calculate the matching score
        self.pred_score = tf.matmul(tf.nn.tanh(tf.nn.xw_plus_b(self.pred_data, self.W_1, self.b)), self.W_2)


    def save_model(self, sess, filename):
        param = sess.run(self.d_params)
        cPickle.dump(param, open(filename, 'w'))