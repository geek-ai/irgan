import tensorflow as tf
import cPickle


class DIS():
    def __init__(self, feature_size, hidden_size, weight_decay, learning_rate, param=None):
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.d_params = []

        self.pred_data = tf.placeholder(tf.float32, shape=[None, self.feature_size], name="pred_data")
        self.pred_data_label = tf.placeholder(tf.float32, shape=[None], name="pred_data_label")

        with tf.variable_scope('discriminator'):
            if param == None:
                self.W_1 = tf.get_variable('weight_1', [self.feature_size, self.hidden_size],
                                         initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
                self.W_2 = tf.get_variable('weight_2', [self.hidden_size, 1],
                                         initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
                self.b_1 = tf.get_variable('b_1', [self.hidden_size], initializer=tf.constant_initializer(0.0))
                self.b_2 = tf.get_variable('b_2', [1], initializer=tf.constant_initializer(0.0))
            else:
                self.W_1 = tf.Variable(param[0])
                self.W_2 = tf.Variable(param[1])
                self.b_1 = tf.Variable(param[2])
                self.b_2 = tf.Variable(param[3])
            self.d_params.append(self.W_1)
            self.d_params.append(self.W_2)
            self.d_params.append(self.b_1)
            self.d_params.append(self.b_2)

        # Given batch query-url pairs, calculate the matching score
        self.pred_score = tf.reshape(tf.nn.xw_plus_b(
            tf.nn.tanh(tf.nn.xw_plus_b(self.pred_data, self.W_1, self.b_1)), self.W_2, self.b_2), [-1])

        # ranking log loss
        with tf.name_scope('log_loss'):
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.pred_score, self.pred_data_label))\
                        + self.weight_decay * (tf.nn.l2_loss(self.W_1) + tf.nn.l2_loss(self.W_2)
                                               + tf.nn.l2_loss(self.b_1) + tf.nn.l2_loss(self.b_2))

        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        # self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.d_updates = self.optimizer.minimize(self.loss, var_list=self.d_params)

        self.reward = (tf.sigmoid(self.pred_score) - 0.5) * 2


    def save_model(self, sess, filename):
        param = sess.run(self.d_params)
        cPickle.dump(param, open(filename, 'w'))
