import tensorflow as tf
import cPickle


class GEN:
    def __init__(self, feature_size, hidden_size, weight_decay, learning_rate, temperature=1.0, param=None):
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.g_params = []

        self.reward = tf.placeholder(tf.float32, shape=[None], name='reward')
        self.pred_data = tf.placeholder(tf.float32, shape=[None, self.feature_size], name="pred_data")
        self.sample_index = tf.placeholder(tf.int32, shape=[None], name='sample_index')
        self.important_sampling = tf.placeholder(tf.float32, shape=[None], name='important_sampling')

        with tf.variable_scope('generator'):
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
            self.g_params.append(self.W_1)
            self.g_params.append(self.W_2)
            self.g_params.append(self.b_1)
            self.g_params.append(self.b_2)

        # Given batch query-url pairs, calculate the matching score
        # For all urls of one query
        self.pred_score = tf.reshape(tf.nn.xw_plus_b(
            tf.nn.tanh(tf.nn.xw_plus_b(self.pred_data, self.W_1, self.b_1)), self.W_2, self.b_2), [-1]) / self.temperature

        self.gan_prob = tf.gather(
            tf.reshape(tf.nn.softmax(tf.reshape(self.pred_score, [1, -1])), [-1]), self.sample_index)

        self.gan_loss = -tf.reduce_mean(tf.log(self.gan_prob) * self.reward * self.important_sampling) \
                        + self.weight_decay * (tf.nn.l2_loss(self.W_1) + tf.nn.l2_loss(self.W_2)
                                               + tf.nn.l2_loss(self.b_1) + tf.nn.l2_loss(self.b_2))

        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.g_updates = self.optimizer.minimize(self.gan_loss, var_list=self.g_params)


    def save_model(self, sess, filename):
        param = sess.run(self.g_params)
        cPickle.dump(param, open(filename, 'w'))