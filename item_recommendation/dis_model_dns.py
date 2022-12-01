import tensorflow as tf
import cPickle


class DIS():
    def __init__(self, itemNum, userNum, emb_dim, lamda, param=None, initdelta=0.05, learning_rate=0.05):
        self.itemNum = itemNum
        self.userNum = userNum
        self.emb_dim = emb_dim
        self.lamda = lamda  # regularization parameters
        self.param = param
        self.initdelta = initdelta
        self.learning_rate = learning_rate
        self.d_params = []

        with tf.variable_scope('discriminator'):
            if self.param is None:
                self.user_embeddings = tf.Variable(
                    tf.random_uniform([self.userNum, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta,
                                      dtype=tf.float32))
                self.item_embeddings = tf.Variable(
                    tf.random_uniform([self.itemNum, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta,
                                      dtype=tf.float32))
                self.item_bias = tf.Variable(tf.zeros([self.itemNum]))
            else:
                self.user_embeddings = tf.Variable(self.param[0])
                self.item_embeddings = tf.Variable(self.param[1])
                self.item_bias = tf.Variable(self.param[2])

        self.d_params = [self.user_embeddings, self.item_embeddings, self.item_bias]

        # placeholder definition
        self.u = tf.placeholder(tf.int32)
        self.pos = tf.placeholder(tf.int32)
        self.neg = tf.placeholder(tf.int32)

        self.u_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.u)
        self.pos_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.pos)
        self.pos_bias = tf.gather(self.item_bias, self.pos)
        self.neg_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.neg)
        self.neg_bias = tf.gather(self.item_bias, self.neg)

        self.pre_logits = tf.sigmoid(
            tf.reduce_sum(tf.multiply(self.u_embedding, self.pos_embedding - self.neg_embedding),
                          1) + self.pos_bias - self.neg_bias)
        self.pre_loss = -tf.reduce_mean(tf.log(self.pre_logits)) + self.lamda * (
            tf.nn.l2_loss(self.u_embedding) +
            tf.nn.l2_loss(self.pos_embedding) +
            tf.nn.l2_loss(self.pos_bias) +
            tf.nn.l2_loss(self.neg_embedding) +
            tf.nn.l2_loss(self.neg_bias)
        )

        d_opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.d_updates = d_opt.minimize(self.pre_loss, var_list=self.d_params)

        # for test stage, self.u: [batch_size]
        self.all_rating = tf.matmul(self.u_embedding, self.item_embeddings, transpose_a=False,
                                    transpose_b=True) + self.item_bias

        self.all_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1) + self.item_bias
        # for dns sample
        self.dns_rating = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1) + self.item_bias

    def save_model(self, sess, filename):
        param = sess.run(self.d_params)
        cPickle.dump(param, open(filename, 'w'))
