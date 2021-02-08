import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import math
import time
import utils

flags = tf.flags
FLAGS = flags.FLAGS


class SVD:
    def __init__(self, num_users, num_items, dataset):
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = FLAGS.embed_size
        self.batch_size=FLAGS.batch_size
        self.reg = 0.01
        self.dataset = dataset
        self.coo_mx = self.dataset.trainMatrix.tocoo()
        self.mu_np = np.mean(self.coo_mx.data)

    def create_placeholders(self):
        with tf.variable_scope('placeholder'):
            self.users_holder = tf.placeholder(tf.int32, shape=[None, 1], name='users')
            self.items_holder = tf.placeholder(tf.int32, shape=[None, 1], name='items')
            self.ratings_holder = tf.placeholder(tf.float32, shape=[None, 1], name='ratings')
            self.mask = tf.placeholder(tf.float32, name='mask')

    def create_user_terms(self):
        num_users = self.num_users
        num_factors = self.num_factors
        w_init = slim.xavier_initializer

        with tf.variable_scope('user'):
            self.user_embeddings = tf.get_variable(
                name='embedding',
                shape=[num_users, num_factors],
                initializer=w_init(), regularizer=slim.l2_regularizer(self.reg))
            self.p_u = tf.reduce_sum(tf.nn.embedding_lookup(
                self.user_embeddings,
                self.users_holder,
                name='p_u'), axis=1)

    def create_item_terms(self):
        num_items = self.num_items
        num_factors = self.num_factors

        w_init = slim.xavier_initializer
        with tf.variable_scope('item'):
            self.item_embeddings = tf.get_variable(
                name='embedding',
                shape=[num_items, num_factors],
                initializer=w_init(), regularizer=slim.l2_regularizer(self.reg))
            self.q_i = tf.reduce_sum(tf.nn.embedding_lookup(
                self.item_embeddings,
                self.items_holder,
                name='q_i'), axis=1)

    def create_prediction(self):
        with tf.variable_scope('prediction'):
            pred = tf.reduce_sum(tf.multiply(self.p_u, self.q_i), axis=1)
            self.pred = tf.expand_dims(pred, axis=-1)
            self.rate = tf.matmul(self.user_embeddings, tf.transpose(self.item_embeddings))
            self.rate_partial = tf.matmul(self.user_embeddings[:200], tf.transpose(self.item_embeddings))

    def create_optimizer(self):
        with tf.variable_scope('loss'):
            loss = tf.nn.l2_loss(tf.subtract(self.ratings_holder, self.pred))
            self.MAE = tf.reduce_mean(tf.abs(self.ratings_holder - self.pred))
            self.RMSE = tf.sqrt(tf.reduce_mean((self.ratings_holder - self.pred) * (self.ratings_holder - self.pred)))
            self.loss = tf.add(loss,
                               tf.add_n(tf.get_collection(
                                   tf.GraphKeys.REGULARIZATION_LOSSES)), name='loss')
            if (FLAGS.dataset == 'ml-1m' or FLAGS.dataset=='yelp'):
                self.optimizer = tf.train.AdamOptimizer(0.005)
            else:
                self.optimizer = tf.train.AdamOptimizer(0.001)
            self.train_op = self.optimizer.minimize(self.loss, name='optimizer')

    def build_graph(self):
        self.create_placeholders()
        self.create_user_terms()
        self.create_item_terms()
        self.create_prediction()
        self.create_optimizer()

    def train(self, dataset, is_train, nb_epochs,per_epochs=1):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        samples = utils.sampling(dataset, 0)
        for cur_epochs in range(nb_epochs):
            batchs = utils.get_batchs(samples, FLAGS.batch_size)
            for i in range(len(batchs)):
                users, items, rates = batchs[i]
                feed_dict = {self.users_holder: users,
                             self.items_holder: items,
                             self.ratings_holder: rates}
                self.sess.run([self.train_op], feed_dict)

            if ((cur_epochs % per_epochs == 0 and cur_epochs>0)):
                rate = self.sess.run(self.rate)
                user = dataset.trainMatrix.toarray()
                mask = user != 0
                rate[mask] = -np.inf
                count = 0
                for i in range(len(dataset.testRatings)):
                    idx = np.argsort(rate[i])[::-1][:FLAGS.top_k]
                    for j in FLAGS.target_item:
                        count += (j in idx)
                all_hr = count / len(dataset.testRatings) / len(FLAGS.target_item)
                rmse = 0
                for i in range(len(dataset.testRatings)):
                    uu, ii, rr = self.dataset.testRatings[i]
                    if (rate[uu, ii] == -np.inf):
                        print(uu, ii)
                    rmse += (rr * dataset.max_rate - rate[uu, ii] * dataset.max_rate) * (
                                rr * dataset.max_rate - rate[uu, ii] * dataset.max_rate)
                rmse /= len(dataset.testRatings)
                print("epochs %d: " % cur_epochs, all_hr, rmse)

    def get_embeddings(self):
        results = self.sess.run([self.rate, self.user_embeddings, self.item_embeddings])
        return results
