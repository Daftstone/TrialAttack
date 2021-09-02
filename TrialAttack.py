import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import sys
import os
import utils

flags = tf.flags
FLAGS = flags.FLAGS

sys.path.append("../")
from k_means import k_means, assign_points


class GAN():
    def __init__(self, dataset, c_dims, z_dims):
        self.dataset = dataset
        self.c_dims = c_dims
        self.z_dims = z_dims
        self.output_dims = dataset.num_items
        self.num_cluster = c_dims

        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.num_factors = 64
        self.reg = 0.01
        self.count = 1
        self.count1 = 1
        self.i_num = 1
        self.build()

    def placeholder_build(self):
        self.z = tf.placeholder(tf.float32)
        self.d = tf.placeholder(tf.float32)
        self.inf_user = tf.placeholder(tf.float32)
        self.inf_user_train = tf.placeholder(tf.float32)
        self.inf_label = tf.placeholder(tf.float32)
        self.c = tf.placeholder(tf.float32)
        self.mask = tf.placeholder(tf.float32)
        self.center = tf.placeholder(tf.float32)
        self.truelabel = tf.placeholder(tf.float32)

        self.value1 = [tf.placeholder(tf.float32) for i in range(self.count1 * self.i_num)]
        self.value2 = [tf.placeholder(tf.float32) for i in range(self.count1 * self.i_num)]
        self.ref_user = [tf.placeholder(tf.float32) for i in range(self.count1 * self.i_num)]

        self.max_if_ph = tf.placeholder(tf.float32)
        self.min_if_ph = tf.placeholder(tf.float32)

        with tf.variable_scope('placeholder'):
            self.users_holder = tf.placeholder(tf.int32, shape=[None, 1], name='users')
            self.items_holder = tf.placeholder(tf.int32, shape=[None, 1], name='items')
            self.ratings_holder = tf.placeholder(tf.float32, shape=[None, 1], name='ratings')
            self.rate_mask = tf.placeholder(tf.float32, name='mask')

    def create_user_terms(self):
        num_users = self.num_users
        num_factors = self.num_factors

        w_init = slim.xavier_initializer
        with tf.variable_scope('user'):
            self.user_embeddings_origin = tf.get_variable(
                name='embedding_origin',
                shape=[num_users * num_factors],
                initializer=w_init(), regularizer=slim.l2_regularizer(self.reg))
            self.user_embeddings = tf.reshape(self.user_embeddings_origin, [num_users, num_factors])
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
            self.item_embeddings_origin = tf.get_variable(
                name='embedding_origin',
                shape=[num_items * num_factors],
                initializer=w_init(), regularizer=slim.l2_regularizer(self.reg))
            self.item_embeddings = tf.reshape(self.item_embeddings_origin, [num_items, num_factors])
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

    def create_optimizer(self):
        with tf.variable_scope('loss'):
            loss = tf.nn.l2_loss(tf.subtract(self.ratings_holder, self.pred))
            self.MAE = tf.reduce_mean(tf.abs(self.ratings_holder - self.pred))
            self.RMSE = tf.sqrt(tf.reduce_mean((self.ratings_holder - self.pred) * (self.ratings_holder - self.pred)))
            self.loss = tf.add(loss,
                               tf.add_n(tf.get_collection(
                                   tf.GraphKeys.REGULARIZATION_LOSSES)), name='loss')
            if (FLAGS.dataset == 'ml-1m'):
                self.optimizer = tf.train.AdamOptimizer(0.002)
            else:
                self.optimizer = tf.train.AdamOptimizer(0.001)
            self.train_op = self.optimizer.minimize(self.loss, name='optimizer')

    def generator_variable(self, hidden_sizes=[512, 64, 512]):
        self.g_weights = []
        self.g_biases = []
        previous_size = self.z_dims
        for ix, layer_size in enumerate(hidden_sizes):
            weight = tf.Variable(
                tf.truncated_normal(shape=[previous_size, layer_size], mean=0.0, stddev=0.01),
                name='g_w_%d' % (ix + 1), dtype=tf.float32)
            bias = tf.Variable(
                tf.zeros(shape=(layer_size)),
                name='g_b_%d' % (ix + 1), dtype=tf.float32)
            self.g_weights.append(weight)
            self.g_biases.append(bias)
            previous_size = layer_size

        weight = tf.Variable(
            tf.truncated_normal(shape=[previous_size, self.output_dims], mean=0.0, stddev=0.01),
            name='g_w_%d' % (len(hidden_sizes) + 1), dtype=tf.float32)
        bias = tf.Variable(
            tf.zeros(shape=[self.output_dims]),
            name='g_b_%d' % (len(hidden_sizes) + 1), dtype=tf.float32)
        self.g_weights.append(weight)
        self.g_biases.append(bias)

    def discriminator_variable(self, hidden_sizes=[512, 256, 64]):
        self.d_weights = []
        self.d_biases = []
        previous_size = self.output_dims + 1
        flag = 0
        for ix, layer_size in enumerate(hidden_sizes):
            # if (ix == len(hidden_sizes) - 1):
            #     flag = 1
            weight = tf.Variable(
                tf.truncated_normal(shape=[previous_size + flag, layer_size], mean=0.0, stddev=0.01),
                name='d_w_%d' % (ix + 1), dtype=tf.float32)
            bias = tf.Variable(
                tf.zeros(shape=(layer_size)),
                name='d_b_%d' % (ix + 1), dtype=tf.float32)
            self.d_weights.append(weight)
            self.d_biases.append(bias)
            previous_size = layer_size

        weight = tf.Variable(
            tf.truncated_normal(shape=[previous_size, 1], mean=0.0, stddev=0.01),
            name='d_w_%d' % (len(hidden_sizes) + 1), dtype=tf.float32)
        bias = tf.Variable(
            tf.zeros(shape=[1]),
            name='d_b_%d' % (len(hidden_sizes) + 1), dtype=tf.float32)
        self.d_weights.append(weight)
        self.d_biases.append(bias)

    def discriminator_variable_v1(self, hidden_sizes=[512, 512, 64]):
        self.d_weights1 = []
        self.d_biases1 = []
        previous_size = self.output_dims
        for ix, layer_size in enumerate(hidden_sizes):
            weight = tf.Variable(
                tf.truncated_normal(shape=[previous_size, layer_size], mean=0.0, stddev=0.01),
                name='d_w1_%d' % (ix + 1), dtype=tf.float32)
            bias = tf.Variable(
                tf.zeros(shape=(layer_size)),
                name='d_b1_%d' % (ix + 1), dtype=tf.float32)
            self.d_weights1.append(weight)
            self.d_biases1.append(bias)
            previous_size = layer_size

        weight = tf.Variable(
            tf.truncated_normal(shape=[previous_size, 1], mean=0.0, stddev=0.01),
            name='d_w1_%d' % (len(hidden_sizes) + 1), dtype=tf.float32)
        bias = tf.Variable(
            tf.zeros(shape=[1]),
            name='d_b1_%d' % (len(hidden_sizes) + 1), dtype=tf.float32)
        self.d_weights1.append(weight)
        self.d_biases1.append(bias)

    def generator_build(self):
        hidden = self.z
        self.slist = []
        for i in range(len(self.g_weights)):
            w = self.g_weights[i]
            b = self.g_biases[i]
            hidden = hidden @ w + b
            if (i != len(self.g_weights) - 1):
                hidden = tf.nn.leaky_relu(hidden)
            self.slist.append(hidden)
        self.generator = (tf.tanh(hidden) / 2. + 0.5) * self.mask

    def discriminator_build(self):
        influence = tf.add_n(
            [- tf.reduce_sum(v2 * (tf.concat([self.generator, self.generator], axis=1) - v3), axis=1, keep_dims=True)
             for v1, v2, v3
             in zip(self.value1, self.value2, self.ref_user)]) / (self.count1 * self.i_num)
        self.influence = (influence - self.min_if_ph) / (self.max_if_ph - self.min_if_ph) * 2. - 1.
        hidden_false = tf.concat([self.generator, self.influence], axis=1)
        hidden_true = tf.concat([self.d, self.truelabel], axis=1)
        hidden_false1 = tf.concat([self.inf_user, self.influence_true], axis=1)
        for i in range(len(self.d_weights)):
            w = self.d_weights[i]
            b = self.d_biases[i]
            hidden_false = hidden_false @ w + b
            hidden_false1 = hidden_false1 @ w + b
            hidden_true = hidden_true @ w + b
            if (i != len(self.d_weights) - 1):
                mean, std = tf.nn.moments(hidden_false, axes=[0, 1])
                hidden_false = (hidden_false - mean) / tf.sqrt(std + 1e-8)
                mean, std = tf.nn.moments(hidden_false1, axes=[0, 1])
                hidden_false1 = (hidden_false1 - mean) / tf.sqrt(std + 1e-8)
                mean, std = tf.nn.moments(hidden_true, axes=[0, 1])
                hidden_true = (hidden_true - mean) / tf.sqrt(std + 1e-8)
                hidden_false = tf.nn.leaky_relu(hidden_false)
                hidden_false1 = tf.nn.leaky_relu(hidden_false1)
                hidden_true = tf.nn.leaky_relu(hidden_true)
        self.discriminator_false = tf.nn.sigmoid(hidden_false)
        self.discriminator_false1 = tf.nn.sigmoid(hidden_false1)
        self.discriminator_true = tf.nn.sigmoid(hidden_true)

    def influence_build(self):
        hidden_false = self.generator
        hidden_true = self.inf_user_train
        hidden_true1 = self.inf_user
        for i in range(len(self.d_weights1)):
            w = self.d_weights1[i]
            b = self.d_biases1[i]
            hidden_false = hidden_false @ w + b
            hidden_true = hidden_true @ w + b
            hidden_true1 = hidden_true1 @ w + b
            if (i != len(self.d_weights1) - 1):
                hidden_false = tf.nn.leaky_relu(hidden_false)
                hidden_true = tf.nn.leaky_relu(hidden_true)
                hidden_true1 = tf.nn.leaky_relu(hidden_true1)
        self.influence_false = hidden_false
        self.influence_true_train = hidden_true
        self.influence_true = hidden_true1

    def TrialAttack_loss_build(self):
        if (FLAGS.dataset == 'ml-1m'):
            atk = 4000.
        elif (FLAGS.dataset == 'filmtrust'):
            atk = 400.
        else:
            atk = 2000.
        alpha1 = 0.01
        alpha2 = 0.
        self.g_loss1 = -tf.reduce_mean(tf.log(self.discriminator_false + 1e-8))
        self.g_loss2 = alpha1 * tf.add_n(
            [tf.nn.l2_loss(v) for v in self.g_weights + self.g_biases])
        self.g_loss3 = tf.reduce_mean(
            tf.reduce_sum((self.generator - self.center) * (self.generator - self.center),
                          axis=1)) * 100.
        self.g_loss4 = tf.reduce_mean(self.influence_false) * -atk
        self.g_loss = self.g_loss1 + self.g_loss2 + self.g_loss3 + self.g_loss4
        self.g_Loss_pre = self.g_loss1 + self.g_loss2 + self.g_loss3
        self.g_optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001)
        self.g_train_op = self.g_optimizer.minimize(self.g_loss,
                                                    var_list=[*self.g_weights, *self.g_biases])
        self.g_optimizer_pre = tf.train.RMSPropOptimizer(learning_rate=0.0001)
        self.g_train_op_pre = self.g_optimizer_pre.minimize(self.g_Loss_pre,
                                                            var_list=[*self.g_weights, *self.g_biases])

        self.d_loss1 = -tf.reduce_mean(tf.log(self.discriminator_true + 1e-8)) + tf.reduce_mean(tf.log(
            self.discriminator_false + 1e-8)) * 0.5 + tf.reduce_mean(tf.log(self.discriminator_false1 + 1e-8)) * 0.5
        self.d_loss2 = alpha2 * tf.add_n(
            [tf.nn.l2_loss(v) for v in self.d_weights + self.d_biases])
        self.d_loss = self.d_loss1 + self.d_loss2
        self.d_optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001)
        self.d_train_op = self.d_optimizer.minimize(self.d_loss,
                                                    var_list=[*self.d_weights, *self.d_biases])

        self.i_loss2 = tf.nn.l2_loss(self.influence_true_train - self.inf_label)
        self.i_loss3 = 0.01 * tf.add_n(
            [tf.nn.l2_loss(v) for v in self.d_weights1 + self.d_biases1])
        self.i_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.i_train_op_pre = self.i_optimizer.minimize(self.i_loss2 + self.i_loss3,
                                                        var_list=[*self.d_weights1, *self.d_biases1])
        self.i_loss = -tf.reduce_mean(tf.log(self.discriminator_false1 + 1e-8))
        self.i_optimizer1 = tf.train.AdamOptimizer(learning_rate=0.001)
        self.i_train_op = self.i_optimizer1.minimize(self.i_loss2 + self.i_loss * 0.1 + self.i_loss3,
                                                     var_list=[*self.d_weights1, *self.d_biases1])

    def build(self):
        self.placeholder_build()
        self.generator_variable()
        self.discriminator_variable()
        self.discriminator_variable_v1()
        self.generator_build()
        self.influence_build()
        self.discriminator_build()
        self.TrialAttack_loss_build()

        # build svd
        self.create_user_terms()
        self.create_item_terms()
        self.create_prediction()
        self.create_optimizer()

        # create session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def svd_train(self, dataset, is_train, nb_epochs):
        ckpt_save_path = "pretrain/%s/svd/embed_64/model_GAN_%s.ckpt" % (FLAGS.dataset, FLAGS.gpu)
        if (not os.path.exists(ckpt_save_path)):
            os.makedirs(ckpt_save_path)

        saver_ckpt = tf.train.Saver()

        if (is_train == False):
            return

        best = 111111
        for cur_epochs in range(nb_epochs):
            samples = utils.sampling(dataset, 0)
            batchs = utils.get_batchs(samples, FLAGS.batch_size)
            for i in range(len(batchs)):
                users, items, rates = batchs[i]
                feed_dict = {self.users_holder: users,
                             self.items_holder: items,
                             self.ratings_holder: rates}
                self.sess.run([self.train_op], feed_dict)

            # evaluation
            pre_rate = self.sess.run(self.rate)
            count = 0
            for i in range(len(self.dataset.testRatings)):
                uu, ii, rr = self.dataset.testRatings[i]
                count += (rr * self.dataset.max_rate - pre_rate[uu, ii] * self.dataset.max_rate) * (
                        rr * self.dataset.max_rate - pre_rate[uu, ii] * self.dataset.max_rate)
            count /= len(self.dataset.testRatings)

            print("cur_epochs: ", cur_epochs, count)
            if (count < best):
                best = count
                saver_ckpt.save(self.sess, ckpt_save_path)
        saver_ckpt.restore(self.sess, ckpt_save_path)

    def train(self):
        assignments, centers = k_means(self.dataset.trainMatrix.toarray(), self.num_cluster)
        np.save("temp/%s/centers_%d_%f.npy" % (FLAGS.dataset, FLAGS.target_item[0], FLAGS.data_size), centers)
        np.save("temp/%s/assignments_%d_%f.npy" % (FLAGS.dataset, FLAGS.target_item[0], FLAGS.data_size), assignments)
        self.assignments = assignments
        centers = np.array(centers)
        epochs = 500
        pre_epochs = 300
        if (FLAGS.dataset == 'filmtrust'):
            pre_epochs = 100
        G_step = 2
        D_step = 1
        I_step = 1
        G_batch = 64
        D_batch = 32
        all_user = self.dataset.trainMatrix.toarray()

        print("begin training")

        # Group users
        assign_list = []
        for i in range(self.num_cluster):
            assign_list.append(np.where(assignments == i)[0])

        # The number of items selected by each group
        selected_items_num = np.zeros((self.num_cluster, 2))
        for i in range(self.num_cluster):
            idx = np.where(assignments == i)[0]
            ii = np.sum(all_user[idx] != 0, axis=1)
            selected_items_num[i, 0] = np.mean(ii) - len(FLAGS.target_item)
            selected_items_num[i, 1] = 0
        np.save("temp/%s/per_user_%d_%f.npy" % (FLAGS.dataset, FLAGS.target_item[0], FLAGS.data_size),
                selected_items_num)
        self.sample_list(centers.copy())

        self.pred_list = []
        for i in range(self.i_num):
            self.influence_user(FLAGS.target_item, all_user)
        train_user, influence_label = self.prepare_influence_data(assignments, selected_items_num)

        # Influence value range, which is used for normalization
        if_placeholder = {self.min_if_ph: self.min_inf, self.max_if_ph: self.max_inf}

        # pretrain influence module
        for d_i1 in range(1000):
            inf_idx = np.random.choice(np.arange(len(train_user)), (128), replace=False)
            cur_train = train_user[inf_idx]
            cur_label = influence_label[inf_idx][:, None]
            self.sess.run(self.i_train_op_pre, feed_dict={self.inf_user_train: cur_train, self.inf_label: cur_label})
            # if (d_i1 % 1000 == 0):
            #     inf_idx = np.random.choice(np.arange(len(train_user)), (128))
            #     cur_train = train_user[-64:]
            #     cur_label = influence_label[-64:][:, None]
            #     loss = self.sess.run(self.i_loss2,
            #                          feed_dict={self.inf_user_train: cur_train, self.inf_label: cur_label})
            #     print(loss)

        d_batch1, d_batch2, d_batch3 = self.train_influence(D_batch)
        g_batch1, g_batch2, g_batch3 = self.train_influence(G_batch)
        for i in range(epochs):

            # train discriminator
            idx = np.arange(self.dataset.num_users)[:D_batch]
            c_idx = np.random.choice(assignments, (D_batch), False)
            for k in range(D_batch):
                idx[k] = np.random.choice(assign_list[c_idx[k]], 1)[0]
            z_batch = np.zeros((D_batch, self.num_items))
            mask = np.zeros((D_batch, self.dataset.num_items))
            for k in range(D_batch):
                t = int(selected_items_num[c_idx[k], 0])
                ci = self.fast_select(c_idx[k], t)
                ci = np.concatenate([np.array(ci), np.array(FLAGS.target_item)])
                mask[k, ci] = 1
                for kk in range(len(ci)):
                    t = np.random.normal(self.dataset.distribution1[ci[kk]][0], self.dataset.distribution1[ci[kk]][1],
                                         1)
                    z_batch[k, ci[kk]] = t
            z_batch = np.clip(z_batch, 0, 1)
            z_batch = np.round(z_batch * self.dataset.max_rate) / self.dataset.max_rate
            feed1 = {place: cur for place, cur in zip(self.value1, d_batch1)}
            feed2 = {place: cur for place, cur in zip(self.value2, d_batch2)}
            feed3 = {place: cur for place, cur in zip(self.ref_user, d_batch3)}

            true_label = self.get_influence(self.dataset.trainMatrix[idx].toarray())[:, None]
            for d_i in range(D_step):
                fake_input = z_batch
                real_input = self.dataset.trainMatrix[idx].toarray()
                self.sess.run(self.d_train_op,
                              feed_dict={self.z: fake_input, self.d: real_input, self.mask: mask,
                                         self.truelabel: true_label, self.inf_user: real_input, **feed1, **feed2,
                                         **feed3, **if_placeholder})

            # train generator
            idx = np.arange(self.dataset.num_users)[:G_batch]
            c_idx = np.random.choice(assignments, (G_batch), False)
            for k in range(G_batch):
                idx[k] = np.random.choice(assign_list[c_idx[k]], 1)[0]
            z_batch = np.zeros((G_batch, self.num_items))
            mask = np.zeros((G_batch, self.dataset.num_items))
            for k in range(G_batch):
                t = int(selected_items_num[c_idx[k], 0])
                ci = self.fast_select(c_idx[k], t)
                ci = np.concatenate([np.array(ci), np.array(FLAGS.target_item)])
                mask[k, ci] = 1
                for kk in range(len(ci)):
                    t = np.random.normal(self.dataset.distribution1[ci[kk]][0], self.dataset.distribution1[ci[kk]][1],
                                         1)
                    z_batch[k, ci[kk]] = t
            z_batch = np.clip(z_batch, 0, 1)
            z_batch = np.round(z_batch * self.dataset.max_rate) / self.dataset.max_rate
            feed1 = {place: cur for place, cur in zip(self.value1, g_batch1)}
            feed2 = {place: cur for place, cur in zip(self.value2, g_batch2)}
            feed3 = {place: cur for place, cur in zip(self.ref_user, g_batch3)}
            for g_i in range(G_step):
                g_input = z_batch
                if (i < pre_epochs):
                    self.sess.run(self.g_train_op_pre,
                                  feed_dict={self.z: g_input, self.mask: mask, self.center: z_batch, **feed1, **feed2,
                                             **feed3, **if_placeholder})
                else:
                    self.sess.run(self.g_train_op,
                                  feed_dict={self.z: g_input, self.mask: mask, self.center: z_batch, **feed1, **feed2,
                                             **feed3, **if_placeholder})

            # train influence
            for d_i1 in range(I_step):
                inf_idx = np.random.choice(np.arange(len(train_user)), (128), False)
                cur_train = train_user[inf_idx]
                cur_label = influence_label[inf_idx][:, None]

                idx = np.random.choice(np.arange(self.dataset.num_users), (128), False)
                true_user = self.dataset.trainMatrix[idx].toarray()
                self.sess.run(self.i_train_op,
                              feed_dict={self.inf_user_train: cur_train, self.inf_label: cur_label,
                                         self.inf_user: true_user})

            if (i % 10 == 0):
                idx = np.arange(self.dataset.num_users)[:D_batch]
                c_idx = np.random.randint(0, self.num_cluster, (D_batch))
                c_idx = np.random.choice(assignments, (D_batch))
                for k in range(D_batch):
                    idx[k] = np.random.choice(assign_list[c_idx[k]], 1)[0]
                z_batch = np.zeros((D_batch, self.num_items))
                mask = np.zeros((D_batch, self.dataset.num_items))
                for k in range(D_batch):
                    t = int(selected_items_num[c_idx[k], 0])
                    ci = self.fast_select(c_idx[k], t)
                    ci = np.concatenate([np.array(ci), np.array(FLAGS.target_item)])
                    mask[k, ci] = 1
                    for kk in range(len(ci)):
                        t = np.random.normal(self.dataset.distribution1[ci[kk]][0],
                                             self.dataset.distribution1[ci[kk]][1],
                                             1)
                        z_batch[k, ci[kk]] = t
                z_batch = np.clip(z_batch, 0, 1)
                z_batch = np.round(z_batch * self.dataset.max_rate) / self.dataset.max_rate
                feed1 = {place: cur for place, cur in zip(self.value1, d_batch1)}
                feed2 = {place: cur for place, cur in zip(self.value2, d_batch2)}
                feed3 = {place: cur for place, cur in zip(self.ref_user, d_batch3)}

                true_label = self.get_influence(self.dataset.trainMatrix[idx].toarray())[:, None]

                real_user = self.dataset.trainMatrix[idx].toarray()
                fake_input = z_batch
                real_input = real_user

                # inf user
                inf_idx = np.random.choice(np.arange(len(train_user)), (64))
                cur_train = train_user[inf_idx]
                cur_label = influence_label[inf_idx][:, None]

                g_loss = self.sess.run(
                    [self.g_loss1, self.g_loss4, self.influence, self.influence_false],
                    feed_dict={self.z: fake_input, self.mask: mask, self.center: z_batch, **feed1,
                               **feed2, **feed3, **if_placeholder})
                d_loss = self.sess.run([self.d_loss1, self.i_loss2, self.i_loss],
                                       feed_dict={self.z: fake_input, self.d: real_input, self.mask: mask,
                                                  self.inf_user_train: cur_train, self.inf_label: cur_label,
                                                  self.truelabel: true_label, self.inf_user: real_input, **feed1,
                                                  **feed2, **feed3, **if_placeholder})
                print("cur epochs %d: g_loss: %.4f %.4f %.4f %.4f d_loss: %.4f %.4f %.4f" % (
                    i, g_loss[0], g_loss[1], np.mean(g_loss[2]), np.mean(g_loss[3]), d_loss[0], d_loss[1], d_loss[2]))
        saver = tf.train.Saver()
        saver.save(self.sess, "pretrain/gan/model_%s_%d.ckpt" % (FLAGS.dataset, FLAGS.target_item[0]))

    def influence_user(self, target_item, all_user):
        params = [self.user_embeddings, self.item_embeddings]
        self.sess.run(tf.variables_initializer(params))
        self.svd_train(self.dataset, True, FLAGS.epochs)

        scale = 10
        i_epochs = 20000
        if (FLAGS.dataset == 'ml-1m'):
            scale = 30
            i_epochs = 30000
        if (FLAGS.dataset == 'filmtrust'):
            scale = 10

        dty = tf.float32
        v_cur_est = [tf.placeholder(dty, shape=a.get_shape(), name="v_cur_est" + str(i)) for i, a in enumerate(params)]
        Test = [tf.placeholder(dty, shape=a.get_shape(), name="test" + str(i)) for i, a in enumerate(params)]

        hessian_vector_val = utils.hessian_vector_product(self.loss, params, v_cur_est, scale, True)
        estimation_IHVP = [g + cur_e - HV
                           for g, HV, cur_e in zip(Test, hessian_vector_val, v_cur_est)]

        rate_mask = self.dataset.trainMatrix.toarray() == 0
        rate_mask[:, target_item] = True
        # define loss, gradient
        sorted_rate = tf.sort(self.rate * rate_mask + (1 - rate_mask) * -9999, axis=1, direction='DESCENDING')
        attack_loss = tf.reduce_sum(tf.add_n([tf.log(
            1. / (1. + tf.exp(
                -(self.rate[:self.dataset.origin_num_users, t][:, None] - sorted_rate[
                                                                          :self.dataset.origin_num_users,
                                                                          :10])))) for t in
            target_item])) / self.dataset.num_users
        attack_grad = tf.gradients(attack_loss, params)
        per_rate = tf.matmul(self.p_u, tf.transpose(self.item_embeddings))
        per_loss1 = tf.nn.l2_loss(
            tf.subtract(tf.transpose(self.ratings_holder), per_rate)) / self.dataset.num_items * FLAGS.batch_size
        per_loss = tf.add(per_loss1,
                          0.01 * tf.add_n([tf.nn.l2_loss(v) for v in [self.user_embeddings, self.item_embeddings]]))
        train_grad = tf.gradients(per_loss, params)

        # IHVP
        import time
        start_time = time.time()
        test_val = self.sess.run(attack_grad)
        print("test_val", np.sum(test_val[1] != 0))
        cur_estimate = test_val.copy()
        feed1 = {place: cur for place, cur in zip(Test, test_val)}
        samples = utils.sampling(self.dataset, 0)
        pre_norm = -11111
        for j in range(i_epochs):
            feed2 = {place: cur for place, cur in zip(v_cur_est, cur_estimate)}
            r = np.random.choice(len(samples[0]), size=[FLAGS.batch_size], replace=False)
            # r=np.arange(len(samples[0]))
            users, items, rates = samples[0][r], samples[1][r], samples[2][r]
            feed_dict = {self.users_holder: users,
                         self.items_holder: items,
                         self.ratings_holder: rates}
            cur_estimate = self.sess.run(estimation_IHVP, feed_dict={**feed_dict, **feed1, **feed2})
            if j % 500 == 0 and j > 0:
                cur_norm = np.linalg.norm(cur_estimate[0])
                if (j % 2500 == 0):
                    print("Inverse HVP epoch:", j, cur_norm)
                if (abs(cur_norm - pre_norm) < 0.005):
                    print("stop early!!!")
                    break
                pre_norm = cur_norm
        inverse_hvp1 = [b / scale for b in cur_estimate]
        inverse_hvp = [np.reshape(v, [-1]) for v in inverse_hvp1]
        duration = time.time() - start_time
        print('Inverse HVP by HVPs+Lissa: took %s minute %s sec' % (duration // 60, duration % 60))

        assignments = np.load("temp/%s/assignments_%d_%f.npy" % (FLAGS.dataset, FLAGS.target_item[0], FLAGS.data_size))
        select_list = []
        sidx = utils.cal_neighbor(all_user, all_user, self.count)
        for i in range(np.max(assignments) + 1):
            select_list.append(sidx)
        self.save_list = []
        temp_dict = {}
        for k in range(len(select_list)):
            select_users = select_list[k]
            cur_list = []
            for j in range(len(select_users)):
                i = select_users[j]
                cur_user = all_user[i]
                if (i in temp_dict):
                    val_lissa = temp_dict[i][0]
                    val_lissa1 = temp_dict[i][1]
                else:
                    user = np.array([[i]], dtype=np.int)
                    feed_dict = {self.users_holder: user,
                                 self.ratings_holder: cur_user[:, None],
                                 self.rate_mask: (cur_user != 0)[None, :]}
                    train_grad_loss_val = self.sess.run(train_grad, feed_dict=feed_dict)
                    train_grad_loss_val = [np.reshape(utils.convert_slice_to_dense(v), [-1]) for v in train_grad_loss_val]
                    val_lissa = -np.dot(np.concatenate(inverse_hvp), np.concatenate(train_grad_loss_val))
                    feed2 = {place: cur for place, cur in zip(v_cur_est, inverse_hvp1)}
                    pert_vector_val = utils.pert_vector_product(per_loss, params, self.ratings_holder,
                                                                v_cur_est, True)
                    val_lissa1 = self.sess.run(pert_vector_val, feed_dict={**feed_dict, **feed2})
                    temp_dict[i] = [val_lissa, val_lissa1]
                    self.pred_list.append([val_lissa, val_lissa1, cur_user])
                cur_list.append([val_lissa, np.concatenate(val_lissa1), cur_user])
            self.save_list.append(cur_list)


    # Generate influence training data
    def prepare_influence_data(self, assignments, per_user):
        if (FLAGS.load_inf == False):
            generator_num = 20000
            if (FLAGS.dataset == 'ml-1m'):
                generator_num = 40000
            if (FLAGS.dataset == 'filmtrust'):
                generator_num = 40000
            c_idx = np.random.choice(assignments, (generator_num))
            z_batch = np.zeros((generator_num, self.dataset.num_items))
            for k in range(generator_num):
                t = int(per_user[c_idx[k], 0])
                ci = np.random.choice(np.arange(self.dataset.num_items), t, replace=False)
                ci = np.concatenate([np.array(ci), np.array(FLAGS.target_item)])
                for kk in range(len(ci)):
                    t = np.random.normal(self.dataset.distribution1[ci[kk]][0], self.dataset.distribution1[ci[kk]][1],
                                         1)
                    z_batch[k, ci[kk]] = t
            z_batch[:, FLAGS.target_item] = 1.
            z_batch = np.clip(z_batch, 0, 1)
            generator_user = np.round(z_batch * self.dataset.max_rate) / self.dataset.max_rate
            label = np.zeros((generator_num))
            for i in range(generator_num):
                for j in range(len(self.pred_list)):
                    val_lissa, val_lissa1, cur_user = self.pred_list[j]
                    label[i] += (np.sum([-np.dot(v.T, (generator_user[i] - cur_user)) for v in val_lissa1]))
            label /= len(self.pred_list)
            np.save("temp/%s/train_generator_user_%d_%f.npy" % (FLAGS.dataset, FLAGS.target_item[0], FLAGS.data_size),
                    generator_user)
            np.save("temp/%s/train_generator_label_%d_%f.npy" % (FLAGS.dataset, FLAGS.target_item[0], FLAGS.data_size),
                    label)
            np.save("temp/%s/lissa_%d_%f.npy" % (FLAGS.dataset, FLAGS.target_item[0], FLAGS.data_size), self.pred_list)
        generator_user = np.load(
            "temp/%s/train_generator_user_%d_%f.npy" % (FLAGS.dataset, FLAGS.target_item[0], FLAGS.data_size))
        label = np.load(
            "temp/%s/train_generator_label_%d_%f.npy" % (FLAGS.dataset, FLAGS.target_item[0], FLAGS.data_size))
        self.pred_list = np.load("temp/%s/lissa_%d_%f.npy" % (FLAGS.dataset, FLAGS.target_item[0], FLAGS.data_size),
                                 allow_pickle=True)
        self.max_inf = np.max(label)
        self.min_inf = np.min(label)
        label = (label - self.min_inf) / (self.max_inf - self.min_inf) * 2 - 1
        return generator_user, label

    def get_influence(self, user):
        label = np.zeros(len(user))
        for i in range(len(user)):
            for j in range(len(self.pred_list)):
                val_lissa, val_lissa1, cur_user = self.pred_list[j]
                label[i] += (val_lissa + np.sum([-np.dot(v.T, (user[i] - cur_user)) for v in val_lissa1]))
        label /= len(self.pred_list)
        label = (label - self.min_inf) / (self.max_inf - self.min_inf) * 2 - 1
        return label

    def generator_user(self):
        if (FLAGS.dataset == 'ml-1m'):
            generator_num = 5000
        else:
            generator_num = 1000
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, "pretrain/gan/model_%s_%d.ckpt" % (FLAGS.dataset, FLAGS.target_item[0]))

        centers = np.load("temp/%s/centers_%d_%f.npy" % (FLAGS.dataset, FLAGS.target_item[0], FLAGS.data_size))
        self.sample_list(centers.copy())
        self.assignments = np.load(
            "temp/%s/assignments_%d_%f.npy" % (FLAGS.dataset, FLAGS.target_item[0], FLAGS.data_size))
        per_user = np.load("temp/%s/per_user_%d_%f.npy" % (FLAGS.dataset, FLAGS.target_item[0], FLAGS.data_size))

        idx = np.random.choice(np.arange(self.dataset.num_users), (generator_num))
        np.save("temp/%s/assignments_poison_%d_%f.npy" % (FLAGS.dataset, FLAGS.target_item[0], FLAGS.data_size), idx)
        idx = np.load("temp/%s/assignments_poison_%d_%f.npy" % (FLAGS.dataset, FLAGS.target_item[0], FLAGS.data_size))
        print(idx.shape)
        c_idx = self.assignments[idx]
        z_batch1 = self.dataset.trainMatrix[idx].toarray()
        z_batch = np.zeros_like(z_batch1)
        mask = np.zeros((generator_num, self.dataset.num_items))

        for k in range(len(z_batch1)):
            t = int(np.maximum(np.random.normal(per_user[c_idx[k], 0], per_user[c_idx[k], 1], 1), 0))
            ci = self.fast_select(c_idx[k], t)
            # ci = np.random.choice(np.arange(self.dataset.num_items), t, replace=False)
            ci = np.concatenate([np.array(ci), np.array(FLAGS.target_item)])
            mask[k, ci] = 1
            for kk in range(len(ci)):
                t = np.random.normal(self.dataset.distribution1[ci[kk]][0], self.dataset.distribution1[ci[kk]][1], 1)
                z_batch[k, ci[kk]] = t
        z_batch = np.clip(z_batch, 0, 1)
        z_batch = np.round(z_batch * self.dataset.max_rate) / self.dataset.max_rate

        generator_users, influence_false, inf_true = sess.run(
            [self.generator, self.influence_false, self.influence_true],
            feed_dict={self.z: z_batch, self.mask: mask, self.inf_user: z_batch})
        # print(influence_false-inf_true)
        print(np.mean(influence_false - inf_true))
        generator_users = np.clip(generator_users, 0., 1.)
        generator_users *= mask
        generator_users = np.round(generator_users * self.dataset.max_rate) / self.dataset.max_rate
        assign = assign_points(generator_users, centers)

        np.save("temp/%s/atk/generator_user_%d_%f.npy" % (FLAGS.dataset, FLAGS.target_item[0], FLAGS.data_size),
                generator_users)
        np.save("temp/%s/atk/influence_false_%d_%f.npy" % (FLAGS.dataset, FLAGS.target_item[0], FLAGS.data_size),
                influence_false)
        np.save("temp/%s/atk/influence_true_%d_%f.npy" % (FLAGS.dataset, FLAGS.target_item[0], FLAGS.data_size),
                inf_true)
        print(assign)
        print(np.sum(assign == c_idx))
        num_count = []
        for i in range(np.max(self.assignments) + 1):
            num_count.append(np.sum(self.assignments == i))
        print(num_count)
        print(per_user)

    def sample_list(self, cs):
        all_list = []
        for i in range(len(cs)):
            center = cs[i]
            center /= np.sum(center)
            all_list.append(center)
        self.all_list = all_list

    def fast_select(self, i, length):
        list = self.all_list[i]
        t = np.random.choice(np.arange(self.dataset.num_items), (length), False, p=list.ravel())
        return t

    def train_influence(self, batch_size):
        self.pred_list = np.load("temp/%s/lissa_%d_%f.npy" % (FLAGS.dataset, FLAGS.target_item[0], FLAGS.data_size),
                                 allow_pickle=True)
        batch1 = []
        batch2 = []
        batch3 = []
        for i in range(len(self.pred_list)):
            b1, b2, b3 = self.pred_list[i]
            batch1.append(np.ones((batch_size, 1)) * b1)
            batch2.append(np.repeat(np.concatenate(b2)[None, :], batch_size, axis=0)[:, :, 0])
            batch3.append(np.repeat(np.concatenate([b3, b3])[None, :], batch_size, axis=0))
        return batch1, batch2, batch3
