from time import time
from time import strftime
from time import localtime
import os
import copy
import scipy.sparse as sp
import tensorflow as tf
import numpy as np
import math
from greedy_selection import GAN_attack

flags = tf.flags
FLAGS = flags.FLAGS

def attack(RS, dataset, attack_size, avg_items, target_item):
    return GAN_attack(RS, dataset, attack_size, avg_items, target_item)

def sampling(dataset, num_neg, bpr=False):
    tt = dataset.trainMatrix.tocoo()
    user_input = np.array(tt.row)
    item_input = np.array(tt.col)
    rate_input = np.array(tt.data)
    t1, t2, t3 = [], [], []
    if (num_neg > 0):
        for i in range(dataset.num_users):
            if (len(dataset.allNegatives[i]) != 0):
                ll = int(len(dataset.trainList[i]) * num_neg)
                t1 += [i for ii in range(ll)]
                j = list(np.random.choice(dataset.allNegatives[i], ll))
                t2 += j
                t3 += [0 for ii in range(ll)]
        if (bpr == False):
            user_input = np.concatenate([user_input, np.array(t1)], axis=0)
            item_input = np.concatenate([item_input, np.array(t2)], axis=0)
            rate_input = np.concatenate([rate_input, np.array(t3)], axis=0)
    user_input = user_input[:, None]
    item_input = item_input[:, None]
    rate_input = rate_input[:, None]
    neg_item_input = np.array(t2)[:, None]
    if (bpr == True):
        return [user_input, item_input, neg_item_input]
    else:
        return [user_input, item_input, rate_input]


def get_batchs(samples, batch_size):
    length = samples[0].shape[0]
    idx = np.arange(length)
    np.random.shuffle(idx)
    samples[0] = samples[0][idx]
    samples[1] = samples[1][idx]
    samples[2] = samples[2][idx]
    num = (length - 1) // batch_size + 1
    batchs = []
    for i in range(num):
        begin = i * batch_size
        end = i * batch_size + batch_size
        batchs.append([samples[0][begin:end], samples[1][begin:end], samples[2][begin:end]])
    return batchs


def recommend(model, dataset, target_item, _k):
    rate = model.sess.run(model.rate)
    user = dataset.trainMatrix.toarray()
    mask = user != 0
    rate[mask] = -np.inf
    count = 0
    ndcg_count = 0
    import math
    for i in range(dataset.origin_num_users):
        idx = np.argsort(rate[i])[::-1][:_k]
        for j in target_item:
            count += (j in idx)
            ndcg_count += math.log(2) / math.log(np.where(idx == j)[0] + 2) if j in idx else 0
    all_hr = count / dataset.origin_num_users / len(target_item)
    all_ndcg = ndcg_count / dataset.origin_num_users / len(target_item)
    print("recommend all user:", all_hr, all_ndcg)
    return all_hr, all_ndcg


def estimate_dataset(dataset, initial_data):
    new_dataset = copy.deepcopy(dataset)
    for i in range(initial_data.shape[0]):
        item = []
        for j in range(initial_data.shape[1]):
            if (initial_data[i, j] != 0):
                item.append(j)
        new_dataset.trainList.append(item)
    csr_matrix = new_dataset.trainMatrix.tocsr()
    new_dataset.trainMatrix = sp.vstack([csr_matrix, sp.csr_matrix(initial_data)]).todok()
    new_dataset.num_users += initial_data.shape[0]
    if (FLAGS.dataset != 'yelp'):
        new_dataset.allNegatives = new_dataset.load_all_negative(new_dataset.trainList)
    return new_dataset


def cal_neighbor(group, all_user, top_k):
    dis = np.linalg.norm(all_user, axis=1)
    idx = np.argsort(dis)[:top_k]
    idx=[len(all_user)-1]
    print("idx",idx)
    return idx

def pert_vector_product(ys, xs1, xs2, v, do_not_sum_up=True):
    # Validate the input
    length = len(xs1)
    if len(v) != length:
        raise ValueError("xs and v must have the same length.")
    # First backprop
    grads = tf.gradients(ys, xs1)

    # grads = xs
    assert len(grads) == length
    elemwise_products = [
        tf.multiply(grad_elem, tf.stop_gradient(v_elem))
        for grad_elem, v_elem in zip(grads, v) if grad_elem is not None
    ]
    # Second backprop
    if do_not_sum_up:
        seperate = []
        for i in range(length):
            seperate.append(tf.gradients(elemwise_products[i], xs2)[0])
        grads_with_none = seperate
    else:
        grads_with_none = tf.gradients(elemwise_products, xs2)

    return_grads = [grad_elem if grad_elem is not None \
                        else tf.zeros_like(xs2) \
                    for grad_elem in grads_with_none]
    return return_grads


def hessian_vector_product(ys, xs, v, do_not_sum_up=True):
    # Validate the input
    length = len(xs)
    if len(v) != length:
        raise ValueError("xs and v must have the same length.")
    # First backprop
    grads = tf.gradients(ys, xs)

    # grads = xs
    assert len(grads) == length
    elemwise_products = [
        tf.multiply(grad_elem, tf.stop_gradient(v_elem))
        for grad_elem, v_elem in zip(grads, v) if grad_elem is not None
    ]
    # Second backprop
    if do_not_sum_up:
        seperate = []
        for i in range(length):
            seperate.append(tf.gradients(elemwise_products[i], xs[i])[0])
        grads_with_none = seperate
    else:
        grads_with_none = tf.gradients(elemwise_products, xs)

    return_grads = [grad_elem if grad_elem is not None \
                        else tf.zeros_like(x) \
                    for x, grad_elem in zip(xs, grads_with_none)]
    return return_grads