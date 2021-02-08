import numpy as np
import tensorflow as tf
import scipy.sparse as sp
import copy
from k_means import *

flags = tf.flags
FLAGS = flags.FLAGS


def GAN_attack(RS, dataset, attack_size, filler_size, target_item):
    if (FLAGS.attack_load == False):
        all_user = dataset.trainMatrix.toarray()
        generator_user = np.load("temp/%s/atk/generator_user_%d_%f.npy" % (FLAGS.dataset, FLAGS.target_item[0],FLAGS.data_size))
        # generator_user[:, target_item] = 1
        centers = np.load("temp/%s/centers_%d_%f.npy" % (FLAGS.dataset, FLAGS.target_item[0],FLAGS.data_size))
        assignments = np.load("temp/%s/assignments_%d_%f.npy" % (FLAGS.dataset, FLAGS.target_item[0],FLAGS.data_size))
        assign_idx = np.load("temp/%s/assignments_poison_%d_%f.npy" % (FLAGS.dataset, FLAGS.target_item[0],FLAGS.data_size))
        # assign = assign_points(generator_user, centers)
        assign = assignments[assign_idx]
        user_list = group(generator_user, centers)
        print(len(user_list[0]), len(user_list[1]), len(user_list[2]), len(user_list[3]), len(user_list[4]), )
        attack_size = int(dataset.full_num_users * attack_size)
        attack_data = np.zeros((attack_size, dataset.num_items))
        count = 0
        get = choice_users(assignments, attack_size)
        print(get)
        np.random.shuffle(get)

        influence = np.load("temp/%s/atk/influence_false_%d_%f.npy" % (FLAGS.dataset,FLAGS.target_item[0],FLAGS.data_size))[:, 0]
        for i in range(np.max(assignments) + 1):
            cur_idx = np.where(assign == i)[0]
            cur_if = influence[cur_idx]
            length = len(np.where(get == i)[0])
            true_idx = cur_idx[np.argsort(cur_if)[::-1][:length]]
            print(influence[true_idx])
            attack_data[count:count + length] = generator_user[true_idx]
            count += length
        print("cur user: ", count)
        np.save("temp/%s/full/GAN_poisoning_%d_%d_%f.npy" % (
        FLAGS.dataset, FLAGS.target_item[0], attack_size, FLAGS.data_size), attack_data)
    attack_data = np.load("temp/%s/full/GAN_poisoning_%d_%d_%f.npy" % (
        FLAGS.dataset, FLAGS.target_item[0], attack_size, FLAGS.data_size))
    return estimate_dataset(dataset, attack_data)

def estimate_dataset(dataset, initial_data):
    initial_data[np.abs(initial_data) < 1e-3] = 0
    new_dataset = copy.deepcopy(dataset)
    for i in range(initial_data.shape[0]):
        item = []
        for j in range(initial_data.shape[1]):
            if (abs(initial_data[i, j]) > 1e-3):
                item.append(j)
        new_dataset.trainList.append(item)
    csr_matrix = new_dataset.trainMatrix.tocsr()
    new_dataset.trainMatrix = sp.vstack([csr_matrix, sp.csr_matrix(initial_data)]).todok()
    new_dataset.num_users += initial_data.shape[0]
    new_dataset.allNegatives = new_dataset.load_all_negative(new_dataset.trainList)
    return new_dataset


def group(user, centers):
    assignments = assign_points(user, centers)
    user_list = []
    for i in range(len(centers)):
        idx = np.where(assignments == i)[0]
        user_list.append(user[idx])
    return user_list


def select_group(assignments, user_list):
    idx = np.random.choice(assignments, (1))[0]
    return user_list[idx], idx


def choice_users(assignments, num):
    max = -1
    maxidx = -1
    get = []
    for i in range(np.max(assignments) + 1):
        count = np.sum(assignments == i)
        if (count > max):
            max = count
            maxidx = i
        get.append(int(round(count / len(assignments) * num)))
    if (sum(get) > num):
        get[maxidx] -= sum(get) - num
    elif (sum(get) < num):
        get[maxidx] += num - sum(get)
    samples = []
    for i in range(len(get)):
        samples += [i for j in range(get[i])]
    samples = np.array(samples)
    np.random.shuffle(samples)
    return samples
