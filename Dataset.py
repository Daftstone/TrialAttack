'''
Created on Aug 8, 2016
Processing datasets. 
@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import scipy.sparse as sp
import numpy as np
from time import time
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS


class Dataset(object):
    '''
    Loading the data file
        trainMatrix: load rating records as sparse matrix for class Data
        trianList: load rating records as list to speed up user's feature retrieval
        testRatings: load leave-one-out rating test for class Evaluate
        testNegatives: sample the items not rated by user
    '''

    def __init__(self, path, reg_data):
        '''
        Constructor
        '''
        self.reg_data = FLAGS.reg_data
        if (FLAGS.dataset == 'filmtrust'):
            self.max_rate = 8
        else:
            self.max_rate = 5
        self.trainMatrix = self.load_training_file_as_matrix(path + ".train.rating")
        self.trainList = self.load_training_file_as_list(path + ".train.rating")
        self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
        self.testNegatives = self.load_negative_file(path + ".test.negative")
        assert len(self.testRatings) == len(self.testNegatives)
        self.num_users, self.num_items = self.trainMatrix.shape
        self.origin_num_users = self.num_users
        self.full_num_users = self.num_users
        if ('yelp' not in FLAGS.dataset):
            self.allNegatives = self.load_all_negative(self.trainList)
        self.distribution1 = self.item_distribution()
        np.save("dis1.npy", self.distribution1)
        if (FLAGS.data_size < 1):
            print("get_sub")
            self.get_subdata(FLAGS.data_size)

        np.save("train_user_yelp.npy",self.trainMatrix.toarray())

    def item_distribution(self):
        dis1 = []
        trainmatrix = self.trainMatrix.toarray()
        print(trainmatrix.shape)
        for i in range(self.num_items):
            item_rate = trainmatrix[:, i]
            rate_idx1 = np.where(item_rate > 0.)[0]
            if (rate_idx1.shape[0] == 0):
                dis1.append([1., 0])
            else:
                item_rate1 = item_rate[rate_idx1]
                dis1.append([np.mean(item_rate1), np.std(item_rate1)])
            # dis.append([np.mean(item_rate[rate_idx]), np.std(item_rate[rate_idx])])
        return dis1

    def load_all_negative(self, trainList):
        allnegative = []
        for i in range(self.num_users):
            item_input = np.array(list(set(range(self.num_items)) - set(trainList[i])))
            allnegative.append(item_input)
        return allnegative

    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rate = int(arr[0]), int(arr[1]), float(arr[2])
                if (self.reg_data == True):
                    rate = rate / self.max_rate
                ratingList.append([user, item, rate])
                line = f.readline()
        return ratingList

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1:]:
                    negatives.append(int(x))
                negativeList.append(negatives[:99])
                line = f.readline()
        return negativeList

    def load_training_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    if (self.reg_data == True):
                        rating = rating / self.max_rate
                    mat[user, item] = rating
                line = f.readline()
        print("already load the trainMatrix...")
        return mat

    def load_training_file_as_list(self, filename):
        # Get number of users and items
        u_ = 0
        lists, items = [], []
        items_count = 0
        with open(filename, "r") as f:
            line = f.readline()
            index = 0
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                if u_ < u:
                    index = 0
                    lists.append(items)
                    items_count += len(items)
                    items = []
                    u_ += 1
                index += 1
                # if index<300:
                items.append(i)
                line = f.readline()
        lists.append(items)
        items_count += len(items)
        self.avg_items = items_count / len(lists)
        print("already load the trainList...")
        return lists

    def get_subdata(self, p):
        num_users = int(np.round(self.num_users * p))
        # idx = np.random.choice(np.arange(self.num_users), num_users, replace=False)
        idx = self.get_idx(num_users, self.trainMatrix.toarray(), FLAGS.target_item)
        # np.save("temp/partial/%s_%d_%f.npy" % (FLAGS.dataset, FLAGS.target_item[0], FLAGS.data_size), idx)
        # idx = np.load("temp/partial/%s_%d_%f.npy" % (FLAGS.dataset, FLAGS.target_item[0], FLAGS.data_size))
        trainMatrix = self.trainMatrix[idx]
        trainList = []
        for i in range(num_users):
            trainList.append(self.trainList[idx[i]])
        testmatrix = np.zeros_like(self.trainMatrix.toarray())
        print(testmatrix.shape)
        for i in range(len(self.testRatings)):
            ratings = self.testRatings[i]
            testmatrix[ratings[0], ratings[1]] = ratings[2]
        testmatrix = testmatrix[idx]
        testRatings = []
        for i in range(num_users):
            ii = np.where(testmatrix[i] != 0)[0]
            assert len(ii) == 1
            testRatings.append([i, ii[0], testmatrix[i, ii[0]]])
        testnegatives = []
        for i in range(num_users):
            testnegatives.append(self.testNegatives[idx[i]])
        allnegatives = []
        for i in range(num_users):
            allnegatives.append(self.allNegatives[idx[i]])

        self.trainMatrix = trainMatrix
        self.trainList = trainList
        self.testRatings = testRatings
        self.testNegatives = testnegatives
        self.num_users = num_users
        self.origin_num_users = num_users
        self.allNegatives = allnegatives
        self.distribution1 = self.item_distribution()

    def get_idx(self, num_user, all_user, init):
        cur_set = set()
        cur_item = init
        must = set()
        temp = set()
        while (len(cur_set) < num_user):
            print("test")
            must = must | temp
            user = all_user[:, cur_item]
            idx = np.where(np.sum(user, axis=1) > 0)[0]
            cur_set = cur_set | set(list(idx))
            temp = set(list(idx))
            cur_user = np.array(list(cur_set))
            item = all_user[cur_user]
            cur_item = np.where(np.sum(item, axis=0) > 0)[0]
        print(len(cur_set))
        cur_set = np.array(list(cur_set - must))
        print(len(cur_set))
        t = np.concatenate(
            [np.random.choice(cur_set, num_user - len(must), replace=False), np.array(list(must), dtype=np.int)])
        print("test", len(t))
        return t
