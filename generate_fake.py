from Dataset import Dataset
import numpy as np
import os
import tensorflow as tf
from TrialAttack import GAN
import utils

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("dataset", "ml-100k", "Choose a dataset.")
flags.DEFINE_integer('batch_size', 2048, 'batch_size')
flags.DEFINE_integer('epochs', 20, 'training epochs')
flags.DEFINE_list("target_item", [1485], "target items for attacking, the option is invalid")
flags.DEFINE_string('gpu', '3', 'GPU ID')
flags.DEFINE_bool("load_inf", False, "load influence from file or calculate online")
flags.DEFINE_float("data_size", 1., "The data available to the attacker")
flags.DEFINE_integer('target_index', 0, 'target items for attacking')

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

a = [[1485, 1320, 821, 1562, 1531], [849], [1587], [816],
     [1018, 946, 597, 575, 516], [383], [1575], [1333],
     [3639, 3698, 3622, 3570, 3503], [3667], [3700], [3528],
     [1032, 3033, 2797, 2060, 1366], [495], [1829], [1899],
     [1576, 926, 942, 848, 107], [848], [107], [1343],
     [539, 117, 1600, 1326, 208], [436], [825], [558]]
FLAGS.target_item = a[FLAGS.target_index]
dataset = Dataset("Data/%s" % FLAGS.dataset, True)

# The mode of each item of the three datasets is 0, so simplified here.
temp = np.zeros((1, dataset.num_items))
temp[0, FLAGS.target_item] = 1

dataset = utils.estimate_dataset(dataset, temp)
dataset.origin_num_users += 1
gan = GAN(dataset, 5, dataset.num_items)
gan.svd_train(dataset, True, FLAGS.epochs)
gan.train()
gan.generator_user()
