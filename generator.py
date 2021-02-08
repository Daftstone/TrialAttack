from Dataset import Dataset
import numpy as np
import os
import tensorflow as tf
from TrialAttack import GAN
import utils

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_bool('reg_data', True, 'Regularization for adversarial loss')
flags.DEFINE_string("dataset", "ml-100k", "Choose a dataset.")
flags.DEFINE_integer('batch_size', 2048, 'batch_size')
flags.DEFINE_integer('epochs', 20, 'batch_size')
flags.DEFINE_list("target_item", [1485], "pass")
flags.DEFINE_string('gpu', '3', 'Regularization for adversarial loss')
flags.DEFINE_bool("is_train", False, "pass")
flags.DEFINE_bool("load_inf", False, "pass")
flags.DEFINE_float("atk_params", 2000., "pass")
flags.DEFINE_float("data_size", 1., "pass")
flags.DEFINE_integer('target_index', 0, 'Embedding size.')

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

a = [[1485, 1320, 821, 1562, 1531], [849], [1587], [816],
     [1018, 946, 597, 575, 516], [383], [1575], [1333],
     [3639, 3698, 3622, 3570, 3503], [3667], [3700], [3528],
     [1032, 3033, 2797, 2060, 1366], [495], [1829], [1899],
     [1576, 926, 942, 848, 107], [848], [107], [1343],
     [539, 117, 1600, 1326, 208], [436], [825], [558]]
FLAGS.target_item = a[FLAGS.target_index]
dataset = Dataset("Data/%s" % FLAGS.dataset, True)
alluser = dataset.trainMatrix.toarray()
meanuser=np.round(np.mean(alluser,axis=0) * dataset.max_rate) / dataset.max_rate
temp = np.zeros((1, dataset.num_items))
temp[0]=meanuser
temp[0, FLAGS.target_item] = 1
dataset = utils.estimate_dataset(dataset, temp)
dataset.origin_num_users += 1
gan = GAN(dataset, 5, dataset.num_items)
gan.svd_train(dataset, True, FLAGS.epochs)
gan.train()
gan.generator_user()
