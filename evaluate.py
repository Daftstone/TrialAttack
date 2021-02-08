import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

from Dataset import Dataset
import utils
from models.SVD import SVD


flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("dataset", "ml-100k", "Choose a dataset.")
flags.DEFINE_string('path', 'Data/', 'Input data path.')
flags.DEFINE_string('gpu', '0', 'Input data path.')
flags.DEFINE_integer('verbose', 1, 'Evaluate per X epochs.')
flags.DEFINE_integer('batch_size', 2048, 'batch_size')
flags.DEFINE_integer('epochs', 20, 'Number of epochs.')
flags.DEFINE_integer('embed_size', 64, 'Embedding size.')
flags.DEFINE_integer('dns', 0, 'number of negative sample for each positive in dns.')
flags.DEFINE_float('reg', 0.02, 'Regularization for user and item embeddings.')
flags.DEFINE_float('lr', 0.05, 'Learning rate.')
flags.DEFINE_bool('reg_data', True, 'Regularization for adversarial loss')
flags.DEFINE_string('rs', 'svd', 'recommender system')
flags.DEFINE_bool("is_train", True, "train online or load model")
flags.DEFINE_bool("attack_load", False, "train online or load model")
flags.DEFINE_bool("use_second", False, "train online or load model")
flags.DEFINE_integer("top_k", 10, "pass")
flags.DEFINE_list("target_item", [1679], "pass")
flags.DEFINE_string('pretrain', '0', 'ckpt path')
flags.DEFINE_float("attack_size", 0.03, "pass")
flags.DEFINE_string("attack_type", "GAN", "attack type")
flags.DEFINE_float("data_size", 1., "pass")
flags.DEFINE_integer('target_index', 0, 'Embedding size.')
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu


def get_rs(dataset):
    if (FLAGS.rs == 'svd'):
        rs = SVD(dataset.num_users, dataset.num_items, dataset)
    else:
        print("error")
        exit(0)
    return rs


if __name__ == '__main__':
    np.random.seed(1234)
    tf.set_random_seed(1234)

    a = [[1485, 1320, 821, 1562, 1531], [849], [1587], [816],
         [1018, 946, 597, 575, 516], [383], [1575], [1333],
         [3639, 3698, 3622, 3570, 3503], [3667], [3700], [3528],
         [1032, 3033, 2797, 2060, 1366], [495], [1829], [1899],
         [1576, 926, 942, 848, 107], [848], [107], [1343],
         [539, 117, 1600, 1326, 208], [436], [825], [558],
         [2504, 19779,  9624, 24064, 17390],
         [2417, 21817, 13064, 3348, 15085]]
    FLAGS.target_item = a[FLAGS.target_index]
    # initialize dataset
    dataset = Dataset(FLAGS.path + FLAGS.dataset, FLAGS.reg_data)
    # print(dataset.num_items, dataset.num_users, dataset.avg_items)
    # FLAGS.target_item = [1485, 1320,  821, 1562, 1531]

    item_count = dataset.trainMatrix.tocsr().getnnz(0)
    idx = np.where(item_count == 10)[0]
    np.random.shuffle(idx)
    print(idx)

    RS = get_rs(dataset)
    RS.build_graph()
    print("Initialize %s" % FLAGS.rs)

    # start training
    RS.train(dataset, FLAGS.is_train, FLAGS.epochs)

    # target item recommendation
    print("origin: target item: ", FLAGS.target_item)
    # utils.recommend(RS, dataset, FLAGS.target_item, FLAGS.top_k)

    hr_list = []
    ndcg_list = []
    # attack
    dataset = utils.attack(RS, dataset, FLAGS.attack_size, dataset.avg_items, FLAGS.target_item)
    for i in range(1):
        print("cur ", i)
        RS = get_rs(dataset)
        tf.reset_default_graph()
        RS.build_graph()
        RS.train(dataset, True, FLAGS.epochs)
        # target item recommendation
        print("after attack: target item: ", FLAGS.target_item)
        hr, ndcg = utils.recommend(RS, dataset, FLAGS.target_item, FLAGS.top_k)
        hr_list.append(hr)
        ndcg_list.append(ndcg)
        print(FLAGS.attack_type)
    print("results")
    print(hr_list)
    print(np.mean(hr_list))
    print(np.mean(ndcg_list))
