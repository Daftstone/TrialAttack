from Dataset import Dataset
import numpy as np
import tensorflow as tf

# flags = tf.flags
# FLAGS = flags.FLAGS
# flags.DEFINE_bool('reg_data', True, 'Regularization for adversarial loss')


def point_avg(points):
    """
    Accepts a list of points, each with the same number of dimensions.
    NB. points can have more dimensions than 2

    Returns a new point which is the center of all the points.
    """
    new_center = np.mean(points, axis=0)
    return new_center


def update_centers(data_set, assignments):
    """
    Accepts a dataset and a list of assignments; the indexes
    of both lists correspond to each other.
    Compute the center for each of the assigned groups.
    Return `k` centers where `k` is the number of unique assignments.
    """
    new_center = []
    for i in range(np.max(assignments) + 1):
        idx = np.where(assignments == i)[0]
        cur_data_set = data_set[idx]
        new_center.append(point_avg(cur_data_set))
    return new_center


def assign_points(data_points, centers):
    """
    Given a data set and a list of points betweeen other points,
    assign each point to an index that corresponds to the index
    of the center point on it's proximity to that point.
    Return a an array of indexes of centers that correspond to
    an index in the data set; that is, if there are N points
    in `data_set` the list we return will have N elements. Also
    If there are Y points in `centers` there will be Y unique
    possible values within the returned list.
    """
    for i in range(len(centers)):
        cur_center = centers[i]
        cur_dis = np.linalg.norm(data_points - cur_center[None, :], axis=1, keepdims=True)
        if (i == 0):
            temp = cur_dis
        else:
            temp = np.concatenate([temp, cur_dis], axis=1)
    assignments = np.argmin(temp, axis=1)
    return assignments


def generate_k(data_set, k):
    """
    Given `data_set`, which is an array of arrays,
    Generate `k` random points between the ranges.
    Return an array of the random points within the ranges.
    """
    num_nodes = len(data_set)
    idx = np.arange(num_nodes)
    np.random.shuffle(idx)
    centers = []
    for i in range(k):
        centers.append(data_set[idx[i]])

    return centers


def k_means(dataset, k):
    k_points = generate_k(dataset, k)
    assignments = assign_points(dataset, k_points)
    old_assignments = None
    count = 0
    while (assignments != old_assignments).any():
        count += 1
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)
    return assignments, new_centers
