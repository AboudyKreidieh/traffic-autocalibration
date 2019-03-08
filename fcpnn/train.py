"""

"""
from model import NN, PNN, FCPNN
import numpy as np
import tensorflow as tf
from collections import defaultdict
import random
import os
import errno
import csv

# number of layers in the neural network model
LAYERS = [1, 2, 3, 4]
# number of nodes in each hidden layer
NODES = [32, 128]
# learning rate for the Adam Optimizer (performed on a log scale)
LEARNING_RATES = [-3, -1]
# number of hyper-parameter tuning trails
N_TRIALS = 10
# number of seeds perform for each hyper-parameter trial
N_SEEDS = 5
# number of input features
N_INPUTS = 4
# number of training iterations
N_ITER = 2000


def ensure_dir(path):
    """Ensure that the directory specified exists, and if not, create it."""
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return path


def train_model(model, dataset, fd):
    """

    Parameters
    ----------
    model : Model
        the neural network model used for regression
    dataset : array_like
        dataset used for training and testing, where the last column is the
        actual output values
    fd : str
        file directory where the results should be saved
    """
    res = []
    for itr in range(N_ITER):
        loss = train_step(model, dataset, dataset, fd, itr)
        res.append(loss)
    return res


def train_step(model, train_set, test_set, fd, itr):
    """Perform a single step of training.

    This includes: updating the weights, saving the new model as a checkpoint,
    and printing/returns the next mean/std loss.

    Parameters
    ----------
    model : Model
        the neural network model used for regression
    train_set : array_like
        training dataset, where the last column is the actual output values
    test_set : array_like
        testing dataset, where the last column is the actual output values
    fd : str
        file directory for saving checkpoint
    itr : int
        training iteration

    Return
    ------
    float
        loss mean
    float
        loss standard deviation
    """
    loss = model.train(train_set)
    # model.save(fd, itr)

    print("Iter {} Return: {}, {}".format(itr, np.mean(loss), np.std(loss)))
    return loss


if __name__ == '__main__':
    file = '../data_collection/data/ring-0.csv'
    columns = []
    with open(file, 'rU') as f:
        reader = csv.reader(f)
        for row in reader:
            if columns:
                for i, value in enumerate(row):
                    columns[i].append(value)
            else:
                columns = [[value] for value in row]

    # you now have a column-major 2D array of your file.
    as_dict = {c[0]: c[1:] for c in columns}

    # the final dataset
    dataset = np.asarray([[float(x) for x in as_dict['headway']],
                          [float(x) for x in as_dict['speed']],
                          [float(x) for x in as_dict['prev_accel']],
                          [float(x) for x in as_dict['lead_speed']],
                          [float(x) for x in as_dict['accel']]]).T
    dataset = dataset[100:, :]

    # hyper-parameter search
    for _ in range(N_TRIALS):
        # extract a list of hyper-parameters
        layer = random.choice(LAYERS)
        node = int(random.uniform(NODES[0], NODES[1]))
        lr = 10 ** random.uniform(LEARNING_RATES[0], LEARNING_RATES[1])

        # create the necessary directories to store the results and meta-data
        ensure_dir('nn')
        ensure_dir('nn_d')
        ensure_dir('nn_bn')

        # training for BLANK different seeds
        loss_data = defaultdict(list)
        for i in range(N_SEEDS):
            # training on a generic neural network model
            model = NN(inputs=N_INPUTS,
                       hidden_size=[node for _ in range(layer)],
                       act_funcs=[tf.nn.relu for _ in range(layer)],
                       batch_size=128,
                       learning_rate=lr,
                       dropout=False,
                       batch_norm=False)

            fd = 'nn/{}_{}_{}'.format(layer, node, lr, i)
            losses = train_model(model, dataset, fd)
            loss_data[fd].append(losses)

            # training on a generic neural network model with dropout
            model = NN(inputs=N_INPUTS,
                       hidden_size=[n for _ in range(l)],
                       act_funcs=tf.nn.relu,
                       dropout=True,
                       batch_norm=False)

            fd = 'nn_d/{}_{}_{}'.format(layer, node, lr, i)
            losses = train_model(model, dataset, fd)
            loss_data[fd].append(losses)

            # training on a generic neural network model with batch norm
            model = NN(inputs=N_INPUTS,    # get training and testing datasets

                       hidden_size=[n for _ in range(l)],
                       act_funcs=tf.nn.relu,
                       dropout=False,
                       batch_norm=True)

            fd = 'nn_bn/{}_{}_{}'.format(layer, node, lr, i)
            losses = train_model(model, dataset, fd)
            loss_data[fd].append(losses)

        # save the losses to csv
        for fd in loss_data.keys():
            np.savetxt(os.path.join(fd, 'results.csv'),
                       np.asarray(loss_data[fd]), delimiter=',')
