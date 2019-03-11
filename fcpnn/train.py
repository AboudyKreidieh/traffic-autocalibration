"""

"""
from model import NN  # , PNN, FCPNN
import numpy as np
import tensorflow as tf
from collections import defaultdict
import random
import os
import errno
import pandas as pd

# number of layers in the neural network model
LAYERS = [1, 2, 3, 4]
# number of nodes in each hidden layer
NODES = [64, 256]
# learning rate for the Adam Optimizer (performed on a log scale)
LEARNING_RATES = [-3, -1]
# number of hyper-parameter tuning trails
N_TRIALS = 10
# number of seeds perform for each hyper-parameter trial
N_SEEDS = 5
# number of input features
N_INPUTS = 4
# number of training iterations
N_ITER = 1000
# SGD batch size
BATCH_SIZE = 512
# number of SGD steps between computing the mean square error loss and updating
# the checkpoint
SAVE_STEPS = 5


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
    # perform N_ITER SGD steps and collect the mse for each step
    res = []
    for itr in range(N_ITER):
        loss = train_step(model, dataset, fd, itr)
        if loss is not None:
            res.append(loss)

    # terminate the model
    model.close()

    return res


def train_step(model, samples, fd, itr):
    """Perform a single step of training.

    This includes: updating the weights, saving the new model as a checkpoint,
    and printing/returns the next mean/std loss.

    Parameters
    ----------
    model : Model
        the neural network model used for regression
    samples : array_like
        training dataset, where the last column is the actual output values
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
    # perform an SGD step within the model
    model.train(samples)

    # save every the checkpoint every 5th training iteration
    loss = None
    if itr % SAVE_STEPS == 0:
        loss = model.compute_mse()
        model.save(fd, itr)
        print("Iter {} Return: {}".format(itr, loss))

    return loss


if __name__ == '__main__':
    # import the individual datasets
    dataset = []
    for i in range(3):
        dataset.append(pd.read_csv('../traffic_autocalibration/data_collection'
                                   '/data/ring-{}.csv'.format(i)))
    dataset = pd.concat(dataset)

    # the final dataset
    dataset = np.asarray([dataset['headway'],
                          dataset['speed'],
                          dataset['prev_accel'],
                          dataset['lead_speed'],
                          dataset['accel']]).T
    np.random.shuffle(dataset)

    # train and test samples
    test_samples = dataset[:int(dataset.shape[0]/4), :]
    train_samples = dataset[int(dataset.shape[0]/4):, :]

    # hyper-parameter search
    for _ in range(N_TRIALS):
        # extract a list of hyper-parameters
        layer = random.choice(LAYERS)
        node = random.randint(NODES[0], NODES[1])
        lr = 10 ** random.uniform(LEARNING_RATES[0], LEARNING_RATES[1])

        # training for N_SEEDS different seeds
        loss_data = defaultdict(list)
        for i in range(N_SEEDS):
            # create the necessary directories to store the results and meta-
            # data
            # ensure_dir('nn/{}l_{}n_{}lr/{}'.format(layer, node, lr, i))
            # ensure_dir('nn_d/{}l_{}n_{}lr/{}'.format(layer, node, lr, i))
            ensure_dir('nn_bn/{}l_{}n_{}lr/{}'.format(layer, node, lr, i))

            # # training on a generic neural network model
            # model = NN(inputs=N_INPUTS,
            #            hidden_size=[node for _ in range(layer)],
            #            act_funcs=[tf.nn.leaky_relu for _ in range(layer)],
            #            batch_size=BATCH_SIZE,
            #            learning_rate=lr,
            #            test_samples=test_samples,
            #            dropout=False,
            #            batch_norm=False)
            #
            # fd = 'nn/{}l_{}n_{}lr/{}'.format(layer, node, lr, i)
            # losses = train_model(model, train_samples, fd)
            # loss_data[fd].append(losses)

            # # training on a generic neural network model with dropout
            # model = NN(inputs=N_INPUTS,
            #            hidden_size=[node for _ in range(layer)],
            #            act_funcs=[tf.nn.relu for _ in range(layer)],
            #            batch_size=BATCH_SIZE,
            #            learning_rate=lr,
            #            test_samples=test_samples,
            #            dropout=True,
            #            batch_norm=False)
            #
            # fd = 'nn_d/{}l_{}n_{}lr/{}'.format(layer, node, lr, i)
            # losses = train_model(model, train_samples, fd)
            # loss_data[fd].append(losses)

            # training on a generic neural network model with batch norm
            model = NN(inputs=N_INPUTS,
                       hidden_size=[node for _ in range(layer)],
                       act_funcs=[tf.nn.leaky_relu for _ in range(layer)],
                       batch_size=BATCH_SIZE,
                       learning_rate=lr,
                       test_samples=test_samples,
                       dropout=False,
                       batch_norm=True)

            fd = 'nn_bn/{}l_{}n_{}lr/{}'.format(layer, node, lr, i)
            losses = train_model(model, train_samples, fd)
            loss_data[fd].append(losses)

        # save the losses to csv
        for fd in loss_data.keys():
            np.savetxt(os.path.join(fd, 'results.csv'),
                       np.asarray(loss_data[fd]), delimiter=',')
