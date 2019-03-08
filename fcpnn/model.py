import tensorflow as tf
import numpy as np


class Model(object):
    """Abstract model class.

    This class contains generic methods needed by each subsequent model for
    training and testing.
    """

    def train(self, samples):
        """Perform a single iteration of training on a sample of data.

        Parameters
        ----------
        samples : array_like
            samples to perform training on
        """
        raise NotImplementedError

    def run(self, obs):
        """Perform a forward pass of the model.

        Parameters
        ----------
        obs : array_like
            the input observations

        Returns
        -------
        np.ndarray
            the output from the model
        """
        raise NotImplementedError

    def save(self, fd, itr):
        """Save a checkpoint.

        Parameters
        ----------
        fd : str
            path to the checkpoint directory
        itr : int
            training iteration
        """
        raise NotImplementedError

    def restore(self, fp):
        """Restore a checkpoint.

        Parameters
        ----------
        fp : str
            path to the checkpoint
        """
        raise NotImplementedError

    def compute_loss(self, expected, actual):
        """Compute the loss function.

        Parameters
        ----------
        expected : array_like
            expected values
        actual : array_like
            values computed by the neural network model

        Returns
        -------
        array_like
            loss values for each sample
        """
        raise NotImplementedError


class NN(Model):
    """A generic fully connected neural network model."""

    def __init__(self,
                 inputs,
                 hidden_size,
                 act_funcs,
                 batch_size,
                 learning_rate,
                 dropout=False,
                 batch_norm=False,
                 col_num=0):
        """Instantiate the neural network model.

        Parameters
        ----------
        inputs : int
            number of inputs to the model
        hidden_size : list of int
            list of units in the hidden layers
        act_funcs : list of tf.nn.*
            list of activation functions for each layer. The size of the list
            should be number of hidden layers
        learning_rate : float
            optimizer learning rate
        dropout : bool, optional
            specifies whether to use dropout
        batch_norm : bool, optional
            specifies whether to use batch normalization

        Example
        -------
        >>> model = NN(inputs=6, hidden_size=[32, 32],
        >>>            learning_rate=1e-3,
        >>>            act_funcs=[tf.nn.relu,tf.nn.relu],
        >>>            dropout=False, batch_norm=True)
        """
        assert len(hidden_size) == len(act_funcs)

        self.inputs = inputs
        self.size = hidden_size
        self.act_funcs = act_funcs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.weights = []
        self.biases = []
        self.col_num = col_num

        # create a tensorflow session
        self.sess = tf.Session()

        # prediction placeholder
        self.pred_ph = tf.placeholder(tf.float32, shape=[None, 1])

        # input placeholder
        self.input_ph = tf.placeholder(tf.float32, shape=[None, inputs])

        # create a placeholder for the actual and expected outputs
        self.pred, self.pred_mean, self.pred_logstd = \
            self.create_single_task_nn()

        # loss function
        # self.loss = tf.reduce_mean(0.5 * tf.square(self.pred_ph - self.pred))
        self.loss = 0.5 * tf.square(self.pred_ph - self.pred)

        # list of hidden layers
        self.h = None  # [self.input_ph].extend(h)

        # optimizer
        self.opt = self._get_optimizer()

        # initialize all variable
        self.sess.run(tf.global_variables_initializer())

    def create_single_task_nn(self):
        """Create a the neural network for regression problem.

        Returns
        -------
        tf.Variable
            Output layer prediction
        list of tf.Variable
            hidden layers
        """
        scope = "nn"
        if self.dropout:
            scope += "_d"
        if self.batch_norm:
            scope += "_bn"

        with tf.variable_scope(scope):
            # create the hidden layers
            last_layer = self.input_ph
            for i, hidden_size in enumerate(self.size):
                last_layer = tf.layers.dense(
                    inputs=last_layer,
                    units=hidden_size,
                    activation=self.act_funcs[i])

            # create the output layer
            last_layer = tf.layers.dense(
                inputs=last_layer,
                units=2,
                activation=None)

        # # create input layer
        # print([self.input_ph.shape[1], self.size[0]])
        # self.weights.append(tf.get_variable(
        #     name='W{}_{}'.format(0, self.col_num),
        #     shape=[self.input_ph.shape[1], self.size[0]],
        #     initializer=tf.contrib.layers.xavier_initializer()
        # ))
        # self.biases.append(tf.get_variable(
        #     name='b{}_{}'.format(0, self.col_num),
        #     shape=[self.size[0]],
        #     initializer=tf.constant_initializer(0.)
        # ))
        #
        # # create hidden and output layers
        # for layer in range(1, len(self.size)):
        #     shape = self.size[layer - 1:layer + 1]
        #     self.weights.append(tf.get_variable(
        #         name='W{}_{}'.format(layer, self.col_num),
        #         shape=shape,
        #         initializer=tf.contrib.layers.xavier_initializer()
        #     ))
        #     self.biases.append(tf.get_variable(
        #         name='b{}_{}'.format(layer, self.col_num),
        #         shape=shape[1],
        #         initializer=tf.constant_initializer(0.)
        #     ))

        # # create computation graph
        # hidden = []
        # for i, (W, b, act_func) in enumerate(
        #         zip(self.weights, self.biases, self.act_funcs)):
        #     last_layer = tf.matmul(last_layer, W) + b
        #     hidden.append(last_layer)
        #     # add activation function
        #     if act_func is not None:
        #         last_layer = act_func(last_layer)
        #     # add dropout if requested
        #     if self.dropout:
        #         last_layer = tf.nn.dropout(last_layer, 0.5)
        #     # add batch normalization if requested
        #     if self.batch_norm:
        #         last_layer = tf.contrib.layers.batch_norm(
        #             last_layer,
        #             center=True,
        #             scale=True,
        #             scope="{}_{}".format(scope, i),
        #             reuse=tf.AUTO_REUSE)

        # compute the prediction from the mean and standard deviation
        output_mean = last_layer[:, 0]
        output_logstd = last_layer[:, 1]
        pred = output_mean + tf.exp(output_logstd) * tf.random_normal([1])

        return pred, output_mean, output_logstd

    def train(self, samples):
        # collect a batch of data
        indices = np.random.randint(low=0,
                                    high=samples.shape[0],
                                    size=self.batch_size)
        obs = samples[indices, :-1]
        actual = samples[indices, -1:]

        # preform a gradient step
        self.sess.run(self.opt, feed_dict={self.pred_ph: actual,
                                           self.input_ph: obs})

        # compute the new loss
        return self.sess.run(
            self.loss,
            feed_dict={
                self.pred_ph: samples[:, -1:],
                self.input_ph: samples[:, :-1]
            }
        )

    def compute_loss(self, obs, pred):
        return self.sess.run(self.loss, feed_dict={self.input_ph: obs,
                                                   self.pred_ph: pred})

    def run(self, obs):
        return self.sess.run(self.pred, feed_dict={self.input_ph: obs})

    def _get_optimizer(self):
        # get the log likelihood of the specified action
        p = tf.contrib.distributions.MultivariateNormalDiag(
            loc=self.pred_mean,
            scale_diag=tf.exp(self.pred_logstd)
        )
        log_likelihoods = p.log_prob(self.pred_ph)

        # compute the loss and generate the optimizer
        loss = tf.reduce_mean(tf.multiply(log_likelihoods, self.loss))

        return tf.train.AdamOptimizer(self.learning_rate).minimize(loss)


class PNN(Model):
    """Progressive neural network.

    The last column will be the target task column.
    Other columns represent source tasks
    """

    def __init__(self,
                 inputs,
                 hidden_size,
                 activations,
                 num_source_cols,
                 dropout=False,
                 batch_norm=False):
        """Instantiate the PNN model.

        Parameters
        ----------
        inputs : int
            number of inputs to the model
        hidden_size : list of int
            list of units in the hidden layers
        activations : list of tf.nn.*
            list of activation functions for each layer. The size of the list
            should be number of hidden layers
        num_source_cols: int,
            Number of source task columns.
        dropout: boolean, optional
            specifies whether to use dropout
        batch_norm: boolean, optional
            specifies whether to use batch normalization
        """
        super().__init__()
        assert len(hidden_size) == len(activations)

        self.inputs = inputs
        self.size = hidden_size + [1]  # 1 for number of outputs
        self.activations = activations
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.weights = []
        self.biases = []
        self.num_source_cols = num_source_cols
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.num_layers = len(self.size)
        self.num_cols = self.num_source_cols + 1  # 1 is the target domain net

        # create a placeholder for the inputs
        self.input_ph = tf.placeholder(tf.float32, shape=[None, inputs])

        self.last_layer = self.input_ph
        self.source_weights = [[] for _ in range(self.num_source_cols)]
        self.source_biases = [[] for _ in range(self.num_source_cols)]
        self.lateral_conns = []
        for l in range(self.num_layers):
            self.lateral_conns.append([[]] * self.num_cols)
        self.lateral_conns[0] = None
        self.target_weights = []
        self.target_biases = []

    def create_progressive_nn(self):
        """ Creates a the neural network for regression problem

        Returns
        ----------
        output_layer: Output layer prediction
        """
        scope = "nn"
        if self.dropout:
            scope += "_d"
        if self.batch_norm:
            scope += "bn"

        # create weights and biases
        col_objs = [
            NN(self.size, self.activations, self.input_ph, col)
            for col in range(self.num_cols)]
        for col_obj in col_objs:
            col_obj.create_single_task_nn()

        # create connections and computation graph
        if self.num_cols == 1:
            return col_objs[0].output_pred

        last_layer = [self.input_ph for _ in range(self.num_cols)]

        for layer in range(self.num_layers):
            for col in range(self.num_cols):
                last_layer[col] = tf.matmul(last_layer[col],
                                            col_objs[col].weights[layer]) + \
                                  col_objs[col].biases[layer]
                if col > 0 and layer > 0:
                    for c in range(col):
                        # from layer l-1 of all prev columns
                        u_shape = [col_objs[c].size[layer - 1],
                                   col_objs[col].size[layer]]
                        new_u = tf.get_variable(
                            name='U_{}_{}_{}'.format(layer, col, c),
                            shape=u_shape,
                            initializer=tf.contrib.layers.xavier_initializer())
                        self.latteral_conns[layer][col].append(new_u)
                        last_layer[col] += tf.matmul(
                            col_objs[c].h[layer - 1], new_u)

                if col_objs[col].activations[layer] is not None:
                    last_layer[col] = col_objs[col].activations[layer](
                        last_layer[col])
                if self.dropout:
                    last_layer[col] = tf.nn.dropout(last_layer[col], 0.5)
                if self.batch_norm:
                    last_layer[col] = tf.contrib.layers.batch_norm(
                        last_layer[col],
                        center=True, scale=True,
                        scope="{}_{}".format(scope, i),
                        reuse=tf.AUTO_REUSE)
                col_objs[col].h[layer] = last_layer[col]
        output_pred = last_layer[col]

        return output_pred


class FCPNN(Model):
    """

    """
    pass
