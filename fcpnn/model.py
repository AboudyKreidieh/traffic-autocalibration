import tensorflow as tf
import numpy as np
import os


class Model(object):
    """Abstract model class.

    This class contains generic methods needed by each subsequent model for
    training and testing.
    """

    def __init__(self,
                 inputs,
                 hidden_size,
                 act_funcs,
                 batch_size,
                 learning_rate,
                 test_samples=None,
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
        >>> model = Model(
        >>>     inputs=6,
        >>>     hidden_size=[32, 32],
        >>>     learning_rate=1e-3,
        >>>     batch_size=128,
        >>>     act_funcs=[tf.nn.relu,tf.nn.relu],
        >>>     dropout=False,
        >>>     batch_norm=True
        >>> )
        """
        assert len(hidden_size) == len(act_funcs)

        self.inputs = inputs
        self.size = hidden_size  # + [2]
        self.act_funcs = act_funcs + [None]
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.test_samples = test_samples
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.weights = []
        self.biases = []
        self.col_num = col_num
        self.h = []

        # create a tensorflow session
        self.sess = tf.Session()

        # prediction placeholder
        self.output_ph = tf.placeholder(tf.float32, shape=[None, 1])

        # input placeholder
        self.input_ph = tf.placeholder(tf.float32, shape=[None, inputs])

        # create a placeholder for the actual and expected outputs
        self.pred, self.pred_mean, self.pred_logstd = self._init_model()

        # loss function (used for training)
        self.loss = tf.reduce_mean(tf.square(self.output_ph - self.pred))

        # optimizer
        self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        # mean square error loss (used for testing)
        self.loss_mse = tf.reduce_mean(tf.square(self.output_ph - self.pred))

        # initialize all variable
        self.sess.run(tf.global_variables_initializer())

        # create saver to save model variables
        self.saver = tf.train.Saver()

    def train(self, samples):
        """Perform a single iteration of training on a sample of data.

        Parameters
        ----------
        samples : array_like
            samples to perform training on
        """
        # collect a batch of data
        indices = np.random.randint(low=0,
                                    high=samples.shape[0],
                                    size=self.batch_size)
        obs = samples[indices, :-1]
        actual = samples[indices, -1:]

        # preform a gradient step
        self.sess.run(
            self.opt,
            feed_dict={
                self.output_ph: actual,
                self.input_ph: obs
            }
        )

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
        return self.sess.run(self.pred, feed_dict={self.input_ph: obs})

    def save(self, fd, itr):
        """Save a checkpoint.

        Parameters
        ----------
        fd : str
            path to the checkpoint directory
        itr : int
            training iteration
        """
        save_loc = os.path.join(fd, "{}.ckpt".format(itr))
        self.saver.save(self.sess, save_loc)

    def restore(self, fp):
        """Restore a checkpoint.

        Parameters
        ----------
        fp : str
            path to the checkpoint
        """
        self.saver.restore(self.sess, fp)

    def compute_mse(self):
        """Compute the mean square error loss.

        Returns
        -------
        array_like or None
            mean squared error loss on the test set; None if no test step was
            initially provided
        """
        if self.test_samples is None:
            return None

        return self.sess.run(
            self.loss_mse,
            feed_dict={
                self.input_ph: self.test_samples[:, :-1],
                self.output_ph: self.test_samples[:, -1:]}
        )

    def close(self):
        """Terminate the model.

        This is responsible for clearing the computation graph and closing the
        tensorflow session.
        """
        self.sess.close()
        tf.reset_default_graph()

    def _init_model(self):
        """Initialize the computation graph corresponding to the model.

        Returns
        -------
        tf.Variable
            output layer prediction
        tf.Variable
            output layer mean prediction
        tf.Variable
            output layer log std prediction
        """
        raise NotImplementedError


class NN(Model):
    """A generic fully connected neural network model."""

    def _init_model(self):
        """Create a the neural network for regression problem.

        Returns
        -------
        tf.Variable
            output layer prediction
        tf.Variable
            output layer mean prediction
        tf.Variable
            output layer log std prediction
        """
        scope = "nn"
        if self.dropout:
            scope += "_d"
        if self.batch_norm:
            scope += "bn"

        with tf.variable_scope(scope):
            # create the hidden layers
            last_layer = self.input_ph
            for i, hidden_size in enumerate(self.size):
                last_layer = tf.layers.dense(
                    inputs=last_layer,
                    units=hidden_size,
                    kernel_initializer=tf.truncated_normal_initializer(),
                    activation=self.act_funcs[i])

                if self.batch_norm:
                    last_layer = tf.contrib.layers.batch_norm(
                        last_layer, center=True, scale=True)

            # create the output layer
            last_layer = tf.layers.dense(
                inputs=last_layer,
                units=2,
                activation=None)

        output_mean = last_layer[:, :-1]
        output_logstd = last_layer[:, -1:]
        output = output_mean + tf.exp(output_logstd) * tf.random_normal([1])

        return output, output_mean, output_logstd


# class PNN(Model):
#     """Progressive neural network.
#
#     The last column will be the target task column.
#     Other columns represent source tasks
#     """
#
#     def __init__(self,
#                  inputs,
#                  hidden_size,
#                  activations,
#                  num_source_cols,
#                  dropout=False,
#                  batch_norm=False):
#         """Instantiate the PNN model.
#
#         Parameters
#         ----------
#         inputs : int
#             number of inputs to the model
#         hidden_size : list of int
#             list of units in the hidden layers
#         activations : list of tf.nn.*
#             list of activation functions for each layer. The size of the list
#             should be number of hidden layers
#         num_source_cols: int,
#             Number of source task columns.
#         dropout: boolean, optional
#             specifies whether to use dropout
#         batch_norm: boolean, optional
#             specifies whether to use batch normalization
#         """
#         super().__init__()
#         assert len(hidden_size) == len(activations)
#
#         self.inputs = inputs
#         self.size = hidden_size + [1]  # 1 for number of outputs
#         self.activations = activations
#         self.dropout = dropout
#         self.batch_norm = batch_norm
#         self.weights = []
#         self.biases = []
#         self.num_source_cols = num_source_cols
#         self.dropout = dropout
#         self.batch_norm = batch_norm
#         self.num_layers = len(self.size)
#         self.num_cols = self.num_source_cols + 1  # 1 is the target domain net
#
#         # create a placeholder for the inputs
#         self.input_ph = tf.placeholder(tf.float32, shape=[None, inputs])
#
#         self.last_layer = self.input_ph
#         self.source_weights = [[] for _ in range(self.num_source_cols)]
#         self.source_biases = [[] for _ in range(self.num_source_cols)]
#         self.lateral_conns = []
#         for l in range(self.num_layers):
#             self.lateral_conns.append([[]] * self.num_cols)
#         self.lateral_conns[0] = None
#         self.target_weights = []
#         self.target_biases = []
#
#     def create_progressive_nn(self):
#         """ Creates a the neural network for regression problem
#
#         Returns
#         ----------
#         output_layer: Output layer prediction
#         """
#         scope = "nn"
#         if self.dropout:
#             scope += "_d"
#         if self.batch_norm:
#             scope += "bn"
#
#         # create weights and biases
#         col_objs = [
#             NN(self.size, self.activations, self.input_ph, col)
#             for col in range(self.num_cols)]
#         for col_obj in col_objs:
#             col_obj.create_single_task_nn()
#
#         # create connections and computation graph
#         if self.num_cols == 1:
#             return col_objs[0].output_pred
#
#         last_layer = [self.input_ph for _ in range(self.num_cols)]
#
#         for layer in range(self.num_layers):
#             for col in range(self.num_cols):
#                 last_layer[col] = tf.matmul(last_layer[col],
#                                             col_objs[col].weights[layer]) + \
#                                   col_objs[col].biases[layer]
#                 if col > 0 and layer > 0:
#                     for c in range(col):
#                         # from layer l-1 of all prev columns
#                         u_shape = [col_objs[c].size[layer - 1],
#                                    col_objs[col].size[layer]]
#                         new_u = tf.get_variable(
#                             name='U_{}_{}_{}'.format(layer, col, c),
#                             shape=u_shape,
#                             initializer=tf.contrib.layers.xavier_initializer())
#                         self.latteral_conns[layer][col].append(new_u)
#                         last_layer[col] += tf.matmul(
#                             col_objs[c].h[layer - 1], new_u)
#
#                 if col_objs[col].activations[layer] is not None:
#                     last_layer[col] = col_objs[col].activations[layer](
#                         last_layer[col])
#                 if self.dropout:
#                     last_layer[col] = tf.nn.dropout(last_layer[col], 0.5)
#                 if self.batch_norm:
#                     last_layer[col] = tf.contrib.layers.batch_norm(
#                         last_layer[col],
#                         center=True, scale=True,
#                         scope="{}_{}".format(scope, i),
#                         reuse=tf.AUTO_REUSE)
#                 col_objs[col].h[layer] = last_layer[col]
#         output_pred = last_layer[col]
#
#         return output_pred
#
#
# class FCPNN(Model):
#     """
#
#     """
#     pass
