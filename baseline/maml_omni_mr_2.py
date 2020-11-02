""" Code for the MAML algorithm and network definitions. """
from __future__ import print_function

import numpy as np
import tensorflow as tf
from absl import flags
from tensorflow.contrib.layers.python import layers as tf_layers


FLAGS = flags.FLAGS

def conv_block(x, weight, bias, reuse, scope):
    x = tf.nn.conv2d(x, weight, [1, 1, 1, 1], 'SAME') + bias
    x = tf_layers.batch_norm(
            x, activation_fn=tf.nn.relu, reuse=reuse, scope=scope)
    if FLAGS.max_pool:
        x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
    return x

## Loss functions
def xent(pred, label):
    return tf.nn.softmax_cross_entropy_with_logits(
            logits=pred, labels=label)/ FLAGS.update_batch_size

class MAML:
    def __init__(self, encoder_w, dim_input=1, dim_output=1):
        """ must call construct_model() after initializing MAML! """
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.channels = 1
        self.img_size = int(np.sqrt(self.dim_input/self.channels))

        self.update_lr = FLAGS.update_lr
        self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr, ())

        self.loss_func = xent
        self.encoder_w = encoder_w
        self.beta = tf.placeholder_with_default(FLAGS.beta, ())

        self.classification = True
        self.dim_hidden = FLAGS.num_filters
        self.forward = self.forward_conv
        self.construct_weights = self.construct_conv_weights



    def construct_model(self, input_tensors=None, prefix='metatrain_',\
                        test_num_updates=0):

        self.inputa = input_tensors['inputa']

        self.inputb = input_tensors['inputb']

        self.labela = input_tensors['labela']
        self.labelb = input_tensors['labelb']

        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):
                training_scope.reuse_variables()
                weights = self.weights
            else:
                self.weights = weights = self.construct_weights()

            outputas, lossesb, outputbs = [], [], []
            accuraciesa, accuraciesb = [], []

            num_updates = max(test_num_updates, FLAGS.num_updates)


            outputbs = [[]]*num_updates
            lossesb = [[]]*num_updates
            xentsb = [[]]*num_updates
            accuraciesb = [[]]*num_updates

            def task_metalearn(inp, reuse=True):
                """ Perform gradient descent for one task in the meta-batch. """
                TRAIN = 'train' in prefix
                inputa, inputb,  labela, labelb = inp
                task_outputbs = []; task_entsb = []; task_accuraciesb = []
                task_lossesb = []

                task_outputa = self.forward(inputa, weights, reuse=reuse)

                task_enta = self.loss_func(task_outputa, labela)
                task_accuracya = tf.contrib.metrics.accuracy(
                        tf.argmax(tf.nn.softmax(task_outputa), 1), \
                        tf.argmax(labela, 1))

                task_kl_loss = sum(self.encoder_w.losses)

                #INNER LOOP (no change with ib)
                grads = tf.gradients(task_enta, list(weights.values()))
                if FLAGS.stop_grad:
                    grads = [tf.stop_gradient(grad) for grad in grads]
                gradients = dict(zip(weights.keys(), grads))
                fast_weights = dict(zip(weights.keys(), 
                            [weights[key] - self.update_lr*gradients[key] \
                            for key in weights.keys()]))

                output = self.forward(inputb, weights, reuse=True)
                task_outputbs.append(output)
                task_entsb.append(self.loss_func(output, labelb))

                
                task_accuraciesb.append(
                    tf.contrib.metrics.accuracy(tf.argmax(
                            tf.nn.softmax(output),1), tf.argmax(labelb, 1)))

                task_lossesb.append(self.loss_func(output, labelb)+ \
                                    self.beta * task_kl_loss)

                def while_body(fast_weights_values):
                    loss = self.loss_func(self.forward(inputa, \
                            dict(zip(fast_weights.keys(),fast_weights_values)),
                            reuse=True), labela)
                    grads = tf.gradients(loss, fast_weights_values)
                    fast_weights_values =  [v - self.update_lr*g \
                            for v, g in zip(fast_weights_values, grads)]
                    return fast_weights_values
                fast_weights_values = tf.while_loop(lambda _: True, \
                    while_body, loop_vars = [fast_weights.values()],
                    maximum_iterations = num_updates - 1, back_prop = TRAIN
                                            )
                fast_weights = dict(zip(fast_weights.keys(),
                                        fast_weights_values))


                output = self.forward(inputb, fast_weights, reuse=True)
                task_outputbs.append(output)

                task_entsb.append(self.loss_func(output, labelb))
                task_accuraciesb.append(tf.contrib.metrics.accuracy(
                    tf.argmax(tf.nn.softmax(task_outputbs[-1]), 1), 
                    tf.argmax(labelb, 1)))
                task_lossesb.append(self.loss_func(
                        output, labelb)+ self.beta * task_kl_loss)

                task_output = [task_outputa, task_outputbs, task_enta, 
                               task_entsb, task_lossesb, task_accuracya, 
                               task_accuraciesb]
                return task_output

            if FLAGS.norm != 'None':
                _ = task_metalearn((self.inputa[0], self.inputb[0], 
                                    self.labela[0], self.labelb[0]), False)
            out_dtype = [tf.float32, [tf.float32]*2, tf.float32,  
                         [tf.float32]*2, [tf.float32]*2, tf.float32, 
                         [tf.float32]*2]

            result = tf.map_fn(
                    task_metalearn, elems=(self.inputa, self.inputb,  \
                             self.labela, self.labelb), dtype=out_dtype)

            outputas, outputbs, xenta, xentsb, lossesb, \
                                            accuraciesa, accuraciesb = result
        ## Performance & Optimization
        if 'train' in prefix:
            self.total_loss1 = \
                tf.reduce_sum(xenta) / tf.to_float(FLAGS.meta_batch_size)
            self.total_losses2 = total_losses2 = \
                [tf.reduce_sum(xentsb[j]) / tf.to_float(FLAGS.meta_batch_size)\
                for j in range(len(xentsb))]
            self.total_losses3 = total_losses3 = \
              [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size)\
               for j in range(len(lossesb))]

            self.total_accuracy1 = \
                tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
            self.total_accuracies2 = total_accuracies2 = \
            [tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size)\
            for j in range(len(accuraciesb))]
            if FLAGS.metatrain_iterations > 0:
                optimizer = tf.train.AdamOptimizer(self.meta_lr)
                THETA = tf.get_collection(
                        tf.GraphKeys.GLOBAL_VARIABLES, scope='model')
                PHI = tf.get_collection(
                        tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')
                self.gvs_theta = gvs_theta = \
                    optimizer.compute_gradients(self.total_losses2[-1], THETA)
                metatrain_theta_op = optimizer.apply_gradients(gvs_theta)

                self.gvs_phi = gvs_phi = optimizer.compute_gradients(
                        self.total_losses3[-1], PHI)
                metatrain_phi_op = optimizer.apply_gradients(gvs_phi)

                with tf.control_dependencies(
                        [metatrain_theta_op, metatrain_phi_op]):
                    self.metatrain_op = tf.no_op()
                scale_v = [v  for v in self.encoder_w.trainable_variables 
                           if 'scale' in v.name]
                scale_norm = [tf.reduce_mean(v) for v in scale_v]
                scale_norm = tf.reduce_mean(scale_norm)

                tf.summary.scalar(prefix+'full_loss', total_losses3[-1])
                tf.summary.scalar(prefix+'regularizer',
                                  total_losses3[-1] - total_losses2[-1])
                tf.summary.scalar(prefix+'untransformed_scale', scale_norm)

        else:
            self.metaval_total_loss1 = \
                tf.reduce_sum(xenta) / tf.to_float(FLAGS.meta_batch_size)
            self.metaval_total_losses2 = total_losses2 = \
            [tf.reduce_sum(xentsb[j]) / tf.to_float(FLAGS.meta_batch_size) \
             for j in range(len(xentsb))]
            self.metaval_total_accuracy1 = \
            tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
            self.metaval_total_accuracies2 = total_accuracies2 =\
            [tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size)\
             for j in range(len(accuraciesb))]

        ## Summaries
        tf.summary.scalar(prefix+'Pre-update accuracy', total_accuracies2[0])
        tf.summary.scalar(prefix+'Post-update accuracy'+\
                          str(num_updates), total_accuracies2[-1])


    def construct_conv_weights(self):
        weights = {}
        dtype = tf.float32
        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(
                dtype=dtype)
        k = 3
        weights['conv1'] = tf.get_variable(
                'conv1', [k, k, self.channels, self.dim_hidden], 
                initializer=conv_initializer, dtype=dtype)
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv2'] = tf.get_variable(
                'conv2', [k, k, self.dim_hidden, self.dim_hidden], 
                initializer=conv_initializer, dtype=dtype)
        weights['b2'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv3'] = tf.get_variable(
                'conv3', [k, k, self.dim_hidden, self.dim_hidden], 
                initializer=conv_initializer, dtype=dtype)
        weights['b3'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv4'] = tf.get_variable(
                'conv4', [k, k, self.dim_hidden, self.dim_hidden], 
                initializer=conv_initializer, dtype=dtype)
        weights['b4'] = tf.Variable(tf.zeros([self.dim_hidden]))

        weights['w5'] = tf.Variable(
            tf.random_normal([self.dim_hidden, self.dim_output]), name='w5')
        weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')
        return weights

    def forward_conv(self, inp, weights, reuse=False, scope=''):
        channels = self.channels
        inp = tf.reshape(inp, [-1, self.img_size, self.img_size, channels])

        hidden1 = conv_block(
                inp, weights['conv1'], weights['b1'], reuse, scope+'0')
        hidden2 = conv_block(
                hidden1, weights['conv2'], weights['b2'], reuse, scope+'1')
        hidden3 = conv_block(
                hidden2, weights['conv3'], weights['b3'], reuse, scope+'2')
        hidden4 = conv_block(
                hidden3, weights['conv4'], weights['b4'], reuse, scope+'3')
        
        hidden4 = tf.reduce_mean(hidden4, [1, 2])

        return tf.matmul(hidden4, weights['w5']) + weights['b5']

