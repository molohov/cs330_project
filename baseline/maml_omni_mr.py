import numpy as np
import pickle
import random
import functools
import os
from absl import app
from absl import flags

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import MaxPooling2D
from tensorflow_probability.python.layers import util as tfp_layers_util
from maml_omni_mr_2 import MAML
#import time
#import ipdb

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('data_dir', None,
                    'Directory of data files.')
flags.DEFINE_list('data',['train_data_omni.pkl', 'val_data_omni.pkl'],'data')
flags.DEFINE_integer('num_classes', 20, 
                     'number of classes used in classification')
flags.DEFINE_integer('update_batch_size', 1, 
                     'number of examples used for inner gradient update')
flags.DEFINE_integer('metatrain_iterations', 60000, \
                     'number of metatraining iterations.')
flags.DEFINE_integer('meta_batch_size', 16, \
                     'number of tasks sampled per meta-update')

## Training options
flags.DEFINE_float('beta', 1e-5, 'beta for MR')
flags.DEFINE_integer('dim_w', 196, 'dimension of w')
flags.DEFINE_integer('dim_im', 28, 'dimension of image')

flags.DEFINE_integer('test_num_updates', 20, \
                     'number of inner gradient updates during test.')
flags.DEFINE_integer('num_updates', 5, \
                     'number of inner gradient updates during training.')
flags.DEFINE_float('update_lr', 0.01, \
                   'step size alpha for inner gradient update.') 
flags.DEFINE_float('meta_lr', 0.005, 'the base learning rate of the generator')
flags.DEFINE_float('var', -3.0, 'var initial')

## Model options
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets \
                     -- 32 for miniimagenet, 64 for omiglot.')
flags.DEFINE_bool('max_pool', False, \
                  'Whether use max pooling rather than strided convolutions')
flags.DEFINE_bool('stop_grad', False,
                  'if True, do not use second derivatives (for speed)')

## Logging, saving, and testing options
flags.DEFINE_string('logdir', './summary_omni', \
                    'directory for summaries and checkpoints.')
flags.DEFINE_integer('trial', 1, 'trial_num')

directory = os.getcwd()+'/eval_output/'
if not os.path.exists(directory):
    os.makedirs(directory)

def train(model, sess, checkpoint_dir):
    print('Done initializing, start training.')
    SUMMARY_INTERVAL = 5
    PRINT_INTERVAL = 5
    EXPERIMENT = 'omniglot'+str(FLAGS.num_classes)+'way'+ \
        str(FLAGS.update_batch_size)+'shot'#+ '_' + str(random.randint(1,101))

    summary_writer = tf.summary.FileWriter(checkpoint_dir, sess.graph)
    prelosses, postlosses = [], []
    prelosses_val, postlosses_val = [], []
    iter_r = []; pre_train_r = []; post_train_r = []; 
    pre_val_r = []; post_val_r = []
    
    for itr in range( FLAGS.metatrain_iterations):
        input_tensors = [model.metatrain_op]

        if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
            input_tensors.extend([model.total_accuracies2[0], \
                                  model.total_accuracies2[-1]])
            input_tensors_val = [model.metaval_total_accuracies2[0], \
                                 model.metaval_total_accuracies2[-1]]
        
        result = sess.run(input_tensors, feed_dict={})

        if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
            summary, result_val = \
                sess.run([model.summ_op, input_tensors_val], feed_dict={})
            summary_writer.add_summary(summary, itr)
            prelosses.append(result[-2])
            postlosses.append(result[-1])
            prelosses_val.append(result_val[-2]) 
            postlosses_val.append(result_val[-1]) 

        if (itr!=0) and itr % PRINT_INTERVAL == 0:  
            print('###############################')
            print(' Iteration ' + str(itr) + ':')
            print('Training: ', 'pre -->', np.mean(prelosses), \
                  'post-->', np.mean(postlosses))            
            print('Validation: ', 'pre-->', np.mean(prelosses_val), \
                  'post-->', np.mean(postlosses_val))
            print('###############################')
            
            iter_r.append(itr)
            pre_train_r.append(np.mean(prelosses)); 
            post_train_r.append(np.mean(postlosses))
            pre_val_r.append(np.mean(prelosses_val)); 
            post_val_r.append(np.mean(postlosses_val))
            all_ = [iter_r, pre_train_r,post_train_r, pre_val_r, post_val_r]
            pickle.dump(all_, open(directory+EXPERIMENT, 'wb'))
            
            prelosses, postlosses = [], []
            prelosses_val, postlosses_val = [], []
        


def get_batch(x, y):
    xs, ys, xq, yq = [], [], [], []
    for i in range(FLAGS.meta_batch_size):
        support_set = []
        query_set = []
        support_sety = []
        query_sety = []
        task_id = np.random.choice(range(np.shape(x)[0]), \
                                   size=np.shape(x)[1], replace=True)
        for k in range(np.shape(x)[1]):
          idx = np.random.choice(range(np.shape(x)[2]), \
                                 size=FLAGS.update_batch_size+\
                                 FLAGS.update_batch_size, replace=False)
          x_k = x[task_id[k], k, idx, :]
          y_k = y[task_id[k], k, idx, :]

          support_set.append(x_k[:FLAGS.update_batch_size])
          query_set.append(x_k[FLAGS.update_batch_size:])
          support_sety.append(y_k[:FLAGS.update_batch_size])
          query_sety.append(y_k[FLAGS.update_batch_size:])

        xs_k = np.concatenate(support_set, 0)
        xq_k = np.concatenate(query_set, 0)
        ys_k = np.concatenate(support_sety, 0)
        yq_k = np.concatenate(query_sety, 0)

        xs.append(xs_k)
        xq.append(xq_k)
        ys.append(ys_k)
        yq.append(yq_k)

    xs, ys = np.stack(xs, 0), np.stack(ys, 0)
    xq, yq = np.stack(xq, 0), np.stack(yq, 0)
    return xs, ys, xq, yq

def gen(x, y):
    while True:
      yield get_batch(np.array(x), np.array(y))


def main(_):
    dim_output = FLAGS.num_classes
    dim_input = FLAGS.dim_im * FLAGS.dim_im * 1

    exp_name = '%s.beta-%g.meta_lr-%g.update_lr-%g.trial-%d' % (
      'maml_mr_omni', FLAGS.beta, FLAGS.meta_lr, FLAGS.update_lr, FLAGS.trial)
    checkpoint_dir = os.path.join(FLAGS.logdir, exp_name)

    x_train, y_train = pickle.load(open(os.getcwd()+'/'+FLAGS.data_dir+FLAGS.data[0],'rb'))
    x_val, y_val = pickle.load(open(os.getcwd()+'/'+FLAGS.data_dir+FLAGS.data[1],'rb'))


    #n_task * n_class_per_task * n_image_per_class(20) *  img_size(784)
    x_train = np.reshape(x_train,[x_train.shape[0], x_train.shape[1], \
                        x_train.shape[2], -1]); y_train =  np.array(y_train) 
    x_val = np.reshape(x_val,[x_val.shape[0], x_val.shape[1], \
                              x_val.shape[2], -1]); y_val = np.array(y_val)
    
    
    ds_train = tf.data.Dataset.from_generator(
     functools.partial(gen, x_train, y_train), \
     (tf.float32, tf.float32, tf.float32, tf.float32), \
     (tf.TensorShape(
             [None, FLAGS.update_batch_size*FLAGS.num_classes,dim_input]), \
      tf.TensorShape(
              [None, FLAGS.update_batch_size*FLAGS.num_classes, dim_output]), \
      tf.TensorShape(
              [None, FLAGS.update_batch_size*FLAGS.num_classes, dim_input]), \
      tf.TensorShape(
              [None, FLAGS.update_batch_size*FLAGS.num_classes, dim_output])))

    ds_val = tf.data.Dataset.from_generator(
     functools.partial(gen, x_val, y_val), \
     (tf.float32, tf.float32, tf.float32, tf.float32), \
     (tf.TensorShape(
             [None, FLAGS.update_batch_size*FLAGS.num_classes,dim_input]), \
     tf.TensorShape(
             [None, FLAGS.update_batch_size*FLAGS.num_classes, dim_output]), \
     tf.TensorShape(
             [None, FLAGS.update_batch_size*FLAGS.num_classes, dim_input]), \
     tf.TensorShape(
             [None, FLAGS.update_batch_size*FLAGS.num_classes, dim_output])))
    
    
    kernel_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(\
        untransformed_scale_initializer = \
        tf.compat.v1.initializers.random_normal(mean=FLAGS.var, stddev=0.1))
    encoder_w = tf.keras.Sequential([
                     tfp.layers.Convolution2DReparameterization(\
                                filters=32, kernel_size=3, strides=(2, 2), \
                                activation='relu', padding='SAME', \
                                kernel_posterior_fn = kernel_posterior_fn),
                     tfp.layers.Convolution2DReparameterization(\
                                filters=48, kernel_size=3, strides=(2, 2), \
                                activation='relu', padding='SAME', \
                                kernel_posterior_fn = kernel_posterior_fn),
                     MaxPooling2D(pool_size=(2, 2)),
                     tfp.layers.Convolution2DReparameterization(
                                filters=64, kernel_size=3, strides=(2, 2), \
                                activation='relu', padding='SAME', \
                                kernel_posterior_fn = kernel_posterior_fn),
                     tf.keras.layers.Flatten(),
                     tfp.layers.DenseReparameterization(
                        FLAGS.dim_w,kernel_posterior_fn = kernel_posterior_fn),
                     ])
    
    xa, labela, xb, labelb = ds_train.make_one_shot_iterator().get_next()
    xa = tf.reshape(xa, [-1, 28, 28, 1])
    xb = tf.reshape(xb, [-1, 28, 28, 1])
    with tf.variable_scope("encoder"):
      inputa = encoder_w(xa)

    inputa = tf.reshape(inputa,[-1, FLAGS.update_batch_size*\
                                FLAGS.num_classes, FLAGS.dim_w])
    inputb  = encoder_w(xb)
    inputb = tf.reshape(inputb,[-1, FLAGS.update_batch_size*\
                                FLAGS.num_classes, FLAGS.dim_w])
    input_tensors = {'inputa': inputa,\
                     'inputb': inputb, \
                     'labela': labela, 'labelb': labelb}

    xa_val, labela_val, xb_val, labelb_val = \
                ds_val.make_one_shot_iterator().get_next()
    xa_val = tf.reshape(xa_val, [-1, 28, 28, 1])
    xb_val = tf.reshape(xb_val, [-1, 28, 28, 1])
    inputa_val = encoder_w(xa_val)
    inputa_val = tf.reshape(inputa_val,\
                [-1, FLAGS.update_batch_size*FLAGS.num_classes, FLAGS.dim_w])
    inputb_val = encoder_w(xb_val)
    inputb_val = tf.reshape(inputb_val,\
                [-1, FLAGS.update_batch_size*FLAGS.num_classes, FLAGS.dim_w])
    metaval_input_tensors = {'inputa': inputa_val,\
                             'inputb': inputb_val, \
                             'labela': labela_val, 'labelb': labelb_val}
    
    model = MAML(encoder_w, FLAGS.dim_w, dim_output)

    model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
    model.construct_model(input_tensors= \
        metaval_input_tensors, prefix='metaval_', \
        test_num_updates=FLAGS.test_num_updates)

    model.summ_op = tf.summary.merge_all()

    sess = tf.InteractiveSession()

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()


    train(model, sess,  checkpoint_dir)

if __name__ == "__main__":
    app.run(main)
