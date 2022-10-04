import logging
import os
import pickle
import time
import itertools

import tensorflow as tf
from tensorflow.contrib.losses.python.metric_learning import triplet_semihard_loss

import numpy as np

from .util import AttrDict
from .util import tf_config

from . import util
# from . import spherical
from . import sphericalcof
from . import tfnp_compatibility as tfnp
from . import params
from .layers import *           # !!!
from . import datasets          # !!!

logger = logging.getLogger('logger')

# may or may not include the dataset
# this is cumbersome, should probably have made a 'Model' class
def get_model(model, args, **kwargs):
    net = globals()[model](args, **kwargs)
  #  net = coe_branch(args, **kwargs)
  #  net = combineCNNCOF(args, **kwargs)
    net = add_loss(net, args)

    return net
    
def dup(x):
    """ Return two references for input; useful when creating NNs and storing references to layers """
    return [x, x]


def init_block(args, dset=None):
    net = {}
    if dset is None:
        net['input'], curr = dup(tf.placeholder(args.dtype,
                                                shape=(None, *get_indim(args)[1:])))
        net['label'] = tf.placeholder('int64', shape=[None])
    else:
        # dset is tuple (iterator, init_ops); iterator returns input and label
        net['input'], net['label'] = dset[0].get_next()
        curr = net['input']

    net['training'] = tf.placeholder('bool', shape=(), name='is_training')

    return net, curr


# def init_sphcnn(args):
#     method = args.transform_method
#     real = args.real_inputs
#     if method == 'naive':
#         fun = lambda *args, **kwargs: spherical.sph_harm_all(*args, **kwargs, real=real)
#
#     with tf.name_scope('harmonics_or_legendre'):
#         res = args.input_res
#         harmonics = [fun(res // (2**i), as_tfvar=True) for i in range(sum(args.pool_layers) + 1)]
#
#     return harmonics

def init_sphcnncof(args):
    method = args.transform_method
    real = args.real_inputs
    order=args.order
    if method == 'naive':
        fun = lambda *args, **kwargs: sphericalcof.sph_harm_all(*args, **kwargs, real=real)

    with tf.name_scope('harmonics_or_legendre'):
        res = args.input_res//2
        harmonics =  [fun(res // (2**i), as_tfvar=True) for i in range(sum(args.pool_layers) + 1)]

    return harmonics


def conv3d( name, x, num_filters, filter_size, stride, training, padding='SAME', relu_alpha=0.1):
    with tf.variable_scope(name) as scope:
        x = tf.layers.conv3d(x, num_filters, filter_size, stride, padding)
        x = tf.layers.batch_normalization(x, training=training)
        return tf.maximum(x, relu_alpha * x)  # leaky relu


def two_branch(args, convfun=sphconv, **kwargs):
    """ Model, that splits input in two branches, and concatenate intermediate feature maps. """
    method = args.transform_method
    l_or_h = init_sphcnncof(args)
    net, curr = init_block(args, **kwargs)

    assert tfnp.shape(curr)[-1] ==30

    curr = [curr[..., 0][..., np.newaxis],
    curr[..., 1][..., np.newaxis],
    curr[..., 2][..., np.newaxis],
    curr[..., 3][..., np.newaxis],
    curr[..., 4][..., np.newaxis],
    curr[..., 5][..., np.newaxis],
    curr[..., 6][..., np.newaxis],
            curr[..., 7][..., np.newaxis],
            curr[..., 8][..., np.newaxis],
            curr[..., 9][..., np.newaxis],
            curr[..., 10][..., np.newaxis],
            curr[..., 11][..., np.newaxis],
            curr[..., 12][..., np.newaxis],
            curr[..., 13][..., np.newaxis],
            curr[..., 14][..., np.newaxis],
            curr[..., 15][..., np.newaxis],
            curr[..., 16][..., np.newaxis],
            curr[..., 17][..., np.newaxis],
            curr[..., 18][..., np.newaxis],
            curr[..., 19][..., np.newaxis],
            curr[..., 20][..., np.newaxis],
            curr[..., 21][..., np.newaxis],
            curr[..., 22][..., np.newaxis],
            curr[..., 23][..., np.newaxis],
            curr[..., 24][..., np.newaxis],
            curr[..., 25][..., np.newaxis],
            curr[..., 26][..., np.newaxis],
            curr[..., 27][..., np.newaxis],
            curr[..., 28][..., np.newaxis],
            curr[..., 29][..., np.newaxis],
            ]

    
    # indices for legendre or harmonics
    high = 0
    low = 1
    for i, (nf, pool) in enumerate(zip(args.nfilters, args.pool_layers)):

         out = conv3d('conv0', tf.expand_dims(tf.concat(curr, axis=-1), axis=-1), 64, 5, 2, net['training'])
             # num_filters, filter_size, stride,
         out = conv3d('conv1', out, 64, 3, 2, net['training'])
         out = conv3d('conv2', out, 128, 2, 2, net['training'])
         # out = conv3d('conv3', out, 128, 2, 2, net['training'])


            #			self.conv3d('conv2', 32, 3, 1)
            #			self.max_pool3d('max_pool')

            # args.input_res = tfnp.shape(out)[-3]
            # l_or_h = init_sphcnncof(args)
     # else:
     #    for b in range(tfnp.shape(out)[-2]):  # 0,1,2,3,4 ,5,6,7,8,9,10,11,12,13,14 ,15,16,17,18,19
     #        # curr[b] = tf.expand_dims(curr[b],axis=-1)
     #        curr[b]= out[:,:,:,b]
     #
     #        name = 'conv{}_b{}'.format(i, b)
         #   name = 'conv{}'.format(i)
           # if concat and b == [0,2,4,6,8]:  # 0,2,4,6,8,10,12,14,16,18
                # top branch also receives features from bottom branch
              #  curr[b] = tf.concat([ curr[b], curr[b+1]], axis=-1)

            # if i == 1:
            #     curr[b] = sphericalcof.sph_harm_transform_batch(curr[b],method, args.order, l_or_h[high],
            #                                                                m0_only=False)

        #     if not pool:
        #         if not i == 0:
        #          with tf.variable_scope(name):  # , reuse=tf.AUTO_REUSE
        #          #   print(curr[b])
        #             net[name], curr[b] = dup(block(args, convfun, net['training'], curr[b], nf,
        #                                            n_filter_params=args.n_filter_params,
        #                                            harmonics_or_legendre=l_or_h[high],
        #                                            method=method))
        #     else:
        #         with tf.variable_scope(name):  # , reuse=tf.AUTO_REUSE
        #             # force spectral pool in first layer if spectral input
        #         #    print(curr[b])
        #             spectral_pool = True if (args.spectral_input and i == 0) else args.spectral_pool
        #             net[name], curr[b] = dup(block(args, convfun, net['training'], curr[b], nf,
        #                                            n_filter_params=args.n_filter_params,
        #                                            harmonics_or_legendre=l_or_h[high],
        #                                            method=method,
        #                                            spectral_pool=pool if spectral_pool else 0,
        #                                            harmonics_or_legendre_low=l_or_h[low]))
        #             if not spectral_pool:
        #                 # weighted avg pooling
        #                 if args.pool == 'wap':
        #                     curr[b] = area_weights(tf.layers.average_pooling2d(area_weights(curr[b]),
        #                                                                        2*pool, 2*pool,
        #                                                                        'same'),
        #                                            invert=True)
        #                 elif args.pool == 'avg':
        #                     curr[b] = tf.layers.average_pooling2d(curr[b],
        #                                                           2*pool, 2*pool,
        #                                                           'same')
        #                 elif args.pool == 'max':
        #                     curr[b] = tf.layers.max_pooling2d(curr[b],
        #                                                       2*pool, 2*pool,
        #                                                       'same')
        #                 else:
        #                     raise ValueError('args.pool')
        #
        # if pool:
        #     high += 1
        #     low += 1
    # print(np.shape(curr))
    # combine for final layer
    # curr = tf.concat(curr[0:tfnp.shape(out)[-2]], axis=-1)

    return sphcnn_afterconv(out, net, args, l_or_h[high])


def sphcnn_afterconv(curr, net, args, l_or_h):
    """ Part of model after convolutional layers;
    should be common for different architectures. """
    # normalize by area before computing the mean
    # with tf.name_scope('wsa'):
    #     if args.weighted_sph_avg:
    #         n = tfnp.shape(curr)[1]
    #         phi, theta = util.sph_sample(n)
    #         phi += np.diff(phi)[0]/2
    #         curr *= np.sin(phi)[np.newaxis, np.newaxis, :, np.newaxis]

    net['final_conv'] = curr

    if 'complex' in args.model:
        curr = tf.abs(curr)
        nlin = 'relu'
    else:
        nlin = args.nonlin

    # curr is last conv layer
    with tf.name_scope('final_pool'):

        # net['gap']= tf.contrib.layers.flatten(tf.reduce_sum(tf.square(curr),
        #                                                           axis=(2, 4)))
        net['gap'] =  tf.contrib.layers.flatten(tf.abs(curr))  #tf.abs    tf.square
            # net['gap'] = tf.reduce_mean(curr, axis=(2, 3))
        if args.final_pool in ['max', 'all']:
            net['max'] = tf.reduce_max(curr, axis=(1, 2))
        if args.final_pool in ['magnitudes', 'all']:
            net['final_coeffs'] = sphericalcof.sph_harm_transform_batch(curr,
                                                                method=args.transform_method,
                                                                harmonics=l_or_h,
                                                                m0_only=False)
            # use per frequency magnitudes
            # net['magnitudes'] = tf.contrib.layers.flatten(tf.reduce_sum(tf.square(net['final_coeffs']),
              #                                                          axis=(1, 3)))
                                                                        
                     # use per frequency magnitudes
            net['magnitudes'] = tf.contrib.layers.flatten(tf.reduce_sum(tf.square(net['final_coeffs']),axis=(1, 3)))
                                                              
            net['magnitudes'] = tf.real(net['magnitudes'])
        if args.final_pool != 'all':
            curr = net[args.final_pool]
        else:
            curr = tf.concat([net['gap'], net['max'], net['magnitudes']], axis=-1)


    # curr=tf.contrib.layers.flatten(curr)
    if args.dropout:
        curr = tf.nn.dropout(curr,
                             keep_prob=tf.cond(net['training'],
                                               lambda: 0.5,
                                               lambda: 1.0))

    if not args.no_final_fc:
        with tf.variable_scope('fc1') as scope:
            net['fc1'], curr = dup(block(AttrDict({**args.__dict__,
                                                   'batch_norm': False,
                                                   'nonlin': nlin}),
                                         tf.layers.dense, net['training'], curr, 1024))
            if args.dropout:
                curr = tf.nn.dropout(curr,
                                     keep_prob=tf.cond(net['training'],
                                                       lambda: 0.5,
                                                       lambda: 1.0))
                for v in scope.trainable_variables():
                    tf.summary.histogram(v.name, v)

    net['descriptor'] = curr

    if args.triplet_loss:
        norm_desc = tf.nn.l2_normalize(curr, dim=-1)
        # this only works w/ fixed batch size
        triplet_loss = triplet_semihard_loss(tf.cast(tf.reshape(net['label'],
                                                                (args.train_bsize,)),
                                                     'int32'),
                                             norm_desc)
        # NaNs may appear if bsize is small:
        triplet_loss = tf.where(tf.is_nan(triplet_loss),
                                tf.zeros_like(triplet_loss),
                                triplet_loss)
        tf.add_to_collection(tf.GraphKeys.LOSSES, triplet_loss)
        net['triplet_loss'] = triplet_loss
    else:
        net['triplet_loss'] = 0

    with tf.variable_scope('out') as scope:
        if args.extra_loss:
            nch = tfnp.shape(curr)[-1]
            net['out'] = tf.layers.dense(curr, args.n_classes)
            second_branch_out = tf.layers.dense(curr[..., nch//2:], args.n_classes)
            tf.losses.softmax_cross_entropy(tf.one_hot(net['label'], args.n_classes),
                                            second_branch_out)
        else:
            net['out'], curr = dup(tf.layers.dense(curr, args.n_classes))
        for v in scope.trainable_variables():
            tf.summary.histogram(v.name, v)

    return net

def add_loss(net, args):
    net['label_onehot'], curr = dup(tf.one_hot(net['label'], args.n_classes))
    with tf.name_scope('loss'):
        tf.losses.softmax_cross_entropy(curr, net['out'])
        # add regularization
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if len(reg_losses) > 0:
            reg = 1./len(reg_losses) * tf.add_n([args.regularization_weight*l for l in reg_losses])
        else:
            reg = 0
        loss = tf.add_n(tf.get_collection(tf.GraphKeys.LOSSES))
        net['loss'] = reg + loss

    global_step = tf.Variable(0, name='global_step', trainable=False)
    net['epoch'] = tf.Variable(0, name='epoch', trainable=False)
    net['learning_rate'] = tf.Variable(args.learning_rate[0],
                                       name='learning_rate',
                                       trainable=False,
                                       dtype=tf.float32)
    net['learning_rate_in'] = tf.placeholder('float32', shape=[])
    
    net['update_epoch'] = tf.assign(net['epoch'], net['epoch'] + 1)
    net['update_learning_rate'] = tf.assign(net['learning_rate'], net['learning_rate_in'])

    # batch norm ops are in UPDATE_OPS; add them as depency to train_op
    with tf.name_scope('train'):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if args.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=net['learning_rate'])
            elif args.optimizer == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate=net['learning_rate'],
                                                       momentum=0.9,
                                                       use_nesterov=True)

            # this allows us to visualize gradients:
            net['train_op'] = tf.contrib.layers.optimize_loss(net['loss'],
                                                              global_step,
                                                              learning_rate=None,
                                                              optimizer=optimizer,
                                                              summaries=['gradients',
                                                                         'gradient_norm',
                                                                         'learning_rate'])

    with tf.name_scope('accuracies'):
        net['all_summ'] = tf.summary.merge_all()

        net['train_acc'] = tf.contrib.metrics.accuracy(tf.argmax(net['out'], axis=1),
                                                       net['label'])
        net['acc'] = tf.contrib.metrics.accuracy(tf.argmax(net['out'], axis=1),
                                                 net['label'])
        net['confusion'] = tf.confusion_matrix(net['label'],
                                               tf.argmax(net['out'], axis=1),
                                               dtype='float32',
                                               num_classes=args.n_classes)
        net['train_summ'] = tf.summary.merge([tf.summary.scalar('train_acc', net['train_acc']),
                                              tf.summary.scalar('total_loss', net['loss']),
                                              tf.summary.scalar('triplet_loss', net['triplet_loss']),
                                              tf.summary.scalar('loss', loss),
                                              tf.summary.scalar('reg_loss', reg)])


        # we'll update after we compute the mean among all batches
        net['valid_acc_epoch'] = tf.placeholder('float32', shape=[])
        net['valid_summ'] = tf.summary.scalar('valid_acc', net['valid_acc_epoch'])
        net['valid_confusion_epoch'] = tf.placeholder('float32', shape=(args.n_classes, args.n_classes))
        net['valid_confusion_summ'] = tf.summary.image('valid_confusion',
                                                       net['valid_confusion_epoch'][np.newaxis, ..., np.newaxis])

    return net


def loop(sess, init, feed_dict, outputs, postprocess, max_loops=np.inf, feed_dict_init={}):
    """ Run tf operations until tf.errors.OutOfRangeError """
    assert len(outputs) == len(postprocess)
    all_out = [[] for _, _ in enumerate(outputs)]
    n_loops = 0
    sess.run(init, feed_dict=feed_dict_init)
    while True:
        try:
            out = sess.run(outputs, feed_dict=feed_dict)
            for o, p, a in zip(out, postprocess, all_out):
                if p == 'list':
                    a.append(o)
                elif callable(p):
                    p(o)
                elif p == 'ignore':
                    pass
                else:
                    raise ValueError('Unrecognized option: {}'.format(p))
            n_loops += 1
            if n_loops >= max_loops:
                break
        except tf.errors.OutOfRangeError:
            break

    return all_out
 

def Dataset_train(net, dset, args):
    """ training function using tf Dataset class"""

    max_loops = args.first_n if args.first_n is not None else np.inf
    indim = get_indim(args)

    # we hack iterators init functions to feed circular list of filenames to
    if args.dset == 'from_cached_tfrecords':
        # keep validation in RAM:
        assert len(dset[1]['fnames']['val']) == 1
        x, y = util.tfrecord2np(dset[1]['fnames']['val'][0],
                                indim,
                                dtype=args.dtype)

        feed_dict_init = {'val': {dset[1]['x']: x, dset[1]['y']: y}}
        fnames_list = dset[1]['fnames']['train']
        fnames_list = itertools.cycle(fnames_list)
        init_iter = {t: dset[0].initializer for t in ['train', 'val','test','noise','miss']}

    else:
        feed_dict_init = {'val': {}, 'train': {}}
        init_iter = {t: dset[1][t] for t in ['train', 'val']}

    saver = tf.train.Saver()
    # note: graph is finalized after supervisor instantiation
    sv = tf.train.Supervisor(logdir=args.logdir,
                             summary_op=None,
                             save_model_secs=0)

    with sv.managed_session(config=tf_config()) as sess:
        epoch = net['epoch'].eval(session=sess)
        train_time, valid_time = [], []
        best_valid_acc = 0
        train_acc, valid_acc = np.zeros(2)
        pod_epoch = 0
        while epoch < args.n_epochs:
            lr = args.learning_rate.get(epoch, None)
            if lr is not None:
                sess.run(net['update_learning_rate'],
                         feed_dict={net['learning_rate_in']: lr})

            # load training chunk to RAM
            t0 = time.time()
            if args.dset == 'from_cached_tfrecords':
                f = fnames_list.__next__()
                # print('caching {}'.format(f))
                x, y = util.tfrecord2np(f, indim, dtype=args.dtype)

                feed_dict_init['train'] = {dset[1]['x']: x, dset[1]['y']: y}

                # shuffle
                x, y = util.shuffle_all(x, y)

            if args.round_batches:
                max_loops = len(x) // args.train_bsize
            t0 = time.time()
            _, train_acc, _ = loop(sess,
                                   init_iter['train'],
                                   {net['training']: True},
                                   [net['train_op'], net['train_acc'], net['train_summ']],
                                   ['ignore', 'list', lambda x: sv.summary_computed(sess, x)],
                                   max_loops=max_loops,
                                   feed_dict_init=feed_dict_init['train'])
            train_acc = np.mean(train_acc)
            train_time.append(time.time() - t0)

            # print('train time {:.2f}'.format(train_time[-1]))

            # validate
            if not args.skip_val:
                if args.round_batches:
                    raise NotImplementedError()

                t0 = time.time()
                valid_acc, valid_confusion = loop(sess,
                                                  init_iter['val'],
                                                  {net['training']: False},
                                                  [net['acc'], net['confusion']],
                                                  ['list', 'list'],
                                                  max_loops=max_loops,
                                                  feed_dict_init=feed_dict_init['val'])
                valid_acc = np.mean(valid_acc)
                valid_confusion = np.array(sum(valid_confusion))
                valid_confusion /= valid_confusion.sum(axis=0)
                valid_time.append(time.time() - t0)
                # print('valid time {:.2f}'.format(valid_time[-1]))
                
                x, y = util.tfrecord2np(dset[1]['fnames']['test'][0],
                                        indim,
                                        dtype=args.dtype)

                feed_dict_init['test'] = {dset[1]['x']: x, dset[1]['y']: y}

                test_acc, test_conf = loop(sess,
                                           init_iter['test'],
                                           {net['training']: False},
                                           [net['acc'], net['confusion']],
                                           ['list', 'list'],
                                           max_loops=args.test_n if args.test_n > 0 else np.inf,
                                           feed_dict_init=feed_dict_init['test'])

                # weighted average per batch size (bc last batch may have different size)
                test_acc = np.mean(test_acc)
                try:
                    x, y = util.tfrecord2np(dset[1]['fnames']['noise'][0],
                                            indim,
                                            dtype=args.dtype)

                    feed_dict_init['noise'] = {dset[1]['x']: x, dset[1]['y']: y}

                    noise_acc, noise_conf = loop(sess,
                                               init_iter['noise'],
                                               {net['training']: False},
                                               [net['acc'], net['confusion']],
                                               ['list', 'list'],
                                               max_loops=args.test_n if args.test_n > 0 else np.inf,
                                               feed_dict_init=feed_dict_init['noise'])

                    noise_acc = np.mean(noise_acc)
                except:
                    print("error loading noise data")
                    noise_acc = 0
                try:
                    x, y = util.tfrecord2np(dset[1]['fnames']['miss'][0],
                                            indim,
                                            dtype=args.dtype)

                    feed_dict_init['miss'] = {dset[1]['x']: x, dset[1]['y']: y}

                    misp_acc, misp_conf = loop(sess,
                                               init_iter['miss'],
                                               {net['training']: False},
                                               [net['acc'], net['confusion']],
                                               ['list', 'list'],
                                               max_loops=args.test_n if args.test_n > 0 else np.inf,
                                               feed_dict_init=feed_dict_init['miss'])

                    misp_acc = np.mean(misp_acc)
                except:
                    print("error loading  missing points data")
                    misp_acc = 0    
            else:
                valid_acc = train_acc
                valid_confusion = np.zeros((args.n_classes, args.n_classes))
                valid_time.append(0)
                test_acc= 0
                noise_acc= 0
                misp_acc= 0
            # compute summaries once per epoch
            summ = sess.run([net['valid_summ'], net['valid_confusion_summ']],
                            feed_dict={net['valid_acc_epoch']: valid_acc,
                                       net['valid_confusion_epoch']: valid_confusion})
            [sv.summary_computed(sess, s) for s in summ]

            epoch = sess.run(net['update_epoch'])
            lr = sess.run(net['learning_rate'])

            # always save the best model so far
            if valid_acc > best_valid_acc:
                saver.save(sess, os.path.join(args.logdir, 'best.ckpt'))
                best_valid_acc = valid_acc
            # and overwrite the latest model
            saver.save(sess, os.path.join(args.logdir, 'latest.ckpt'))

            logger.info('epoch={}; lr={:.4f} train: {:.4f}, valid: {:.4f}, augm: {:.4f}, noise: {:.4f}, missp: {:.4f}'
                        .format(epoch, lr, train_acc, valid_acc, test_acc, noise_acc, misp_acc))

        train_time = np.mean(train_time)
        valid_time = np.mean(valid_time)
        saver.save(sess, os.path.join(args.logdir, 'final.ckpt'))

    return train_acc, valid_acc, train_time, valid_time


def Dataset_test(net, dset, args, ckpt='final.ckpt'):
    indim = get_indim(args)

    if args.dset == 'from_cached_tfrecords':
        res, nch = args.input_res, args.nchannels
        assert len(dset[1]['fnames']['test']) == 1
        x, y = util.tfrecord2np(dset[1]['fnames']['test'][0],
                                indim,
                                dtype=args.dtype)
        feed_dict_init = {'test': {dset[1]['x']: x, dset[1]['y']: y}}
        feed_dict_init['test'] = {dset[1]['x']: x, dset[1]['y']: y}
        # init_iter = dset[0].initializer
        init_iter = {t: dset[0].initializer for t in [ 'val', 'test', 'noise', 'miss']}

        weights = [args.train_bsize] * (len(x) // args.train_bsize)
        rem = len(x) % args.train_bsize
        if rem > 0:
            weights.append(rem)
        weights1 = weights / np.array(weights).sum()

        x, y = util.tfrecord2np(dset[1]['fnames']['val'][0],
                                indim,
                                dtype=args.dtype)

        feed_dict_init['val'] = {dset[1]['x']: x, dset[1]['y']: y}
        weights = [args.train_bsize] * (len(x) // args.train_bsize)
        rem = len(x) % args.train_bsize
        if rem > 0:
            weights.append(rem)
        weight = weights / np.array(weights).sum()


        x, y = util.tfrecord2np(dset[1]['fnames']['noise'][0],
                                indim,
                                dtype=args.dtype)

        feed_dict_init['noise'] = {dset[1]['x']: x, dset[1]['y']: y}

        weights = [args.train_bsize] * (len(x) // args.train_bsize)
        rem = len(x) % args.train_bsize
        if rem > 0:
            weights.append(rem)
        weights2 = weights / np.array(weights).sum()

        x, y = util.tfrecord2np(dset[1]['fnames']['miss'][0],
                                indim,
                                dtype=args.dtype)

        feed_dict_init['miss'] = {dset[1]['x']: x, dset[1]['y']: y}

        weights = [args.train_bsize] * (len(x) // args.train_bsize)
        rem = len(x) % args.train_bsize
        if rem > 0:
            weights.append(rem)
        weights3 = weights / np.array(weights).sum()


    else:
        feed_dict_init = {}
        init_iter = dset[1]['test']

    saver = tf.train.Saver()
    test_acc, test_conf = [], []

    t0 = time.time()
    with tf.Session(config=tf_config()).as_default() as sess:
        saver.restore(sess, os.path.join(args.logdir, ckpt))
        test_acc, test_conf = loop(sess,
                                   init_iter,
                                   {net['training']: False},
                                   [net['acc'], net['confusion']],
                                   ['list', 'list'],
                                   max_loops=args.test_n if args.test_n > 0 else np.inf,
                                   feed_dict_init=feed_dict_init['test'])
        val_acc, val_conf = loop(sess,
                                     init_iter['val'],
                                     {net['training']: False},
                                     [net['acc'], net['confusion']],
                                     ['list', 'list'],
                                     max_loops=args.test_n if args.test_n > 0 else np.inf,
                                     feed_dict_init=feed_dict_init['val'])

         # ayman
        try:


            noise_acc, noise_conf = loop(sess,
                                         init_iter['noise'],
                                         {net['training']: False},
                                         [net['acc'], net['confusion']],
                                         ['list', 'list'],
                                         max_loops=args.test_n if args.test_n > 0 else np.inf,
                                         feed_dict_init=feed_dict_init['noise'])


        except:
            print("error loading noise data")
            noise_acc = 0
        try:

            misp_acc, misp_conf = loop(sess,
                                       init_iter['miss'],
                                       {net['training']: False},
                                       [net['acc'], net['confusion']],
                                       ['list', 'list'],
                                       max_loops=args.test_n if args.test_n > 0 else np.inf,
                                       feed_dict_init=feed_dict_init['miss'])


        except:
            print("error loading  missing points data")
            misp_acc = 0

            # weighted average per batch size (bc last batch may have different size)
    # weights = [args.train_bsize] * (len(x) // args.train_bsize)
    # rem = len(x) % args.train_bsize
    # if rem > 0:
    #     weights.append(rem)
    # weights = weights / np.array(weights).sum()
    val_acc = np.sum(val_acc * weight)
    test_acc = np.sum(test_acc * weights1)
    misp_acc= np.sum(misp_acc * weights3)
    noise_acc = np.sum(noise_acc * weights2)

    # confusion matrix
    test_conf = np.array(sum(test_conf))
    test_conf = test_conf / test_conf.sum(axis=1, keepdims=True)

    return test_acc, test_conf , noise_acc, misp_acc , val_acc


def get_tfrecord_activations(basedir, fname_or_xy, layers,
                             ckptfile=None, max_batches=np.inf, args_in=None, xy=None, **kwargs):
    """ Load model from 'basedir', compute 'layer' activations on 'fname_or_xy' """
    assert isinstance(layers, list)
    assert isinstance(layers[0], str)

    model, args, dset = load_model_dset(basedir, args_in if args_in is not None else {})
    if isinstance(fname_or_xy, str):
        x, y = util.tfrecord2np(fname_or_xy,
                                get_indim(args),
                                dtype=args.dtype)
    else:
        x, y = fname_or_xy

    feed_dict_init = {dset[1]['x']: x, dset[1]['y']: y}
    init_iter = dset[0].initializer

    saver = tf.train.Saver()
    t0 = time.time()

    with tf.Session(config=tf_config()).as_default() as sess:
        saver.restore(sess, os.path.join(basedir, ckptfile))
        out = loop(sess,
                   init_iter,
                   {model['training']: False},
                   [model[l] for l in layers],
                   ['list']*len(layers),
                   max_loops=args.test_n if args.test_n > 0 else max_batches,
                   feed_dict_init=feed_dict_init)

    return {l: np.concatenate(o) for l, o in zip(layers, out)}


def load(modeldir, args_in={}):
    with open(os.path.join(modeldir, 'flags.pkl'), 'rb') as fin:
        args = pickle.load(fin)
    args = params.parse({**args.__dict__, **args_in})
    tf.reset_default_graph()
    kwa = {'dset': datasets.load(args)} if args.model == 'two_branch' else {}
    net = globals()[args.model](args, **kwa)
    net = add_loss(net, args)

    return net, args


def load_model_dset(modeldir, args_in={}):
    """ Load model and dataset.
    Because some models require the dataset structure at definition time
    """
    with open(os.path.join(modeldir, 'flags.pkl'), 'rb') as fin:
        args = pickle.load(fin)
    args = params.parse({**args.__dict__, **args_in})
    tf.reset_default_graph()
    dset = datasets.load(args)
    net = globals()[args.model](args, dset=dset)
    net = add_loss(net, args)

    return net, args, dset
        

def get_indim(args):
    res, nch = args.input_res, args.nchannels
    if args.spectral_input:
        indim = (-1, 2, res, res, nch)
    else:
        indim = (-1, res, res, nch)

    return indim
