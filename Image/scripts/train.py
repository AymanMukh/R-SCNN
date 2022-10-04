#!/usr/bin/env python3
""" Main training script """

import os
os.environ["CUDA_VISIBLE_DEVICES"]="5"
import logging
import pickle
import glob
import socket
# CUDA_VISIBLE_DEVICES="1"
import numpy as np
import tensorflow as tf

 # @params/model-64.txt                                @params/m40-64.txt                                @params/training.txt                                --dset_dir /media/SSD/DATA/ayman/papers/spherical-cnn/data/2k/7gz  --logdir tmp/newexp                --run_id m40-so3
# argh!
import sys
from os.path import abspath, dirname
sys.path.append(dirname(dirname(abspath(__file__))))
from spherical_cnn import params
from spherical_cnn import   models #   I changed this
from spherical_cnn import datasets
import argparse



def init_logger(args):
    # create logger
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    # create file handler
    logdir = os.path.expanduser(args.logdir)
    os.makedirs(logdir, exist_ok=True)
    logfile = os.path.join(logdir, 'logging.log')
    fh = logging.FileHandler(logfile)
    # create console handler
    ch = logging.StreamHandler()
    # create formatter and add it to the handlers
    formatter = logging.Formatter('[%(asctime)s:%(levelname)s] %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def main():
    args = params.parse()
    # args1 = args
    logger = init_logger(args)
    logger.info('Running on {}'.format(socket.gethostname()))

    if not args.test_only:
        logger.info(args)
        logger.info('Loading dataset from {}...'.format(args.dset))
        dset = datasets.load(args)

        with open(os.path.join(args.logdir, 'flags.pkl'), 'wb') as fout:
            pickle.dump(args, fout)

        logger.info('Loading model {}. Logdir {}'.format(args.model, args.logdir))
        if os.path.isdir(args.logdir):
            if args.from_scratch:
                logger.info('Training from scratch; removing existing data in {}'.format(args.logdir))
                for f in (glob.glob(args.logdir + '/*ckpt*') +
                          glob.glob(args.logdir + '/events.out*') +
                          glob.glob(args.logdir + '/checkpoint') +
                          glob.glob(args.logdir + '/graph.pbtxt')):
                    os.remove(f)
            else:
                logger.info('Continuing from checkpoint in {}'.format(args.logdir))

        dsetarg = {'dset': dset}
        net = models.get_model(args.model, args, **dsetarg)
        logger.info('trainable parameters {}'.format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
        
        logger.info('Start training...')
        trainfun = models.Dataset_train
        train, valid, train_time, _ = trainfun(net, dset, args)

    else:
        dset_dir = args.dset_dir
        with open(os.path.join(args.logdir, 'flags.pkl'), 'rb') as fin:
            args = pickle.load(fin)
        args = params.parse(args.__dict__)
        args.dset_dir = dset_dir
        # args.nfilters = args1.nfilters
        # args.pool_layers = args1.pool_layers
        train, valid, train_time = 0, 0, 0
        logger.info(args)

    logger.info('Start testing...')
    tf.reset_default_graph()
    # need to reload tf dataset because graph was reset
    dset = datasets.load(args)

    dsetarg = {'dset': dset}
    net = models.get_model(args.model, args, **dsetarg)
    logger.info('trainable parameters {}'.format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
    test_final, conf_final, n, m ,v = models.Dataset_test(net, dset, args, 'final.ckpt')

    logger.info('|{}|{}|{}|{}|{}|{}|{}|'
                .format('model', 'train', 'val',
                        'test', 'noise','missp', 'train time'))
    logger.info('|{}|{:.4f}|{:.4f}|{:.4f}|{:.4f}|{:.4f}|{:.2f}|'
                .format(args.run_id, train, v, test_final,n,m,
                        train_time))

    # f = open('conf.txt', 'ab')
    # np.savetxt(f, conf_final, fmt="%s")
    # f.close()

if __name__ == '__main__':
    main()
