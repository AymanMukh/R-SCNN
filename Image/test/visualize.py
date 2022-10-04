import sys
import os
import warnings
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from os.path import abspath, dirname
sys.path.append(dirname(dirname(abspath(__file__))))

from spherical_cnn import sphericalcof

from spherical_cnn import datasets

try:
    import pyshtools
except:
    pass


from spherical_cnn import util
from spherical_cnn.util import tf_config
from spherical_cnn import models


shape = 1
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--dset_dir', default='/media/SSD/DATA/ayman/papers/spherical-cnn/data/test/', help=' datasers directory  ')
parser.add_argument('--order', '-or', type=int, default=16,help='order of coefficients')
parser.add_argument('--nchannels', default=1, type=int, help='Number of input channels')
parser.add_argument('--input_res', '-res', type=int, default=64,help='resolution for spherical inputs; may subsample if larger')
# parser.add_argument('--n_classes', '-nc', type=int, default=40, help='number of classes in dataset')
# parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')

parser.add_argument('--dset', '-d', type=str, default='from_cached_tfrecords',                help='dataset loader')
parser.add_argument('--logdir', '-ld', type=str, default='', help='directory to save models, logs and checkpoints.')
parser.add_argument('--dtype',type=str, default='float32', )
parser.add_argument('--train_bsize', type=int, default=1)
parser.add_argument('spectral_input', action='store_false', default=False)

args = parser.parse_args()

assert os.path.isdir(args.dset_dir)

dset = datasets.load(args)
indim = models.get_indim(args)
# dsetarg = {'dset': dset}

# read data
x, y = util.tfrecord2np(dset[1]['fnames']['test'][0],
                                indim,
                                dtype=args.dtype)
x= x/np.max(x)
# X=(X>0)*1

#choose record
# for i in range(len(y)):
#     if y[i]==shape:
#         inputs = x[i]
#         break

# x, y = util.tfrecord2np(dset[1]['fnames']['test'][0],
#                                 indim,
#                                 dtype=args.dtype)
inputs = x[0]
inputs_out = x[1]

# choose centric sphere
# input = inputs[...,8]
# input_out = inputs_out[...,8]
# if input.any()==0:
#     print('zero elements')
#     for i in range(3,10):
#         input = inputs[..., i]
#         input_out = inputs_out[..., i]
#         if input.any() != 0:
#             break

# f = open('datain.txt', 'ab')
# inp = np.array(x)
#
# for jj in range(7):
#     # f.write(str(inp[0,: ,:,jj]))
#     np.savetxt(f, inp[ 0, :, :, jj], fmt="%s")
# for jj in range(7):
#     np.savetxt(f, inp[ 1, :, :, jj], fmt="%s")
# f.close()



# generate sh for resulotion res and a given order
# sh = sphericalcof.sph_harm_all(args.input_res, order=args.input_res//4)

# coeff= sphericalcof.sph_harm_transform(input, harmonics=sh,order=args.input_res//4)
coeff= sphericalcof.sph_harm_transform_batch(x, order=args.input_res//4)
coeff= tf.reshape(coeff, [2, -1])
f = open('datainc.txt', 'ab')
inp = np.array(x)
np.savetxt(f, inp[ 0, :])
np.savetxt(f, inp[ 1, :])
f.close()



coeff_out= sphericalcof.sph_harm_transform(input_out, harmonics=sh,order=args.order)

def stack_uneven(arrays, fill_value=0.):
    '''
    Fits arrays into a single numpy array, even if they are
    different sizes. `fill_value` is the default value.

    Args:
            arrays: list of np arrays of various sizes
                (must be same rank, but not necessarily same size)
            fill_value (float, optional):

    Returns:
            np.ndarray
    '''
    # sizes = [np.shape(a) for a in arrays]
    results=[]
    # max_sizes = np.max(list(zip(*sizes)), -1)
    # The resultant array has stacked on the first dimension
    # result = np.full((len(arrays),) + tuple(max_sizes), fill_value)
    for i, a in enumerate(arrays):
      # The shape of this array `a`, turned into slices
      # slices = tuple(slice(0,s) for s in sizes[i])
      # Overwrite a block slice of `result` with this array `a`
      # result[i][slices] = a
      results = np.append(results, a)
    return results


coeff_s=stack_uneven(coeff)
coeff_outs=stack_uneven(coeff_out)
norm=coeff_outs/coeff_s

# plot coeff
# Prepare the data
degrees = np.linspace(0,args.order**2,args.order**2)
# Plot the data
# plt.plot(degrees,abs(coeff_s), label='clean')
# plt.plot(degrees,abs(coeff_outs) , label='outliers')

plt.plot(degrees,abs(norm) , label='norm')
# Add a legend
plt.legend()
# Show the plot
plt.show()

# output= sphericalcof.sph_harm_inverse(coeff, res=args.input_res, harmonics=sh)

