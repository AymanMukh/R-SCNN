import sys

import tensorflow as tf
import numpy as np
try:
    from pyshtools import SHGrid
except:
    pass


def sph_sample(n, mode='DH'):
    """ Sample grid on a sphere.

    Args:
        n (int): dimension is n x n
        mode (str): sampling mode; DH or GLQ

    Returns:
        theta, phi (1D arrays): polar and azimuthal angles
    """
    assert n % 2 == 0
    j = np.arange(0, n)
    if mode == 'DH':
        return j*np.pi/n, j*2*np.pi/n
    elif mode == 'ours':
        return (2*j+1)*np.pi/2/n, j*2*np.pi/n
    elif mode == 'GLQ':
        from pyshtools.shtools import GLQGridCoord
        phi, theta = GLQGridCoord(n-1)
        # convert latitude to [0, np.pi/2]
        return np.radians(phi+90), np.radians(theta)
    elif mode == 'naive':
        # repeat first and last points; useful for plotting
        return np.linspace(0, np.pi, n), np.linspace(0, 2*np.pi, n)


def sphrot_shtools(f, x, lmax=None, latcols=True):
    """ Rotate function on sphere f by Euler angles x (Z-Y-Z?)  """
    if 'pyshtools' not in sys.modules:
        raise ImportError('pyshtools not available!')

    if latcols:
        f = f.T
    c = SHGrid.from_array(f).expand()
    c_r = c.rotate(*x, degrees=False)
    f_r = c_r.expand(lmax=lmax).to_array()
    if latcols:
        f_r = f_r.T
    
    return f_r


def tfrecord2np(fname, shape,
                dtype='float32', get_meta=False, max_records=np.inf):
    """ Load tfrecord containing serialized tensors x and y as numpy arrays. """
    example = tf.train.Example()
    X = []
    X1=[]
    X2=[]
    X3=[]
    X4=[]
    X5=[]
    X6=[]
    X7=[]
    X8=[]
    X9=[]
    X10=[]
    X11=[]
    X12=[]
    X13=[]
    X14=[]
    X15=[]
    X16=[]
    X17=[]
    X18=[]
    X19=[]
    X20=[]
    X21=[]
    X22=[]
    X23=[]
    X24=[]
    X25=[]
    X26=[]
    X27=[]
    X28=[]
    X29=[]
    X30=[]
    X31=[]
    Y = []
    # dataset may contain angles
    A = []
    meta = []
    for i, record in enumerate(tf.python_io.tf_record_iterator(fname)):
        example.ParseFromString(record)
        f = example.features.feature
        X.append(np.fromstring(f['x'].bytes_list.value[0], dtype=dtype))
        X1.append(np.fromstring(f['x1'].bytes_list.value[0], dtype=dtype))
        X2.append(np.fromstring(f['x2'].bytes_list.value[0], dtype=dtype))
        X3.append(np.fromstring(f['x3'].bytes_list.value[0], dtype=dtype))
        X4.append(np.fromstring(f['x4'].bytes_list.value[0], dtype=dtype))
        X5.append(np.fromstring(f['x5'].bytes_list.value[0], dtype=dtype))
        X6.append(np.fromstring(f['x6'].bytes_list.value[0], dtype=dtype))

        # X7.append(np.fromstring(f['x7'].bytes_list.value[0], dtype=dtype))
        # X8.append(np.fromstring(f['x8'].bytes_list.value[0], dtype=dtype))
        # X9.append(np.fromstring(f['x9'].bytes_list.value[0], dtype=dtype))
        # X10.append(np.fromstring(f['x10'].bytes_list.value[0], dtype=dtype))
        # X11.append(np.fromstring(f['x11'].bytes_list.value[0], dtype=dtype))
        # X12.append(np.fromstring(f['x12'].bytes_list.value[0], dtype=dtype))
        # X13.append(np.fromstring(f['x13'].bytes_list.value[0], dtype=dtype))

        # X14.append(np.fromstring(f['x14'].bytes_list.value[0], dtype=dtype))
        # X15.append(np.fromstring(f['x15'].bytes_list.value[0], dtype=dtype))
        # X16.append(np.fromstring(f['x16'].bytes_list.value[0], dtype=dtype))
        # X17.append(np.fromstring(f['x17'].bytes_list.value[0], dtype=dtype))
        # X18.append(np.fromstring(f['x18'].bytes_list.value[0], dtype=dtype))
        # X19.append(np.fromstring(f['x19'].bytes_list.value[0], dtype=dtype))
        # X20.append(np.fromstring(f['x20'].bytes_list.value[0], dtype=dtype))
        # X21.append(np.fromstring(f['x21'].bytes_list.value[0], dtype=dtype))
        # X22.append(np.fromstring(f['x22'].bytes_list.value[0], dtype=dtype))
        # X23.append(np.fromstring(f['x23'].bytes_list.value[0], dtype=dtype))
        # X24.append(np.fromstring(f['x24'].bytes_list.value[0], dtype=dtype))
        # X25.append(np.fromstring(f['x25'].bytes_list.value[0], dtype=dtype))
        # X26.append(np.fromstring(f['x26'].bytes_list.value[0], dtype=dtype))
        # X27.append(np.fromstring(f['x27'].bytes_list.value[0], dtype=dtype))
        # X28.append(np.fromstring(f['x28'].bytes_list.value[0], dtype=dtype))
        # X29.append(np.fromstring(f['x29'].bytes_list.value[0], dtype=dtype))
        # X30.append(np.fromstring(f['x30'].bytes_list.value[0], dtype=dtype))
        # X31.append(np.fromstring(f['x31'].bytes_list.value[0], dtype=dtype))

#        X14.append(np.fromstring(f['z'].bytes_list.value[0], dtype=dtype))
#        X15.append(np.fromstring(f['z1'].bytes_list.value[0], dtype=dtype))
#        X16.append(np.fromstring(f['z2'].bytes_list.value[0], dtype=dtype))
#        X17.append(np.fromstring(f['z3'].bytes_list.value[0], dtype=dtype))
#        X18.append(np.fromstring(f['z4'].bytes_list.value[0], dtype=dtype))
#        X19.append(np.fromstring(f['z5'].bytes_list.value[0], dtype=dtype))
#        X20.append(np.fromstring(f['z6'].bytes_list.value[0], dtype=dtype))

        # X21.append(np.fromstring(f['z7'].bytes_list.value[0], dtype=dtype))
        # X22.append(np.fromstring(f['z8'].bytes_list.value[0], dtype=dtype))
        # X23.append(np.fromstring(f['z9'].bytes_list.value[0], dtype=dtype))
        # X24.append(np.fromstring(f['z10'].bytes_list.value[0], dtype=dtype))
        # X25.append(np.fromstring(f['z11'].bytes_list.value[0], dtype=dtype))
        # X26.append(np.fromstring(f['z12'].bytes_list.value[0], dtype=dtype))
        # X27.append(np.fromstring(f['z13'].bytes_list.value[0], dtype=dtype))

        Y.append(f['y'].int64_list.value[0])
        if f.get('a', None) is not None:
            A.append(np.fromstring(f['a'].bytes_list.value[0], dtype=dtype))
        if get_meta:
            meta.append({'fname': f['fname'].bytes_list.value[0],
                         'idrot': f['idrot'].int64_list.value[0]})

        if i >= max_records-1:
            break

    if len(A) > 0:
        # dataset contain angle channel;
        # load 'a' and 'x', fix shapes and concatenate into channels
        shapea = (*shape[:-1], 1)
        A = np.stack(A).reshape(shapea)
        shape = (*shape[:-1], shape[-1] - 1)
        X = np.stack(X).reshape(shape)
        X = np.concatenate([X, A], axis=-1)
    else: 
        X = np.stack(X).reshape(shape)
        X1= np.stack(X1).reshape(shape)
        X2= np.stack(X2).reshape(shape)
        X3= np.stack(X3).reshape(shape)
        X4= np.stack(X4).reshape(shape)
        X5= np.stack(X5).reshape(shape)
        X6= np.stack(X6).reshape(shape)
        # X7= np.stack(X7).reshape(shape)
        # X8= np.stack(X8).reshape(shape)
        # X9= np.stack(X9).reshape(shape)
        # X10= np.stack(X10).reshape(shape)
        # X11= np.stack(X11).reshape(shape)
        # X12= np.stack(X12).reshape(shape)
        # X13= np.stack(X13).reshape(shape)

#        X14= np.stack(X14).reshape(shape)
#        X15= np.stack(X15).reshape(shape)
#        X16= np.stack(X16).reshape(shape)
#        X17= np.stack(X17).reshape(shape)
#        X18= np.stack(X18).reshape(shape)
#        X19= np.stack(X19).reshape(shape)
#        X20= np.stack(X20).reshape(shape)

        # X21= np.stack(X21).reshape(shape)
        # X22= np.stack(X22).reshape(shape)
        # X23= np.stack(X23).reshape(shape)
        # X24= np.stack(X24).reshape(shape)
        # X25= np.stack(X25).reshape(shape)
        # X26= np.stack(X26).reshape(shape)
        # X27= np.stack(X27).reshape(shape)
        # X28= np.stack(X28).reshape(shape)
        # X29= np.stack(X29).reshape(shape)
        # X30 = np.stack(X30).reshape(shape)
        # X31 = np.stack(X31).reshape(shape)


        X = np.concatenate([X,X1,X2,X3,X4,X5,X6,], axis=-1)
        # X = np.concatenate([X, X1, X2, X3, X4, X5, X6,X10,X11,X12,X13,X14,X15,X16 ], axis=-1)

    #    X = np.concatenate([X,X1,X2,X3,X4,X5,X6, X14,X15,X16,X17,X18,X19,X20], axis=-1)  # X21,X22,X23,X24,X25,X26,X27,X28,X29
      #  X = np.concatenate([X,X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16,X17,X18,X19,X20,X21,X22,X23,X24,X25,X26,X27,X28,X29,X30,X31],axis=-1)
      #   X= X/np.max(X)*200
        # X[...,:7] = (X[...,:7]>0).astype('float32')

        # X = np.concatenate( [X, X1, X2, X3, X4, X5, X6, X7, X8, X9], axis=-1)

        # print(np.max(X))
        # print(fname)
        # print(X.shape[0])
        if "train"  in fname:
             X[..., :7] = X[..., :7] / np.max(X[...,:7]) * 200
        else:
            X[...,:7] = X[...,:7] / np.amax(X[...,:7], axis=(1,2,3),keepdims=True )  * 200   #  np.max(X[...,:7] )
        # X[..., :7] = X[..., :7] / np.amax(X[..., :7], axis=(1, 2, 3), keepdims=True) * 200

       # X = np.concatenate([X,X10,X1,X11,X2,X12,X3,X13,X4,X14,X5,X15,X6,X16,X7,X17,X8,X18,X9,X19], axis=-1)

        
    Y = np.stack(Y)

    assert len(X) == len(Y)

    if get_meta:
        return X, Y, meta
    else:
        return X, Y


def tf_config():
    """ Default tensorflow config. """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    return config


def safe_cast(x, y):
    """ Cast x to type of y or y to type of x, without loss of precision.

    Works with complex and floats of any precision
    """
    t = 'complex' if (x.dtype.is_complex or y.dtype.is_complex) else 'float'
    s = max(x.dtype.size, y.dtype.size)
    dtype = '{}{}'.format(t, s*8)

    return tf.cast(x, dtype), tf.cast(y, dtype)


class AttrDict(dict):
    """ Dict that allows access like attributes (d.key instead of d['key']) .

    From: http://stackoverflow.com/a/14620633/6079076
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def shuffle_all(*args):
    """ Do the same random permutation to all inputs. """
    idx = np.random.permutation(len(args[0]))
    return [a[idx] for a in args]
