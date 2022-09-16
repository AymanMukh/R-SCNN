
import os
import tensorflow as tf
import numpy as np
import provider

# notes : rotations are 23 , then they are repeated.
# y is label.


shape_names = ['airplane','bag','basket','bathtub','bed','bench','birdhouse','bookshelf','bottle','bowl','bus','cabinet','can','camera','cap','car','cellphone','chair','clock','dishwasher','earphone','faucet','file','guitar','helmet','jar','keyboard','knife','lamp','laptop','mailbox','microphone','microwave','monitor','motorcycle','mug','piano','pillow','pistol','pot','printer','remote_control','rifle','rocket','skateboard','sofa','speaker','stove','table','telephone','tin_can','tower','train','vessel','washer']



def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_and_save_to(images, theta, labels , name, params, rot=0):
        num_examples = images.shape[0]
        print(num_examples)
       # print(images.shape)
        #num_examples = 2000
        filename = os.path.join(params['tfrecord_location'], name + '.tfrecord')
        print('Writing', filename)
        with tf.python_io.TFRecordWriter(filename) as writer:
                for index in range(num_examples):
                        # 1. Convert your data into tf.train.Feature
#                        x = images[index, :, : ].tostring()
                        x = images[index,:,:, 0].tostring()
                        x1 = images[index, :,: , 1].tostring()
                        x2 = images[index, :,: , 2].tostring()
                        x3 = images[index, :,: , 3].tostring()
                        x4 = images[index, :,: , 4].tostring()
                        x5 = images[index, :,: , 5].tostring()
                        x6 = images[index, :,: , 6].tostring()
#                        x7 = images[index, :,: , 7].tostring()
#                        x8 = images[index, :,: , 8].tostring()
#                        x9 = images[index, :,: , 9].tostring()
#                        x10 = images[index, :,:, 10].tostring()
#                        x11 = images[index, :,: , 11].tostring()
#                        x12 = images[index, :,: , 12].tostring()
#                        x13 = images[index, :,: , 13].tostring()
                                
                        
#                        z = theta[index, :,:,   0].tostring()
#                        z1 = theta[index, :,: , 1].tostring()
#                        z2 = theta[index, :,: , 2].tostring()
#                        z3 = theta[index, :,: , 3].tostring()
#                        z4 = theta[index, :,: , 4].tostring()
#                        z5 = theta[index, :,:,  5].tostring()
#                        z6 = theta[index, :,: , 6].tostring()
#                        z7 = theta[index, :,: , 7].tostring()
#                        z8 = theta[index, :,: , 8].tostring()
#                        z9 = theta[index, :,: , 9].tostring()


                     #   x10 = images[index, :,: , 10].tostring()
                        # x3 = images[3,:,:,index].tostring()
                        # x4 = images[4,:,:,index].tostring()


                        feature = {
                                'fname': _bytes_feature(bytes(shape_names[int(labels[index])-1],encoding='utf8')),
                                'y': _int64_feature(int(labels[index])),
                                'x': _bytes_feature(x),
                                'x1': _bytes_feature(x1),
                                'x2': _bytes_feature(x2),
                                'x3': _bytes_feature(x3),
                                'x4': _bytes_feature(x4),
                                'x5': _bytes_feature(x5),
                                'x6': _bytes_feature(x6),
#                                'x7': _bytes_feature(x7),
#                                'x8': _bytes_feature(x8),
#                                'x9': _bytes_feature(x9),
#                                'x10': _bytes_feature(x10),
#                                'x11': _bytes_feature(x11),
#                                'x12': _bytes_feature(x12),
#                                'x13': _bytes_feature(x13),                                
                                
#                               'z': _bytes_feature(z),
#                                'z1': _bytes_feature(z1),
#                                'z2': _bytes_feature(z2),
#                                'z3': _bytes_feature(z3),
#                                'z4': _bytes_feature(z4),
#                                'z5': _bytes_feature(z5),
#                                'z6': _bytes_feature(z6),


                           #     'x10': _bytes_feature(x8),
                                'idrot': _int64_feature(rot),
                              
                        }
                        features = tf.train.Features(feature=feature)
                        example = tf.train.Example(features=features)
                        writer.write(example.SerializeToString())
                        
                        

params = {}
loc='data/shapenet/7zgn/'

params['download_data_location'] = 'data'
params['tfrecord_location'] = loc

#current_data, current_label, theta = provider.load_h5_theta(loc+'voxltout.50.h5')
#current_label=np.reshape(current_label, (-1))
## print(current_data.shape)
#convert_and_save_to(current_data, theta, current_label, 'voxltout.50', params, rot=0)

#current_data, current_label, theta = provider.load_h5_theta(loc+'voxlt0.h5')
#current_label=np.reshape(current_label, (-1))
## print(current_data.shape)
#convert_and_save_to(current_data, theta, current_label, 'val', params, rot=0)

#current_data, current_label, theta = provider.load_h5_theta(loc+'voxltt0.h5')
#current_label=np.reshape(current_label, (-1))
## print(current_data.shape)
#convert_and_save_to(current_data, theta, current_label, 'test', params, rot=0)


for i in range(0, 11):
  current_data, current_label, theta = provider.load_h5_theta(loc+'voxl'+ str(i)+ '.h5')
  current_label=np.reshape(current_label, (-1))
  # theta=[]
  convert_and_save_to(current_data, theta, current_label, 'train'+ str(5+i), params, rot=0)


