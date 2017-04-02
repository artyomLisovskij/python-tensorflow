from __future__ import print_function

import numpy as np
import tflearn
import dicom
import os
import h5py
import tensorflow
import tensorflow as tf


def collect_lungs(dir, image_size, cropper, patsize=-1, nonpatsize=-1):
    pat_list = os.listdir(dir + 'pat/')
    pat_list_size = len(pat_list)
    pat_list.sort()

    nonpat_list = os.listdir(dir + 'nonpat/')
    nonpat_list_size = len(nonpat_list)
    nonpat_list.sort()

    coef = 5
    # Create cropper
    # cropper = Cropper()

    if patsize > pat_list_size or patsize == -1:
        # print '[WARNING] Count of patalogy is lower then you want or patsize == -1'
        patsize = pat_list_size
    if nonpatsize > nonpat_list_size or nonpatsize == -1:
        # print '[WARNING] Count of nonpatalogy is lower then you want or nonpatsize == -1'
        nonpatsize = nonpat_list_size

    batch_size = (patsize + nonpatsize) * coef

    batch = np.ndarray(shape=(batch_size, image_size, image_size, 1), dtype='float32')
    labels = np.ndarray(shape=(batch_size, 2), dtype='int32')
    for i in range(0, patsize, coef):
        file_name = pat_list[i]
        image = dicom.read_file(dir + 'pat/' + file_name).pixel_array
        image = crop_lungs(image, cropper)
        image = resize(image, (image_size, image_size))

        rotated_right = skimage.transform.rotate(image, angle=(random.uniform(10, 25) + 0.0))
        rotated_right_resized = skimage.transform.rotate(image, angle=(random.uniform(10, 25) + 0.0), resize=True)
        rotated_left = skimage.transform.rotate(image, angle=-(random.uniform(10, 25) + 0.0))
        rotated_left_resized = skimage.transform.rotate(image, angle=-(random.uniform(10, 25) + 0.0), resize=True)

        batch[i, :, :, 0] = image
        batch[i + 1, :, :, 0] = resize(rotated_right, (image_size, image_size))
        batch[i + 2, :, :, 0] = resize(rotated_right_resized, (image_size, image_size))
        batch[i + 3, :, :, 0] = resize(rotated_left, (image_size, image_size))
        batch[i + 4, :, :, 0] = resize(rotated_left_resized, (image_size, image_size))
        # batch[i, :, :, 0] = scm.imresize(dicom.read_file(dir + 'pat/' +file_name).pixel_array, [image_size, image_size]) \
        #                     / 255.0
        # batch[i, :, :, 0] = dicom.read_file(dir + 'pat/' +file_name).pixel_array / 255.0
        labels[i, :] = [1, 0]
        labels[i + 1, :] = [1, 0]
        labels[i + 2, :] = [1, 0]
        labels[i + 3, :] = [1, 0]
        labels[i + 4, :] = [1, 0]
    for i in range(0, nonpatsize, coef):
        file_name = nonpat_list[i]
        image = dicom.read_file(dir + 'nonpat/' + file_name).pixel_array
        image = crop_lungs(image, cropper)
        image = resize(image, (image_size, image_size))

        rotated_right = skimage.transform.rotate(image, angle=(random.uniform(10, 25) + 0.0))
        rotated_right_resized = skimage.transform.rotate(image, angle=(random.uniform(10, 25) + 0.0), resize=True)
        rotated_left = skimage.transform.rotate(image, angle=-(random.uniform(10, 25) + 0.0))
        rotated_left_resized = skimage.transform.rotate(image, angle=-(random.uniform(10, 25) + 0.0), resize=True)

        batch[i + patsize, :, :, 0] = image
        batch[i + patsize + 1, :, :, 0] = resize(rotated_right, (image_size, image_size))
        batch[i + patsize + 2, :, :, 0] = resize(rotated_right_resized, (image_size, image_size))
        batch[i + patsize + 3, :, :, 0] = resize(rotated_left, (image_size, image_size))
        batch[i + patsize + 4, :, :, 0] = resize(rotated_left_resized, (image_size, image_size))
        # batch[i+patsize, :, :, 0] = scm.imresize(dicom.read_file(dir + 'nonpat/' +file_name).pixel_array, [image_size, image_size]) \
        #                             / 255.0
        # batch[i+patsize, :, :, 0] = dicom.read_file(dir + 'nonpat/'+ file_name).pixel_array / 255.0
        labels[i + patsize, :] = [0, 1]
        labels[i + patsize + 1, :] = [0, 1]
        labels[i + patsize + 2, :] = [0, 1]
        labels[i + patsize + 3, :] = [0, 1]
        labels[i + patsize + 4, :] = [0, 1]
    return batch, labels
    
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def deconv2d(x, W, shape):
  return tf.nn.conv2d_transpose(x, W, shape, strides=[1, 1, 1, 1], padding='SAME')

def batch_norm(x, n_out, phase_train, scope='bn'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def block(x, size, phase_train):
    W = weight_variable([3, 3, size, size])
    b = bias_variable([size])

    return tf.nn.relu(batch_norm(conv2d(x, W) + b, size, phase_train))

def create_net(x, y, phase_train):
    x_image = x
    W_conv1 = weight_variable([3, 3, 1, 16])
    b_conv1 = bias_variable([16])

    h_conv1 = tf.nn.relu(batch_norm(conv2d(x_image, W_conv1) + b_conv1, 16, phase_train))

    h_conv2 = block(h_conv1, 16, phase_train)
    h_conv3 = block(h_conv2, 16, phase_train)
    h_conv4 = block(h_conv3, 16, phase_train)
    h_conv5 = block(h_conv4, 16, phase_train)
    h_conv6 = block(h_conv5, 16, phase_train)
    h_conv7 = block(h_conv6, 16, phase_train)
    h_conv8 = block(h_conv7, 16, phase_train)
    h_conv9 = block(h_conv8, 16, phase_train)
    h_conv10 = block(h_conv9, 16, phase_train)
    h_conv11 = block(h_conv10, 16, phase_train)

    W_convOut = weight_variable([1, 1, 16, 1])
    b_convOut = bias_variable([1])

    h_convOut = tf.nn.relu(conv2d(h_conv11, W_convOut) + b_convOut)
    y_out = h_convOut

    loss = tf.nn.l2_loss(tf.sub(y_out, y))
    return loss, y_out

def conv_block(x, size):
    W1 = weight_variable([3, 3, size, size])
    b1 = bias_variable([size])
    #W2 = weight_variable([3, 1, size, size])
    #b2 = bias_variable([size])
    return conv2d(x, W1) + b1 #conv2d(conv2d(x, W1) + b1, W2) + b2

def res_block(x, size, phase_train):
    inp = x
    inp = tf.nn.relu(batch_norm(conv_block(inp, size), size, phase_train))
    inp = batch_norm(conv_block(inp, size), size, phase_train)

    return tf.nn.relu(inp + x)

def create_resnet(x, y, phase_train):
    x_image = x

    W_conv1 = weight_variable([3, 3, 1, 16])
    b_conv1 = bias_variable([16])

    h = tf.nn.relu(batch_norm(conv2d(x_image, W_conv1) + b_conv1, 16, phase_train))

    for i in range(7):
        h = res_block(h, 16, phase_train)

    W_convOut = weight_variable([1, 1, 16, 1])
    b_convOut = bias_variable([1])

    h_convOut = tf.nn.relu(conv2d(h, W_convOut) + b_convOut)
    y_out = tf.minimum(h_convOut, tf.constant(1, dtype='float32'))

    loss = tf.nn.l2_loss(y - y_out)
    return loss, y_out

def create_preview_net(x, y, phase_train):
    x_image = x

    x_preview = tf.image.resize_images(x, [32, 32])

    W_conv_p = weight_variable([3, 3, 1, 16])
    b_conv_p = bias_variable([16])

    h = tf.nn.relu(batch_norm(conv2d(x_preview, W_conv_p) + b_conv_p, 16, phase_train))

    for i in range(7):
        h = res_block(h, 16, phase_train)

    W_conv_p = weight_variable([1, 1, 16, 4])
    b_conv_p = bias_variable([4])

    h = tf.nn.relu(conv2d(h, W_conv_p) + b_conv_p)
    h1 = tf.image.resize_images(h, [x.get_shape()[1], x.get_shape()[2]])

    W_conv1 = weight_variable([3, 3, 1, 12])
    b_conv1 = bias_variable([12])

    h = tf.nn.relu(batch_norm(conv2d(x_image, W_conv1) + b_conv1, 12, phase_train))

    h = tf.concat([h, h1], 3)

    for i in range(3):
        h = res_block(h, 16, phase_train)

    W_convOut = weight_variable([1, 1, 16, 1])
    b_convOut = bias_variable([1])

    h_convOut = tf.nn.relu(conv2d(h, W_convOut) + b_convOut)
    y_out = tf.minimum(h_convOut, tf.constant(1, dtype='float32'))

    loss = tf.nn.l2_loss(y - y_out)
    return loss, y_out

def cascade_block(x, channels, out_channels, depth, phase_train):
    h = x
    for i in range(depth):
        h = res_block(h, channels, phase_train)

    W_conv_p = weight_variable([1, 1, channels, out_channels])
    b_conv_p = bias_variable([out_channels])

    h = tf.nn.relu(conv2d(h, W_conv_p) + b_conv_p)
    return h

def resize(x, channels, size, phase_train):
    x_preview = tf.image.resize_images(x, [size, size])

    W_conv_p = weight_variable([3, 3, 1, channels])
    b_conv_p = bias_variable([channels])

    h = tf.nn.relu(batch_norm(conv2d(x_preview, W_conv_p) + b_conv_p, channels, phase_train))
    return h

def create_cascade_net(x, y, phase_train):
    x_image = batch_norm(x, 1, phase_train)

    x_preview = resize(x_image, 16, 16, phase_train)
    h = x_preview
    h = cascade_block(h, 16, 8, 4, phase_train)

    h = tf.image.resize_images(h, [32, 32])
    x_preview = resize(x_image, 8, 32, phase_train)
    h = tf.concat([h, x_preview], 3)
    h = cascade_block(h, 16, 8, 4, phase_train)

    h = tf.image.resize_images(h, [64, 64])
    x_preview = resize(x_image, 8, 64, phase_train)
    h = tf.concat([h, x_preview], 3)
    h = cascade_block(h, 16, 8, 3, phase_train)

    h = tf.image.resize_images(h, [128, 128])
    x_preview = resize(x_image, 8, 128, phase_train)
    h = tf.concat([h, x_preview], 3)
    h = cascade_block(h, 16, 8, 2, phase_train)

    h = tf.image.resize_images(h, [x.get_shape()[1], x.get_shape()[1]])
    x_preview = resize(x_image, 8, x.get_shape()[1], phase_train)
    h = tf.concat([h, x_preview], 3)
    h = cascade_block(h, 16, 1, 2, phase_train)

    y_out = tf.minimum(h, tf.constant(1, dtype='float32'))

    loss = tf.nn.l2_loss(y - y_out)
    return loss, y_out
    
    
def create_cascade_net(x, y, phase_train):
    x_image = batch_norm(x, 1, phase_train)

    x_preview = resize(x_image, 16, 16, phase_train)
    h = x_preview
    h = cascade_block(h, 16, 8, 4, phase_train)

    h = tensorflow.image.resize_images(h, [32, 32])
    x_preview = resize(x_image, 8, 32, phase_train)
    h = tensorflow.concat([h, x_preview], 3)
    h = cascade_block(h, 16, 8, 4, phase_train)

    h = tensorflow.image.resize_images(h, [64, 64])
    x_preview = resize(x_image, 8, 64, phase_train)
    h = tensorflow.concat([h, x_preview], 3)
    h = cascade_block(h, 16, 8, 3, phase_train)

    h = tensorflow.image.resize_images(h, [128, 128])
    x_preview = resize(x_image, 8, 128, phase_train)
    h = tensorflow.concat([h, x_preview], 3)
    h = cascade_block(h, 16, 8, 2, phase_train)

    h = tensorflow.image.resize_images(h, [x.get_shape()[1], x.get_shape()[1]])
    x_preview = resize(x_image, 8, x.get_shape()[1], phase_train)
    h = tensorflow.concat([h, x_preview], 3)
    h = cascade_block(h, 16, 1, 2, phase_train)

    y_out = tensorflow.minimum(h, tensorflow.constant(1, dtype='float32'))

    loss = tensorflow.nn.l2_loss(y - y_out)
    return loss, y_out
    
    
class Cropper(object):
    def __init__(self):
        batch_size = 1
        image_size = 256

        self.x = tensorflow.placeholder(tensorflow.float32, shape=[batch_size, image_size, image_size, 1])
        self.y = tensorflow.placeholder(tensorflow.float32, shape=[batch_size, image_size, image_size, 1])
        self.phase_train = tensorflow.placeholder(tensorflow.bool, name='phase_train')
        self.loss, self.y_out = create_cascade_net(self.x, self.y, self.phase_train)
        self.sess = tensorflow.InteractiveSession()
        self.saver = tensorflow.train.Saver()
        self.saver.restore(self.sess, "checkpoints/cropnet.ckpt")

    def __del__(self):
        self.sess.close()

    def eval(self, batch, map_batch):
        with self.sess.as_default():
            res = self.y_out.eval(feed_dict={self.x: batch, self.y: map_batch, self.phase_train: False})
        return res
        
def collect_batch(image_size, data_dir):
    train_dir = data_dir + 'train/'
    test_dir = data_dir + 'test/'

    cropper = Cropper()

    print("Colecting train lungs")
    trainX, trainY = collect_lungs(train_dir, image_size, cropper)

    print("Colecting test lungs")
    testX, testY = collect_lungs(test_dir, image_size, cropper)

    del cropper

    return trainX, trainY, testX, testY
    
    
def create_dataset(image_size, data_dir, ):
    print("Creating dataset")
    X, Y, testX, testY = collect_batch(image_size, data_dir)

    # Reshape for tensorflow???
    # X = X.reshape([-1, image_size, image_size, 1])
    # testX = testX.reshape([-1, image_size, image_size, 1])

    # Create dataset
    h5f = h5py.File(data_dir + 'full_data_set.h5', 'w')
    h5f.create_dataset('X', data=X)
    h5f.create_dataset('Y', data=Y)
    h5f.create_dataset('testX', data=testX)
    h5f.create_dataset('testY', data=testY)
    h5f.close()


def load_dataset(data_dir):
    # Get dataset
    h5f = h5py.File(data_dir + 'full_data_set.h5', 'r')
    X = h5f['X']
    Y = h5f['Y']
    X_test = h5f['testX']
    Y_test = h5f['testY']
    return X, Y, X_test, Y_test
    



def create_alexnet(input_size):
    network = tflearn.input_data(shape=[None, input_size, input_size, 1])
    network = tflearn.conv_2d(network, 48, 11, strides=4, activation='relu')
    network = tflearn.max_pool_2d(network, 3, strides=2)
    network = tflearn.local_response_normalization(network)
    network = tflearn.conv_2d(network, 128, 5, activation='relu')
    network = tflearn.max_pool_2d(network, 3, strides=2)
    network = tflearn.local_response_normalization(network)
    network = tflearn.conv_2d(network, 192, 3, activation='relu')
    network = tflearn.conv_2d(network, 192, 3, activation='relu')
    network = tflearn.conv_2d(network, 128, 3, activation='relu')
    network = tflearn.max_pool_2d(network, 3, strides=2)
    network = tflearn.local_response_normalization(network)
    network = tflearn.fully_connected(network, 2048, activation='tanh')
    network = tflearn.dropout(network, 0.5)
    network = tflearn.fully_connected(network, 2048, activation='tanh')
    network = tflearn.dropout(network, 0.5)
    network = tflearn.fully_connected(network, 2, activation='softmax')
    network = tflearn.regression(network, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    return network
    
                         
def train_alexnet(network, X, Y, X_test, Y_test):
    # Training
    model = tflearn.DNN(network, checkpoint_path='alexnet_flug.ckpt',
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='tensorboard_log/')
    model.fit(X, Y, n_epoch=1000, validation_set=(X_test, Y_test), shuffle=True,
              show_metric=True, batch_size=1, snapshot_step=80,
              snapshot_epoch=True, run_id='alexnet_flug')


# some image size
input_size = 1234
data_dir = 'data/'

# file_list = os.listdir(data_dir)
# image_dir = "images/"
    
# for file in file_list:
#     image = init_input_by_name(file, folder=data_dir)
#     # plt.imsave(image_dir + file[:-4] + '_2.png', image)
#     cropper = Cropper()
#     image = crop_lungs(image, cropper)
#     original_size_res = get_resized_image(image, 2340)
#     plt.imsave(image_dir + file[:-4] + '2.png', original_size_res)
#     plt.imsave(image_dir + file[:-4] + '.png', image)

create_dataset(input_size, data_dir)
X, Y, X_test, Y_test = load_dataset(data_dir)

network = create_alexnet(input_size)
# train_alexnet(network, X, Y, X_test, Y_test,
#                              checkpoint_file='checkpoints/inception_resnet_flug.ckpt-989')

train_alexnet(network, X, Y, X_test, Y_test)
