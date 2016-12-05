import numpy as np
import tensorflow as tf

import constants as C
from batch import BucketLabelTransformer, ParallelSatPopBatch
from metadata_utils import generate_even_divisions

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(pick, x):
    if pick == 2:
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    elif pick == 4:
        return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

def main():
    num_buckets = 3
    transformer = BucketLabelTransformer(generate_even_divisions(num_buckets))
    batchsize = 5

    OPTIONS = transformer.number_of_labels()

    IMAGE_HEIGHT = 512
    IMAGE_WIDTH = 512

    # Start TensorFlow InteractiveSession
    sess = tf.Session()
    
    # Placeholders
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT, 3])
    y_ = tf.placeholder(tf.float32, shape=[None, OPTIONS])

    # First Convolutional Layer
    patch_size = 5
    W_conv1 = weight_variable([patch_size, patch_size, 3, 32])
    b_conv1 = bias_variable([32])
    # We have -1 in the shape to retain data size. We need this for unknown batchsize.
    x_image = tf.reshape(x, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, 3])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # Size: h_conv1 is [batchsize, 28, 28, 32]
    # Size: h_conv1 is [batchsize, 512, 512, 32]
    h_pool1 = max_pool(4, h_conv1)
    print(h_pool1.get_shape())
    
    # Second Convolutional Layer
    # Size: h_pool1 is [batchsize, 14, 14, 32]
    # Size: h_pool1 is [batchsize, 256, 256, 32]
    W_conv2 = weight_variable([patch_size, patch_size, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool(4, h_conv2)

    print(h_pool2.get_shape())
    quarter_size = IMAGE_WIDTH / (4 * 4)
    # Densly connected layer
    # Our features are now all 64 filters at all locations, just picture this as
    # the original neural network from assign1.
    # hidden_size = 1024
    hidden_size = 128
    W_fc1 = weight_variable([quarter_size * quarter_size * 64, hidden_size])
    b_fc1 = bias_variable([hidden_size])
    h_pool2_flat = tf.reshape(h_pool2, [-1, quarter_size * quarter_size * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Readout
    W_fc2 = weight_variable([hidden_size, OPTIONS])
    b_fc2 = bias_variable([OPTIONS])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # normalized = tf.sigmoid(y_conv)
    # weighted = tf.reduce_sum(tf.mul(normalized, transformer.bucket_maxes))
    # weighted_error = tf.reduce_sum(tf.square(y_ - weighted))
    # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    # train_step = tf.train.AdamOptimizer(1e-4).minimize(weighted_error)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.global_variables_initializer())

    print(weighted.get_shape())
    # for i in range(2000):
    # batch = mnist.train.next_batch(50)
    # if i%100 == 0:
    # train_accuracy = accuracy.eval(session=sess, feed_dict={
    # x:batch[0], y_: batch[1], keep_prob: 1.0})
    # print("step %d, training accuracy %g"%(i, train_accuracy))
    # sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    # print("test accuracy %g"%accuracy.eval(session=sess, feed_dict={
    # x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


    pspb = ParallelSatPopBatch(C.SATPOP_MAIN_DATA_FILE, C.SATPOP_IMAGE_FOLDER, batch_size=batchsize, label_transformer=transformer, image_dimension=3)
    with pspb as spb:
        for i, batch in enumerate(spb):
            if i >= 80:
                break
            
            batching_img, batching_lab = batch
            
            if i % 2 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batching_img, y_: batching_lab, keep_prob: 1.0})
                
                print(train_accuracy)
            
            train_step.run(feed_dict={x: batching_img, y_: batching_lab, keep_prob: 0.5})

    print("TESTING")

    ptest_spb = ParallelSatPopBatch(C.SATPOP_MAIN_DATA_FILE, C.SATPOP_IMAGE_FOLDER, batch_size=batchsize, image_dimension=3, random=True, label_transformer=transformer)

    all_test_accuracy = []

    with ptest_spb as test_spb:
        for i, batch in enumerate(test_spb):
            if i >= 20:
                break
            
            print("iteration {}".format(i))
            
            batching_img, batching_lab = batch
            new_batch_accuracy = sess.run(accuracy, feed_dict={x: batching_img,
                                                               y_: batching_lab,
                                                               keep_prob: 1.0})
            all_test_accuracy.append(new_batch_accuracy)

    print("Done with testing")
    print(all_test_accuracy)
    print(np.mean(all_test_accuracy))

if __name__ == '__main__':
    main()