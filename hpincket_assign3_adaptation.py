import numpy as np
import tensorflow as tf

import constants as C
from batch import BucketLabelTransformer, ParallelSatPopBatch

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# transformer = BucketLabelTransformer(generate_even_divisions(5))

transformer = BucketLabelTransformer([1.0, 100000])
batchsize = 40

OPTIONS = transformer.number_of_labels()

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')


def max_pool(pick, x):
    return tf.nn.max_pool(x, ksize=[1, pick, pick, 1], strides=[1, pick, pick, 1], padding='SAME')

sess = tf.Session()

x = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT, 3])
y_ = tf.placeholder(tf.float32, shape=[None, OPTIONS])

W_conv1 = weight_variable([3, 3, 3, 64])
b_conv1 = bias_variable([64])
# We have -1 in the shape to retain data size. We need this for unknown batchsize.
# x_image = tf.reshape(x, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1])

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
# Size: h_conv1 is [batchsize, 28, 28, 32]
# Size: h_conv1 is [batchsize, 512, 512, 32]
h_pool1 = max_pool(2, h_conv1)

# Size: h_pool1 is [batchsize, 14, 14, 32]
# Size: h_pool1 is [batchsize, 256, 256, 32]
# print(h_pool1.get_shape())
W_conv2 = weight_variable([3, 3, 64, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool(2, h_conv2)

# print(h_pool2.get_shape())

print(h_pool1.get_shape())
W_conv3 = weight_variable([3, 3, 64, 64])
b_conv3 = bias_variable([64])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool(2, h_conv3)

print(h_pool3.get_shape())

W_conv4 = weight_variable([3, 3, 64, 64])
b_conv4 = bias_variable([64])
h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
h_pool4 = max_pool(2, h_conv4)

quarter_size = IMAGE_WIDTH / (2 * 2 * 2 * 2)
# Densly connected layer
# Our features are now all 64 filters at all locations, just picture this as
# the original neural network from assign1.
NEW_SIZE = 1024
W_fc1 = weight_variable([quarter_size * quarter_size * 64, NEW_SIZE])
b_fc1 = bias_variable([NEW_SIZE])
h_pool2_flat = tf.reshape(h_pool4, [-1, quarter_size * quarter_size * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout
W_fc2 = weight_variable([NEW_SIZE, OPTIONS])
b_fc2 = bias_variable([OPTIONS])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# y = tf.nn.sigmoid(y_conv)
# weighted = tf.reduce_sum(tf.mul(normalized, transformer.bucket_maxes))
# weighted_error = tf.reduce_sum(tf.square(y_ - weighted))
# mse = tf.reduce_mean(tf.square(y - y_))

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
# train_step = tf.train.AdamOptimizer(1e-4).minimize(weighted_error)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
argmax_conv = tf.argmax(y_conv, 1)
argmax_ = tf.argmax(y_, 1)
correct_prediction = tf.equal(argmax_conv, argmax_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())


# print(weighted.get_shape())
# for i in range(2000):
# batch = mnist.train.next_batch(50)
# if i%100 == 0:
# train_accuracy = accuracy.eval(session=sess, feed_dict={
# x:batch[0], y_: batch[1], keep_prob: 1.0})
# print("step %d, training accuracy %g"%(i, train_accuracy))
# sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# print("test accuracy %g"%accuracy.eval(session=sess, feed_dict={
# x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

def test(n):
    ptest_spb = ParallelSatPopBatch(C.SATPOP_MAIN_DATA_FILE, C.SATPOP_IMAGE_FOLDER, batch_size=batchsize,
                                    image_dimension=3,
                                    random=True, label_transformer=transformer)
    all_test_accuracy = []
    with ptest_spb as test_spb:
        for i, batch in enumerate(test_spb):
            if i >= n:
                break
            batching_img, batching_lab = batch
            new_batch_accuracy, guess, correct = sess.run((accuracy, argmax_conv, argmax_), feed_dict={x: batching_img,
                                                                                                       y_: batching_lab,
                                                                                                       keep_prob: 1.0})
            for g, c in zip(guess, correct):
                if g != c:
                    print("{}\t{}".format(g, c))
            all_test_accuracy.append(new_batch_accuracy)
    print(np.mean(all_test_accuracy))

for i in range(2):
    pspb = ParallelSatPopBatch(C.SATPOP_MAIN_DATA_FILE, C.SATPOP_IMAGE_FOLDER, batch_size=batchsize,
                               label_transformer=transformer, image_dimension=3)
    with pspb as spb:
        for i, batch in enumerate(spb):
            if i > 60:
                break
            batching_img, batching_lab = batch
            if i % 2 == 0:
                train_accuracy, y_conv_res = sess.run(
                    (accuracy, y_conv), feed_dict={
                        x: batching_img,
                        y_: batching_lab,
                        keep_prob: 1.0
                    })

                # def normalize(v):
                # mmin = np.amin(v)
                # mmax = np.amax(v)
                # nv = []
                # for e in v:
                # nv.append(e - mmin)
                # nnv = []
                # for e in nv:
                # nnv.append(e / (mmax - mmin))
                # s = sum(nnv)
                # return [e / s for e in nnv]
                # for yc, l in zip(list(normalize(y_conv_res)), list(batching_lab)):
                # print(yc)
                # print(l)
                # print("--")


                # print(zip(list(y_conv_res), list(batching_lab)))
                # print(weighted_res)
                # print(weighted_error_res)
                print(train_accuracy)
            sess.run(train_step, feed_dict={x: batching_img,
                                            y_: batching_lab,
                                            keep_prob: 0.5})

print("Fin")
test(80)
