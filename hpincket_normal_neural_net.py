import numpy as np
import tensorflow as tf

import constants as C
from batch import ParallelSatPopBatch

# ########################################

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * 3

OPTIONS = 5

batchsize = 100
learning_rate = 0.05

img = tf.placeholder(tf.float32, [None, IMAGE_SIZE])
labels = tf.placeholder(tf.float32, shape=[None, OPTIONS])
W = tf.Variable(tf.zeros([IMAGE_SIZE, OPTIONS]))
B = tf.Variable(tf.zeros([1, OPTIONS]))
P = tf.nn.softmax(tf.matmul(img, W) + B)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(P), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# ########################################

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

pspb = ParallelSatPopBatch(C.SATPOP_MAIN_DATA_FILE, C.SATPOP_IMAGE_FOLDER, batch_size=batchsize)
with pspb as spb:
    for i, batch in enumerate(spb):
        print(batch)
        if i > 4:
            break
        batching_img, batching_lab = batch
        sess.run(train_step, feed_dict={img: batching_img,
                                        labels: batching_lab})

# ########################################

correct_prediction = tf.equal(tf.argmax(P, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

ptest_spb = ParallelSatPopBatch(C.SATPOP_MAIN_DATA_FILE, C.SATPOP_IMAGE_FOLDER, batch_size=batchsize, random=True)
test_imgs = np.array((0, 2))
test_labs = np.array((0, 2))
all_test_accuracy = []
with ptest_spb as test_spb:
    for i, batch in enumerate(test_spb):
        print("iteration {}".format(i))
        if i > 2:
            break
        batching_img, batching_lab = batch
        new_batch_accuracy = sess.run(accuracy, feed_dict={img: batching_img,
                                                           labels: batching_lab})
        all_test_accuracy.append(new_batch_accuracy)

print("Done with testing")
print(all_test_accuracy)
print(np.mean(all_test_accuracy))
