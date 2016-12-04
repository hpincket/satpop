"""
mnist_cnn.py

Implement a Convolutional Neural Network for the MNIST classification task. Consists of two
Convolution + Max Pooling layers with ReLU Activation, plus 2 Fully-Connected Layers.
"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


class MnistCNN():
    def __init__(self, im_size, num_classes, patch_size=5, conv1_channel=32, conv2_channel=64,
                 hidden_size=1024, learning_rate=1e-4):
        """
        Initialize the Convolutional Neural Network (CNN) for the MNIST Task, with the necessary
        parameters.

        :param im_size: Width (or Height) of square MNIST image (in pixels).
        :param num_classes: Number of output classes.
        :param patch_size: Patch size, for convolution.
        :param conv1_channel: Number of features to compute for first convolution layer.
        :param conv2_channel: Number of features to compute for second convolution layer.
        :param hidden_size: Size of the fully-connected hidden layer.
        :param learning_rate: Learning rate for the Adam Optimizer.
        """
        self.im_size, self.num_classes, self.patch_size = im_size, num_classes, patch_size
        self.conv1_channel, self.conv2_channel = conv1_channel, conv2_channel
        self.after_pool_size, self.hidden_size = self.im_size / (2 * 2), hidden_size
        self.learning_rate = learning_rate

        # Initialize Placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, self.im_size * self.im_size])
        self.y = tf.placeholder(tf.float32, shape=[None, self.num_classes])
        self.keep_prob = tf.placeholder(tf.float32)

        # Reshape Image to be of shape [batch, width, height, channel]
        self.x_image = tf.reshape(self.x, [-1, self.im_size, self.im_size, 1])

        # Instantiate Weights
        self.instantiate_weights()

        # Build Inference Graph
        self.logits = self.inference()

        # Build Loss Computation
        self.loss_val = self.loss()

        # Build Training Operation
        self.train_op = self.train()

        # Create operations for computing the accuracy
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def instantiate_weights(self):
        """
        Instantiate the network parameters, for each of the layers.
        """
        self.W_conv1 = self.weight_variable([self.patch_size, self.patch_size, 1,
                                             self.conv1_channel], "W_conv1")
        self.b_conv1 = self.weight_variable([self.conv1_channel], "B_conv1")

        self.W_conv2 = self.weight_variable([self.patch_size, self.patch_size, self.conv1_channel,
                                             self.conv2_channel], "W_conv2")
        self.b_conv2 = self.bias_variable([self.conv2_channel], "B_conv2")

        self.W_fc1 = self.weight_variable([self.after_pool_size * self.after_pool_size *
                                           self.conv2_channel, self.hidden_size], "W_fc1")
        self.b_fc1 = self.bias_variable([self.hidden_size], "B_fc1")

        self.W_fc2 = self.weight_variable([self.hidden_size, self.num_classes], "W_fc2")
        self.b_fc2 = self.bias_variable([self.num_classes], "B_fc2")

    def inference(self):
        """
        Build the inference computation graph, performing the convolution, max pooling, and feed
        forward operations.

        :return Tensor corresponding to the output after the softmax layer.
        """
        # Convolution and Pooling 1
        h_conv1 = tf.nn.relu(self.conv2d(self.x_image, self.W_conv1) + self.b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        # Convolution and Pooling 2
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, self.W_conv2) + self.b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        # Fully Connected (Hidden) Layer
        h_pool2_flat = tf.reshape(h_pool2, [-1, self.after_pool_size * self.after_pool_size *
                                            self.conv2_channel])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.W_fc1) + self.b_fc1)

        # Dropout self.keep_prob fraction of the units
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # Fully Connected (Logits) Layer
        logits = tf.matmul(h_fc1_drop, self.W_fc2) + self.b_fc2
        return logits

    def loss(self):
        """
        Build the cross-entropy loss operation, using the softmax output from the inference graph.

        :return: Scalar corresponding to cross-entropy loss of the given batch.
        """
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, self.y))

    def train(self):
        """
        Build the training operation, consisting of initializing an optimizer, and minimizing
        the cross-entropy loss.

        :return: Operation consisting of a single backpropagation pass.
        """
        return tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_val)

    @staticmethod
    def weight_variable(shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    @staticmethod
    def bias_variable(shape, name):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    @staticmethod
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Main Training Block
if __name__ == "__main__":
    # Read in data, write gzip files to "data/" directory
    mnist_data = input_data.read_data_sets("data/", one_hot=True)
    img_size, num_class, batch_size = 512, 10, 50

    # Start Tensorflow Session
    with tf.Session() as sess:
        mnist_cnn = MnistCNN(img_size, num_class)
        sess.run(tf.initialize_all_variables())

        # Start Training Loop
        for i in range(2000):
            batch = mnist_data.train.next_batch(batch_size)
            if i % 100 == 0:
                train_accuracy = mnist_cnn.accuracy.eval(feed_dict={mnist_cnn.x: batch[0],
                                                                    mnist_cnn.y: batch[1],
                                                                    mnist_cnn.keep_prob: 1.0})
                print "Step %d, Training Accuracy %g" % (i, train_accuracy)
            mnist_cnn.train_op.run(feed_dict={mnist_cnn.x: batch[0], mnist_cnn.y: batch[1],
                                              mnist_cnn.keep_prob: 0.5})

        # Evaluate Test Accuracy
        print "Test Accuracy %g" % mnist_cnn.accuracy.eval(feed_dict={
            mnist_cnn.x: mnist_data.test.images, mnist_cnn.y: mnist_data.test.labels,
            mnist_cnn.keep_prob: 1.0})