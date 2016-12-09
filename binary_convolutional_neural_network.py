#!/usr/bin/env python

import csv
import numpy as np
import sys
import tensorflow as tf

from collections import defaultdict
from os import listdir
from os.path import isfile, join
from scipy import misc

# Weight Initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Convolution and Pooling
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def max_pool_4x4(x):
    return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

def color_to_grayscale(rgb):
    return (float(rgb[0]) / 3 + float(rgb[1]) / 3 + float(rgb[2]) / 3) / 255

def cross_validation_slicing(values, test_slice_index, batch_size):
    test_values = values[test_slice_index * batch_size:(test_slice_index + 1) * batch_size]
    
    preceding_training_values = values[:test_slice_index * batch_size]
    following_training_values = values[(test_slice_index + 1) * batch_size:]
    
    training_values = np.concatenate((preceding_training_values, following_training_values), axis=0)
    
    return (training_values, test_values)

def main():
    image_directory = sys.argv[1]
    tsv_filepath = sys.argv[2]
    
    image_width = 512
    population_density_by_image = defaultdict(float)
    unpopulated_populated = list()
    unpopulated_populated_by_image = defaultdict(list)
    num_classes = 2
    num_batches = 5
    batch_size = 50
    
    y_0 = [1.0, 0.0] # unpopulated
    y_1 = [0.0, 1.0] # populated
    
    # Map image names to population densities
    with open(tsv_filepath, 'rb') as tsv_file:
        tsv_data = csv.reader(tsv_file, delimiter='\t')
        
        for row in tsv_data:            
            file_name = row[0] + '.png'
            population_density = float(row[3])
            # print population_density
            
            if population_density > 10.0:
                unpopulated_populated.append(y_0)
                unpopulated_populated_by_image[file_name] = y_0
            else:
                unpopulated_populated.append(y_1)
                unpopulated_populated_by_image[file_name] = y_1
            
            population_density_by_image[file_name] = population_density
    
    print('done with tsv file')
    
    # Load satellite images
    file_names = [f for f in listdir(image_directory) if isfile(join(image_directory, f))]
    
    grayscale_values_by_image = defaultdict(list)
    grayscale_values_in_sequence = list()
    unpopulated_populated_in_sequence = list()
    
    print('loading satellite images')
    count = 0
    
    for fn in file_names:
        count += 1
        
        if count > num_batches * batch_size:
            break
        
        if count % 10 == 0:
            print(count)
        
        image_array = misc.imread(join(image_directory, fn)) # 512 x 512 x 3
        grayscale_values = list()
        
        for i in range(image_width):
            for j in range(image_width):
                grayscale_values.append(color_to_grayscale(image_array[i][j]))
        
        grayscale_values_by_image[fn] = grayscale_values
        grayscale_values_in_sequence.append(grayscale_values)
        unpopulated_populated_in_sequence.append(unpopulated_populated_by_image[fn])
        # print fn
        # print population_density_by_image[fn]
        # print unpopulated_populated_by_image[fn]
    
    print len(grayscale_values_in_sequence)
    print len(unpopulated_populated_in_sequence)
    
    y_0_count = 0
    y_1_count = 0

    for i in unpopulated_populated_in_sequence:
        if i == y_0:
            y_0_count += 1
        elif i == y_1:
            y_1_count += 1
    
    print('y_0 count')
    print(y_0_count)
    print('y_1 count')
    print(y_1_count)
    
    # Start TensorFlow InteractiveSession
    sess = tf.InteractiveSession()
    
    # Placeholders
    x = tf.placeholder(tf.float32, shape=[None, image_width * image_width])
    y_ = tf.placeholder(tf.float32, shape=[None, num_classes])
    
    # First Convolutional Layer
    patch_size = 5
    W_conv1 = weight_variable([patch_size, patch_size, 1, 32])
    b_conv1 = bias_variable([32])
    
    x_image = tf.reshape(x, [-1, image_width, image_width, 1])
    
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_4x4(h_conv1)
    
    # Second Convolutional Layer
    W_conv2 = weight_variable([patch_size, patch_size, 32, 64])
    b_conv2 = bias_variable([64])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_4x4(h_conv2)
    
    # Densely Connected Layer
    hidden_size = 1024
    after_pool_width = image_width / (4 * 4)
    W_fc1 = weight_variable([after_pool_width * after_pool_width * 64, hidden_size])
    b_fc1 = bias_variable([hidden_size])
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, after_pool_width * after_pool_width * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    print('h_fc1_drop')
    print(h_fc1_drop.get_shape())
    
    # Readout Layer
    W_fc2 = weight_variable([hidden_size, num_classes])
    print('W_fc2')
    print(W_fc2.get_shape())
    b_fc2 = bias_variable([num_classes])
    
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    
    # Train and Evaluate the Model
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    print('y_conv')
    print(y_conv.get_shape())
    print('tf.argmax(y_conv, 1)')
    print(tf.argmax(y_conv, 1).get_shape())
    print('y_')
    print(y_.get_shape())
    print('tf.argmax(y_, 1)')
    print(tf.argmax(y_, 1).get_shape())
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.global_variables_initializer())
    
    print np.asarray(grayscale_values_in_sequence).shape
    print np.asarray(unpopulated_populated_in_sequence).shape
    
    images_array = np.asarray(grayscale_values_in_sequence)
    labels_array = np.asarray(unpopulated_populated_in_sequence)
    
    training_steps = num_batches - 1
    
    for test_slice_index in range(num_batches): # cross-validation
        print("fold: " + str(test_slice_index))
        
        (training_images, test_images) = cross_validation_slicing(images_array, test_slice_index, batch_size)
        (training_labels, test_labels) = cross_validation_slicing(labels_array, test_slice_index, batch_size)
        
        for i in range(training_steps):
            batch_start = i * batch_size
            batch_end = batch_start + batch_size
            train_accuracy = accuracy.eval(feed_dict={x: training_images[batch_start:batch_end], y_: training_labels[batch_start:batch_end], keep_prob: 1.0})

            print("step %d, training accuracy %g"%(i, train_accuracy))
            
            train_step.run(feed_dict={x: training_images[batch_start:batch_end], y_: training_labels[batch_start:batch_end], keep_prob: 0.5})
        
        test_results = []
        
        for i in range(batch_size):
            test_result = accuracy.eval(feed_dict={x: [test_images[i]], y_: [test_labels[i]], keep_prob: 1.0})
            test_results.append(test_result)
        
        test_accuracy = np.mean(test_results)
        
        print("test accuracy %g"%test_accuracy)

if __name__ == '__main__':
    main()