#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np

from web.modules.mnist.mnist_deep import deepnn
from web.config import *

from web.modules.mnist.predict import getTestPicArray

import tempfile
from tensorflow.examples.tutorials.mnist import input_data

def trans():
    mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)
    x = tf.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Build the graph for the deep net
    y_conv, keep_prob = deepnn(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    saver = tf.train.Saver()
    # saver = tf.train.import_meta_graph(ckpt_path+".meta")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        batch = mnist.train.next_batch(1)
        print(y_conv.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}))
        for i in range(20000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        saver.save(sess, ckpt_path)

def predict(image_path):
    nm = getTestPicArray(image_path)

    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    y_conv, keep_prob = deepnn(x)

    saver = tf.train.Saver()
    # saver = tf.train.import_meta_graph(ckpt_path+".meta")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt_path)
        test_output = sess.run(y_conv, feed_dict={
            x: nm, keep_prob: 1})
        inferred_y = np.argmax(test_output, 1)
        print(inferred_y, '推测的数字')  # 推测的数字
        return inferred_y


if __name__ == '__main__':
    image_path = os.path.join(test_images_dir, "6.jpg")

    result = predict(image_path)
    print(result)
    # trans()
