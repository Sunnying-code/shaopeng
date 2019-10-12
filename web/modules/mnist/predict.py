# coding=utf-8

import tensorflow as tf
import numpy as np
from PIL import Image
import tempfile

from tensorflow.examples.tutorials.mnist import input_data

# from src.example.mnist_deep import deepnn
# from src.config import *

from web.modules.mnist.mnist_deep import deepnn
from web.config import *

FLAGS = '/tmp/tensorflow/mnist/input_data'

class Predict(object):
    def __init__(self):
        # self.net = Network()
        # Create the model
        self.x = tf.placeholder(tf.float32, [None, 784])

        # Define loss and optimizer
        self.y_ = tf.placeholder(tf.float32, [None, 10])
        self.y_conv, self.keep_prob = deepnn(self.x)
    #     self.sess = tf.Session()
    #     self.sess.run(tf.global_variables_initializer())
    #     # self.restore()  # 加载模型到sess中
    #
    # def restore(self):
    #     # saver = tf.train.Saver()
    #     saver = tf.train.import_meta_graph(model_path+'.meta')
    #     ckpt = tf.train.get_checkpoint_state(CKPT_DIR)
    #     if ckpt and ckpt.model_checkpoint_path:
    #         saver.restore(self.sess, ckpt.model_checkpoint_path)
    #     else:
    #         raise FileNotFoundError("未保存任何模型")


    def test(self, nm=None):
        # mnist = input_data.read_data_sets(FLAGS, one_hot=True)
        # tf.reset_default_graph()
        # Create the model
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

        # graph_location = tempfile.mkdtemp()
        # print('Saving graph to: %s' % graph_location)
        # train_writer = tf.summary.FileWriter(graph_location)
        # train_writer.add_graph(tf.get_default_graph())

        saver = tf.train.Saver()
        # saver = tf.train.import_meta_graph(ckpt_path+".meta")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # batch = mnist.train.next_batch(1)
            # print(y_conv.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}))
            # for i in range(20000):
            #     batch = mnist.train.next_batch(50)
            #     if i % 100 == 0:
            #         train_accuracy = accuracy.eval(feed_dict={
            #             x: batch[0], y_: batch[1], keep_prob: 1.0})
            #         print('step %d, training accuracy %g' % (i, train_accuracy))
            #     train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            # module_file = tf.train.get_checkpoint_state(ckpt_path)
            # if module_file is not None:s
            #     saver.restore(sess, ckpt_path)
            # saver.restore(sess, ckpt_path)
            # print(ckpt_path)
            saver.restore(sess, ckpt_path)
            # saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(ckpt_path)))
            # self.restore()
            # saver = tf.train.import_meta_graph(model_path + '.meta')
            # ckpt = tf.train.get_checkpoint_state(CKPT_DIR)
            # if ckpt and ckpt.model_checkpoint_path:
            #     saver.restore(sess, ckpt.model_checkpoint_path)
            # saver.save(sess, "./model/me")
            # print('test accuracy %g' % accuracy.eval(feed_dict={
            #     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
            # res = sess.run(accuracy, feed_dict={
            #     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
            # test_output = sess.run(y_conv, {x: mnist.test.images[:20]})
            # inferred_y = np.argmax(test_output, 1)
            # print(test_output)
            # print(inferred_y)
            # print(mnist.test.images[0])
            # print(type(mnist.test.images[0]))
            # print(mnist.test.labels[0])
            # test_output = sess.run(y_conv, feed_dict={
            #     x: mnist.test.images[:10], y_: mnist.test.labels[:10], keep_prob: 1})
            # inferred_y = np.argmax(test_output, 1)
            # print(inferred_y, '推测的数字')  # 推测的数字
            # print(np.argmax(mnist.test.labels[:10], 1), '真实的数字')  # 真实的数字


            test_output = sess.run(y_conv, feed_dict={
                x: nm, keep_prob: 1})
            inferred_y = np.argmax(test_output, 1)
            print(inferred_y, '推测的数字')  # 推测的数字
            return inferred_y
            # print(np.argmax(mnist.test.labels[:10], 1), '真实的数字')  # 真实的数字


    def test2(self):
        mnist = input_data.read_data_sets(FLAGS, one_hot=True)
        # Create the model
        x = tf.placeholder(tf.float32, [None, 784])

        # Define loss and optimizer
        y_ = tf.placeholder(tf.float32, [None, 10])

        # Build the graph for the deep net
        y_conv, keep_prob = deepnn(x)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print(model_path)
            # saver = tf.train.import_meta_graph(model_path + '.meta')

            module_file = tf.train.latest_checkpoint(model_path)
            if module_file is not None:
                saver.restore(sess, module_file)

            test_output = self.sess.run(y_conv, feed_dict={
                x: mnist.test.images[:10], y_: mnist.test.labels[:10], keep_prob: 1})
            inferred_y = np.argmax(test_output, 1)
            print(inferred_y, '推测的数字')  # 推测的数字




    # def predict(self, image_path):
    #     # 读图片并转为黑白的
    #     print(image_path)
    #
    #     img = Image.open(image_path)
    #     # img = Image.open(image_path).convert('L')
    #     # flatten_img = np.reshape(img, 784)
    #     # Create the model
    #     x_ = tf.placeholder(tf.float32, [None, 784])
    #
    #     # Define loss and optimizer
    #     y_ = tf.placeholder(tf.float32, [None, 10])
    #
    #     # x = np.array([1 - flatten_img])
    #     x = np.array(img)
    #     # y = self.sess.run(self.net.y, feed_dict={self.net.x: x})
    #     y = self.sess.run(self.y_conv, feed_dict={x_: x, self.keep_prob: 0.5})
    #     print(111)
    #
    #
    #     # 因为x只传入了一张图片，取y[0]即可
    #     # np.argmax()取得独热编码最大值的下标，即代表的数字
    #     print(image_path)
    #     print('        -> Predict digit', np.argmax(y[0]))


def getTestPicArray(filename):
    im = Image.open(filename)
    x_s = 28
    y_s = 28
    out = im.resize((x_s, y_s), Image.ANTIALIAS)

    im_arr = np.array(out.convert('L'))

    num0 = 0
    num255 = 0
    threshold = 100

    for x in range(x_s):
        for y in range(y_s):
            if im_arr[x][y] > threshold:
                num255 = num255 + 1
            else:
                num0 = num0 + 1

    if (num255 > num0):
        print("convert!")
        for x in range(x_s):
            for y in range(y_s):
                im_arr[x][y] = 255 - im_arr[x][y]
                if (im_arr[x][y] < threshold):  im_arr[x][y] = 0
            # if(im_arr[x][y] > threshold) : im_arr[x][y] = 0
            # else : im_arr[x][y] = 255
            # if(im_arr[x][y] < threshold): im_arr[x][y] = im_arr[x][y] - im_arr[x][y] / 2

    out = Image.fromarray(np.uint8(im_arr))
    # out.save(filename.split('/')[0] + '/28pix/' + filename.split('/')[1])
    # print im_arr
    nm = im_arr.reshape((1, 784))

    nm = nm.astype(np.float32)
    nm = np.multiply(nm, 1.0 / 255.0)

    return nm

if __name__ == '__main__':
    import os
    print(CKPT_DIR)
    # image_path = os.path.join(images_dir, "0", "7.bmp")
    image_path = os.path.join(test_images_dir, "6.jpg")

    nm = getTestPicArray(image_path)
    predict = Predict()
    predict.test(nm)


    # predict.predict(image_path)
    # img = Image.open(image_path)
    # x = np.array(img)
    # deepnn(x)
