#!/usr/bin/env python
# -*- coding: utf8 -*-

# build vgg net example

import numpy as np
import tensorflow as tf

class VGG:

    def __init__(self, input_shape, output_shape):
        assert isinstance(input_shape, tuple)
        assert isinstance(output_shape, tuple)

        self.input_shape = input_shape
        self.output_shape = output_shape

    def __weight_variable(self, shape, name="weight"):
        initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=0.1)
        return tf.Variable(initial, name=name)

    def __bias_variable(self, shape, name="biases"):
        initial = tf.constant(0.1, dtype=tf.float32, shape=shape)
        return tf.Variable(initial, name=name)

    def __conv2d(self, input, w, name="conv2d"):
        return tf.nn.conv2d(input, filter=w, strides=[1,1,1,1], padding="SAME", name=name)
        
    def __max_pool_2x2(self, input, name="max_pool"):
        return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
    
    def __fc(self, input, w, b):
        return tf.matmul(input, w) + b

    def build_vgg_19(self):

        self.x = tf.placeholder(tf.float32, shape=[None] + list(self.input_shape), name="X")
        self.y = tf.placeholder(tf.int64, shape=[None] + list(self.output_shape), name="Y")

        # conv 1
        with tf.name_scope('conv1_1') as scope:
            kernel = self.__weight_variable([3, 3, 3, 64])
            biases = self.__bias_variable([64])
            output_conv1_1 = tf.nn.relu(self.__conv2d(self.x, kernel) + biases, name=scope)

        with tf.name_scope('conv1_2') as scope:
            kernel = self.__weight_variable([3, 3, 64, 64])
            biases = self.__bias_variable([64])
            output_conv1_2 = tf.nn.relu(self.__conv2d(output_conv1_1, kernel) + biases, name=scope)
        
        pool_1 = self.__max_pool_2x2(output_conv1_2, name="conv1_max_pool")

        # conv 2
        with tf.name_scope('conv2_1') as scope:
            kernel = self.__weight_variable([3, 3, 64, 128])
            biases = self.__bias_variable([128])
            output_conv2_1 = tf.nn.relu(self.__conv2d(pool_1, kernel) + biases, name=scope)

        with tf.name_scope('conv2_2') as scope:
            kernel = self.__weight_variable([3, 3, 128, 128])
            biases = self.__bias_variable([128])
            output_conv2_2 = tf.nn.relu(self.__conv2d(output_conv2_1, kernel) + biases, name=scope)
        
        pool_2 = self.__max_pool_2x2(output_conv2_2, name="conv2_max_pool")

        # conv 3
        with tf.name_scope('conv3_1') as scope:
            kernel = self.__weight_variable([3, 3, 128, 256])
            biases = self.__bias_variable([256])
            output_conv3_1 = tf.nn.relu(self.__conv2d(pool_2, kernel) + biases, name=scope)

        with tf.name_scope('conv3_2') as scope:
            kernel = self.__weight_variable([3, 3, 256, 256])
            biases = self.__bias_variable([256])
            output_conv3_2 = tf.nn.relu(self.__conv2d(output_conv3_1, kernel) + biases, name=scope)

        with tf.name_scope('conv3_3') as scope:
            kernel = self.__weight_variable([3, 3, 256, 256])
            biases = self.__bias_variable([256])
            output_conv3_3 = tf.nn.relu(self.__conv2d(output_conv3_2, kernel) + biases, name=scope)

        with tf.name_scope('conv3_4') as scope:
            kernel = self.__weight_variable([3, 3, 256, 256])
            biases = self.__bias_variable([256])
            output_conv3_4 = tf.nn.relu(self.__conv2d(output_conv3_3, kernel) + biases, name=scope)

        pool_3 = self.__max_pool_2x2(output_conv3_4, name="conv3_max_pool")

        # conv 4
        with tf.name_scope('conv4_1') as scope:
            kernel = self.__weight_variable([3, 3, 256, 512])
            biases =self.__bias_variable([512])
            output_conv4_1 = tf.nn.relu(self.__conv2d(pool_3, kernel) + biases, name=scope)

        with tf.name_scope('conv4_2') as scope:
            kernel = self.__weight_variable([3, 3, 512, 512])
            biases = self.__bias_variable([512])
            output_conv4_2 = tf.nn.relu(self.__conv2d(output_conv4_1, kernel) + biases, name=scope)

        with tf.name_scope('conv4_3') as scope:
            kernel = self.__weight_variable([3, 3, 512, 512])
            biases = self.__bias_variable([512])
            output_conv4_3 = tf.nn.relu(self.__conv2d(output_conv4_2, kernel) + biases, name=scope)

        with tf.name_scope('conv4_4') as scope:
            kernel = self.__weight_variable([3, 3, 512, 512])
            biases = self.__bias_variable([512])
            output_conv4_4 = tf.nn.relu(self.__conv2d(output_conv4_3, kernel) + biases, name=scope)

        pool_4 = self.__max_pool_2x2(output_conv4_4, name="conv4_max_pool")

        # conv 5
        with tf.name_scope('conv5_1') as scope:
            kernel = self.__weight_variable([3, 3, 512, 512])
            biases = self.__bias_variable([512])
            output_conv5_1 = tf.nn.relu(self.__conv2d(pool_4, kernel) + biases, name=scope)

        with tf.name_scope('conv5_2') as scope:
            kernel = self.__weight_variable([3, 3, 512, 512])
            biases = self.__bias_variable([512])
            output_conv5_2 = tf.nn.relu(self.__conv2d(output_conv5_1, kernel) + biases, name=scope)

        with tf.name_scope('conv5_3') as scope:
            kernel = self.__weight_variable([3, 3, 512, 512])
            biases = self.__bias_variable([512])
            output_conv5_3 = tf.nn.relu(self.__conv2d(output_conv5_2, kernel) + biases, name=scope)

        with tf.name_scope('conv5_4') as scope:
            kernel = self.__weight_variable([3, 3, 512, 512])
            biases = self.__bias_variable([512])
            output_conv5_4 = tf.nn.relu(self.__conv2d(output_conv5_3, kernel) + biases, name=scope)

        pool_5 = self.__max_pool_2x2(output_conv5_4, name="conv5_max_pool")

        # fc 1
        with tf.name_scope('fc_1') as scope:
            shape = int(np.prod(pool_5.get_shape()[1:]))
            w = self.__weight_variable([shape, 4096])
            b = self.__bias_variable([4096])
            pool_5_flat = tf.reshape(pool_5, [-1, shape])
            fc_1 = tf.nn.relu(self.__fc(pool_5_flat, w, b), name=scope)

        # fc 2
        with tf.name_scope('fc_2') as scope:
            w = self.__weight_variable([4096, 4096])
            b = self.__bias_variable([4096])
            fc_2 = tf.nn.relu(self.__fc(fc_1, w, b), name=scope)

        # fc 3
        with tf.name_scope('fc_3') as scope:
            w = self.__weight_variable([4096]+list(self.output_shape))
            b = self.__bias_variable(list(self.output_shape))
            fc_3 = tf.nn.relu(self.__fc(fc_2, w, b), name=scope)

        self.output = tf.nn.softmax(fc_3, name="output")

        return {"x": self.x, "y": self.y, "output": self.output}


# test
if __name__ == '__main__':
    # 输入为 (300,300,3) 的图片，输出为 10 的分类
    n = VGG((300,300,3), (10,))
    ret = n.build_vgg_19()
    print (ret)
