#!/usr/bin/env python
# -*- coding: utf8 -*-

import os
import glob

import cv2
import numpy as np

import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util

width = 224
height = 224
channels = 3

# 生成数据集
def read_data():
    print "开始准备数据集"

    img_path = "./picture/"

    # ========================================================
    # 方法一： 少量图片数据可以直接读取到内存中，生成 Dataset
    # ========================================================
    cate   = [img_path + x for x in os.listdir(img_path) if os.path.isdir(img_path + x)]
    imgs   = []
    labels = []
    for idx, folder in enumerate(cate):
        for img in glob.glob(folder + '/*.jpg'):
            print('add image to list: %s' % (img))
            imgs.append(img)
            labels.append(idx)
    
    # 把数组转换成 Tensor
    filenames = tf.constant(imgs)
    labels = tf.constant(labels, dtype=tf.int64)
    
    # 定义 dataset
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    
    # 解析函数，功能是将 filename 对应的图片文件读进来，并缩放到统一的大小
    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.image.resize_images(image_decoded, [width, height])
        return image_resized, label

    # 批量处理数据，把 dataset 中的图片读取进来
    dataset = dataset.map(_parse_function)
    
    # 在每个 epoch 内将图片打乱组成大小为32的batch，并重复10次
    dataset = dataset.shuffle(buffer_size=1000).batch(10).repeat(20)
    print dataset.output_shapes

    print "结束准备数据集"
    return dataset
    
    # ==========================================================
    # 方法二： 大量数据（无法全部加载进内存）
    # ==========================================================
    # 待补充

# 生成网络模型 (VGG 19)
def build_network(dataset):
    print "开始构建模型"

    def weight_variable(shape, name="weight"):
        initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(shape, name="biases"):
        initial = tf.constant(0.1, dtype=tf.float32, shape=shape)
        return tf.Variable(initial, name=name)

    def conv2d(input, w, name="conv2d"):
        return tf.nn.conv2d(input, filter=w, strides=[1,1,1,1], padding="SAME", name=name)
        
    def max_pool_2x2(input, name="max_pool"):
        return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
    
    def fc(input, w, b):
        return tf.matmul(input, w) + b

    # vgg 19 layers
    iterator = dataset.make_one_shot_iterator()
    one_element = iterator.get_next()
    x = one_element[0]
    y = one_element[1]
    #x = tf.placeholder(tf.float32, shape=[None, width, height, channels], name="input")
    #y = tf.placeholder(tf.int64, shape=[None, 2], name="labels")
    print x
    print y
    
    # conv 1
    with tf.name_scope('conv1_1') as scope:
        kernel = weight_variable([3, 3, 3, 64])
        biases = bias_variable([64])
        output_conv1_1 = tf.nn.relu(conv2d(x, kernel) + biases, name=scope)

    with tf.name_scope('conv1_2') as scope:
        kernel = weight_variable([3, 3, 64, 64])
        biases = bias_variable([64])
        output_conv1_2 = tf.nn.relu(conv2d(output_conv1_1, kernel) + biases, name=scope)
    
    pool_1 = max_pool_2x2(output_conv1_2, name="conv1_max_pool")

    # conv 2
    with tf.name_scope('conv2_1') as scope:
        kernel = weight_variable([3, 3, 64, 128])
        biases = bias_variable([128])
        output_conv2_1 = tf.nn.relu(conv2d(pool_1, kernel) + biases, name=scope)

    with tf.name_scope('conv2_2') as scope:
        kernel = weight_variable([3, 3, 128, 128])
        biases = bias_variable([128])
        output_conv2_2 = tf.nn.relu(conv2d(output_conv2_1, kernel) + biases, name=scope)
    
    pool_2 = max_pool_2x2(output_conv2_2, name="conv2_max_pool")

    # conv 3
    with tf.name_scope('conv3_1') as scope:
        kernel = weight_variable([3, 3, 128, 256])
        biases = bias_variable([256])
        output_conv3_1 = tf.nn.relu(conv2d(pool_2, kernel) + biases, name=scope)

    with tf.name_scope('conv3_2') as scope:
        kernel = weight_variable([3, 3, 256, 256])
        biases = bias_variable([256])
        output_conv3_2 = tf.nn.relu(conv2d(output_conv3_1, kernel) + biases, name=scope)

    #with tf.name_scope('conv3_3') as scope:
    #    kernel = weight_variable([3, 3, 256, 256])
    #    biases = bias_variable([256])
    #    output_conv3_3 = tf.nn.relu(conv2d(output_conv3_2, kernel) + biases, name=scope)

    #with tf.name_scope('conv3_4') as scope:
    #    kernel = weight_variable([3, 3, 256, 256])
    #    biases = bias_variable([256])
    #    output_conv3_4 = tf.nn.relu(conv2d(output_conv3_3, kernel) + biases, name=scope)

    pool_3 = max_pool_2x2(output_conv3_2, name="conv3_max_pool")

    # conv 4
    with tf.name_scope('conv4_1') as scope:
        kernel = weight_variable([3, 3, 256, 512])
        biases = bias_variable([512])
        output_conv4_1 = tf.nn.relu(conv2d(pool_3, kernel) + biases, name=scope)

    with tf.name_scope('conv4_2') as scope:
        kernel = weight_variable([3, 3, 512, 512])
        biases = bias_variable([512])
        output_conv4_2 = tf.nn.relu(conv2d(output_conv4_1, kernel) + biases, name=scope)

    #with tf.name_scope('conv4_3') as scope:
    #    kernel = weight_variable([3, 3, 512, 512])
    #    biases = bias_variable([512])
    #    output_conv4_3 = tf.nn.relu(conv2d(output_conv4_2, kernel) + biases, name=scope)

    #with tf.name_scope('conv4_4') as scope:
    #    kernel = weight_variable([3, 3, 512, 512])
    #    biases = bias_variable([512])
    #    output_conv4_4 = tf.nn.relu(conv2d(output_conv4_3, kernel) + biases, name=scope)

    pool_4 = max_pool_2x2(output_conv4_2, name="conv4_max_pool")

    # conv 5
    with tf.name_scope('conv5_1') as scope:
        kernel = weight_variable([3, 3, 512, 512])
        biases = bias_variable([512])
        output_conv5_1 = tf.nn.relu(conv2d(pool_4, kernel) + biases, name=scope)

    with tf.name_scope('conv5_2') as scope:
        kernel = weight_variable([3, 3, 512, 512])
        biases = bias_variable([512])
        output_conv5_2 = tf.nn.relu(conv2d(output_conv5_1, kernel) + biases, name=scope)

    #with tf.name_scope('conv5_3') as scope:
    #    kernel = weight_variable([3, 3, 512, 512])
    #    biases = bias_variable([512])
    #    output_conv5_3 = tf.nn.relu(conv2d(output_conv5_2, kernel) + biases, name=scope)

    #with tf.name_scope('conv5_4') as scope:
    #    kernel = weight_variable([3, 3, 512, 512])
    #    biases = bias_variable([512])
    #    output_conv5_4 = tf.nn.relu(conv2d(output_conv5_3, kernel) + biases, name=scope)

    pool_5 = max_pool_2x2(output_conv5_2, name="conv5_max_pool")

    # fc 1
    with tf.name_scope('fc_1') as scope:
        shape = int(np.prod(pool_5.get_shape()[1:]))
        w = weight_variable([shape, 4096])
        b = bias_variable([4096])
        pool_5_flat = tf.reshape(pool_5, [-1, shape])
        fc_1 = tf.nn.relu(fc(pool_5_flat, w, b), name=scope)
        tf.summary.histogram("fc_1", fc_1)

    # fc 2
    #with tf.name_scope('fc_2') as scope:
    #    w = weight_variable([4096, 4096])
    #    b = bias_variable([4096])
    #    fc_2 = tf.nn.relu(fc(fc_1, w, b), name=scope)

    # fc 3
    with tf.name_scope('fc_3') as scope:
        w = weight_variable([4096, 2])
        b = bias_variable([2])
        fc_3 = tf.nn.relu(fc(fc_1, w, b), name=scope)

    final_output = tf.nn.softmax(fc_3, name="output")

    # 定义优化器
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=final_output, labels=y))
    optimize = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
    tf.summary.scalar("cross_entropy", loss)
    
    # 预测标签
    # 预测的标签值
    prediction_labels = tf.argmax(final_output, axis=1, name="result")
    # 实际的标签值
    #real_labels = tf.argmax(y, axis=1, name="standard")
    # 预测与实际进行对比
    #prediction = tf.equal(prediction_labels, real_labels)
    prediction = tf.equal(prediction_labels, y)
    # 准确率
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32), name="accuracy")
    tf.summary.scalar("accuracy", accuracy)
    
    print "结束构建模型"
    return {"x":x, "y":y, "optimize":optimize, "loss":loss, "accuracy":accuracy, "predict":prediction_labels}

# 训练并保存结果
def train_and_save(network, dataset):
    print "开始训练模型"

    # 开始训练
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter("./output/tb/train", sess.graph)
        test_writer = tf.summary.FileWriter("./output/tb/test")
        # 训练结束后，使用 tensorboard --logdir=./log/ --port port 命令打开tensorboard
        sess.run(init)
        #iterator = dataset.make_one_shot_iterator()
        #one_element = iterator.get_next()
        #print one_element
        counter = 0
        try:
            while True:
                #rets = sess.run([network['optimize'], network['loss'], network['accuracy']], feed_dict={network['x']: one_element[0], network['y']: one_element[1]})
                rets = sess.run([network['optimize'], network['loss'], network['accuracy'], merged])
                counter += 1
                print "%d: loss: %f,  accuracy: %f" % (counter, rets[1], rets[2])
                train_writer.add_summary(rets[3], counter)
                # TODO: 这里测试集应该用另外的图片数据
                if counter % 10 == 0:
                    test_writer.add_summary(rets[3], counter)
        except tf.errors.OutOfRangeError:
            print("end! counter=%d" % counter)

        # 保存模型数据
        print "准备保存模型数据"
        saver = tf.train.Saver()
        saver.save(sess, "./output/model/data")

    print "结束训练模型"

# 加载模型数据，应用训练结果识别图片
def load_and_predict(network):
    print "加载已保存的模型数据"

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        # 读取保存的模型数据
        saver = tf.train.Saver()
        saver.restore(sess, "./output/model/data")

        for i in range(20):
            rets = sess.run([network['predict'], network['y']])
            #print "%d: predict: %f,  actually: %f" % (i, rets[0], rets[1])
            print "%d" % (i)
            print rets

    print "结束"

# 加载模型数据，并保存成pb文件
def load_and_write_pb(network):
    print "加载已保存的模型数据"

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        # 读取保存的模型数据
        saver = tf.train.Saver()
        saver.restore(sess, "./output/model/data")

        # 导出pb文件
        print "准备保存pb文件"
        write_pb_file(sess, ["accuracy"], "./output/model.pb")

    print "结束"



# 导出pb文件
def write_pb_file(sess, output_node, output_file_path):
    input_graph_def = sess.graph.as_graph_def()
    output_graph_def = graph_util.convert_variables_to_constants(sess, input_graph_def, output_node)
    with gfile.FastGFile(output_file_path, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    pass

# 读取pb文件，并应用其识别图片
def read_pb_file(pb_file):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        with gfile.FastGFile(pb_file, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            #for node in sess.graph.as_graph_def().node:
            #    print node.name

            accuracy = sess.graph.get_tensor_by_name("accuracy:0")
            rets = sess.run(accuracy)
            print rets
    '''
    with tf.Session() as sess:
        graph_def = sess.graph.as_graph_def()

        with gfile.FastGFile(pb_file, "rb") as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")

        init = tf.global_variables_initializer()
        sess.run(init)

        accuracy = sess.graph.get_tensor_by_name("accuracy:0")
        rets = sess.run(accuracy)
        print rets
    '''

def train():
    dataset = read_data()
    network = build_network(dataset)
    train_and_save(network, dataset)

def test():
    dataset = read_data()
    network = build_network(dataset)
    load_and_predict(network)

def test_write_pb():
    dataset = read_data()
    network = build_network(dataset)
    load_and_write_pb(network)

def test_read_pb():
    read_pb_file("./output/model.pb")

if __name__ == '__main__':
    #train()
    #test()
    #test_write_pb()
    test_read_pb()
