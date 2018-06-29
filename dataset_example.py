#!/usr/bin/env python
# -*- coding: utf8 -*-

import os
import glob

import cv2
import numpy as np

import tensorflow as tf


#################################################
# Flags define start
#################################################

FLAGS = tf.app.flags.FLAGS

#################################################
# Flags define end
#################################################

def get_data_array(path):
    if not os.path.exists(path):
        raise ValueError('您输入的目录不存在')
    cate   = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs   = []
    labels = []
    for idx, folder in enumerate(cate):
        for img in glob.glob(folder + '/*.jpg'):
            print('add image to list: %s' % (img))
            imgs.append(img)
            labels.append(idx)
    return imgs, labels


########################################
# 生成 Dataset 示例1
# 关键点：
# 1、把文件和标签列表当作常量 Tensor 放进图中，当array很大时，会导致计算图变得很大，给传输、保存带来不便。
#
########################################
def gen_dataset_from_slices_1(path, batch_size=32, epoch_size=None, shuffle_data=False):
    imgs, labels = get_data_array(path)

    print("测试 1 开始")

    ######################
    # 生成 dataset
    ######################

    # 把数组转换成 Tensor
    filenames = tf.constant(imgs, dtype=tf.string)
    labels = tf.constant(labels, dtype=tf.int64)

    # 定义 dataset
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    # 图片预处理，功能是将 filename 对应的图片文件读进来，并缩放到统一的大小
    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.image.resize_images(image_decoded, [28, 28])
        return image_resized, label

    # 预处理数据，把 dataset 中的图片统一处理一遍，主要用于把图片读取进内存中
    dataset = dataset.map(_parse_function)

    if shuffle_data:
        dataset = dataset.shuffle(buffer_size=1000)

    # 先 batch 再 repeat 可能在每个 epoch 最后会有一个尾巴。那么可以先 repeat 再 batch ，就只会在最后有尾巴。
    # 若 repeat 方法不传入参数，则默认是无限循环，训练时要通过其他方式进行停止。
    dataset = dataset.repeat(epoch_size).batch(batch_size)

    ######################
    # session 调用
    ######################
    iterator = dataset.make_one_shot_iterator()
    one_element = iterator.get_next()

    # 这里的 element 是一个 Tensor， 可以用来创建网络
    # x = one_element[0]
    # y = one_element[1]

    count = 0
    with tf.Session() as sess:
        try:
            while True:
                print("count: %d" % count)
                count += 1
                print(sess.run(one_element))
        except tf.errors.OutOfRangeError:
            print("end!")

    print("测试 1 结束")


########################################
# 生成 Dataset 示例2
# 关键点：
# 1、dataset 的输入 Tensor 使用了 placeholder， 并在使用之前进行初始化。
#
########################################
def gen_dataset_from_slices_2(path, batch_size=32, epoch_size=None, shuffle_data=False):
    imgs, labels = get_data_array(path)

    print("测试 2 开始")

    ######################
    # 生成 dataset
    ######################

    # 把数组转换成 Tensor
    files_placeholder = tf.placeholder(tf.string)
    labels_placeholder = tf.placeholder(tf.int64)

    # 定义 dataset
    dataset = tf.data.Dataset.from_tensor_slices((files_placeholder, labels_placeholder))

    # 图片预处理，功能是将 filename 对应的图片文件读进来，并缩放到统一的大小
    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.image.resize_images(image_decoded, [28, 28])
        return image_resized, label

    # 预处理数据，把 dataset 中的图片统一处理一遍，主要用于把图片读取进内存中
    dataset = dataset.map(_parse_function)

    if shuffle_data:
        dataset = dataset.shuffle(buffer_size=1000)

    # 先 batch 再 repeat 可能在每个 epoch 最后会有一个尾巴。那么可以先 repeat 再 batch ，就只会在最后有尾巴。
    # 若 repeat 方法不传入参数，则默认是无限循环，训练时要通过其他方式进行停止。
    dataset = dataset.repeat(epoch_size).batch(batch_size)

    ######################
    # session 调用
    ######################
    # 注意使用 placeholder 时，这里用的是 make_initializable_iterator
    iterator = dataset.make_initializable_iterator()
    one_element = iterator.get_next()

    # 这里的 element 是一个 Tensor， 可以用来创建网络
    # x = one_element[0]
    # y = one_element[1]

    count = 0
    with tf.Session() as sess:
        # 使用 make_initializable_iterator 时，必须先初始化
        sess.run(iterator.initializer, feed_dict={files_placeholder: imgs, labels_placeholder: labels})
        try:
            while True:
                print("count: %d" % count)
                count += 1
                print(sess.run(one_element))
        except tf.errors.OutOfRangeError:
            print("end!")

    print("测试 2 结束")


########################################
# 使用 TFRecord 示例
# 介绍：
#   TensorFlow提供了TFRecord的格式来统一存储数据，TFRecord格式是一种将图像数据和标签放在一起的二进制文件，能更好的利用内存，在tensorflow中快速的复制，移动，读取，存储 等等。 
#   TFRecords文件包含了tf.train.Example 协议内存块(protocol buffer)(协议内存块包含了字段 Features)。
#
########################################
def image_to_tfrecord(path):
    '''
    1.通常我们的图片尺寸并不是统一的，所以在生成的 TFRecord 中需要包含图像的 width 和 height 这两个信息，这样在解析图片的时候，我们才能把二进制的数据重新 reshape 成图片； 
    2.TensorFlow 官方的建议是一个 TFRecord 中最好图片的数量为 1000 张左右，所以我们需要根据图像数据自动去选择到底打包几个 TFRecord 出来。
    ''' 
    imgs, labels = get_data_array(path)

    batch_num = 1000    # 每个 TFRecord 文件所包含的图片个数
    file_index = 0      # TFRecord 文件索引

    count = len(imgs)
    writer = None
    # print(count)
    for i in range(count):
        # print(i)

        # 用 cv2 图片库读取文件
        img = cv2.imread(imgs[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(img.shape)  # 注意： shape 的格式是 (height, width, channel)
        img_raw = img.tobytes()
        width = img.shape[1]
        height = img.shape[0]

        # 用 PIL 图片库读取文件
        #img = Image.open(imgs[i], 'r')
        #size = img.size  # 注意： size 的格式是 (width, height)
        #img_raw = img.tobytes()
        #width = size[0]
        #height = size[1]

        # 生成 Example 对象
        example = tf.train.Example(
             features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[i]])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                'img_width':tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                'img_height':tf.train.Feature(int64_list=tf.train.Int64List(value=[height]))
        }))

        # 创建 TFRecordWriter
        if i % batch_num == 0:
            if writer is not None:
                writer.close()
            writer = tf.python_io.TFRecordWriter("./output/tfrecord/traindata.tfrecord-%.3d" % file_index)
            file_index += 1

        # 写入序列化后的 Example 数据
        writer.write(example.SerializeToString())

    if writer is not None:
        writer.close()


def read_tfrecord():
    files = ["./output/tfrecord/traindata.tfrecord-000"]
    filename_queue = tf.train.string_input_producer(files)

    reader = tf.TFRecordReader()
    # 返回文件名和文件
    _, serialized_example = reader.read(filename_queue)
    # 取出包含 image 和 label 的 feature 对象
    features = tf.parse_single_example(serialized_example,
                                    features={
                                        'label': tf.FixedLenFeature([], tf.int64),
                                        'img_raw' : tf.FixedLenFeature([], tf.string),
                                        'img_width': tf.FixedLenFeature([], tf.int64),
                                        'img_height': tf.FixedLenFeature([], tf.int64),
                                    })

    # tf.decode_raw 可以将字符串解析成图像对应的像素数组
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)
    # 根据宽高重新 reshape 图片
    height = tf.cast(features['img_height'],tf.int32)
    width = tf.cast(features['img_width'],tf.int32)
    channel = 3
    image = tf.reshape(image, [height, width, channel])

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # 启动多线程读取 queue 中的数据
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(10):
            single, l = sess.run([image, label])

            # 测试图片回显
            test_img = cv2.cvtColor(single, cv2.COLOR_RGB2BGR)
            cv2.imshow("test", test_img)
            cv2.waitKey(0)

        # 停止多线程
        coord.request_stop()
        coord.join(threads)



def main(args):
    print("测试开始")
    # gen_dataset_from_slices_1("./picture/", batch_size=10, epoch_size=20, shuffle_data=True)
    # gen_dataset_from_slices_2("./picture/", batch_size=10, epoch_size=20, shuffle_data=True)
    # image_to_tfrecord("./picture/")
    read_tfrecord()
    print("测试结束")


if __name__ == '__main__':
    tf.app.run()
