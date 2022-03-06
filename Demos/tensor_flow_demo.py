#! /usr/bin/python
# -*- coding: UTF-8 -*-
# Author: UMR
import tensorflow as tf
import input_data
'''
导入MNIST数据集
'''
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
'''
交互式会话定义方式
'''
sess = tf.InteractiveSession()  # 通过它可以再运行图的时候插入一些计算图
'''
定义输入输出，初始化权值偏置
'''
X = tf.placeholder('float', [None, 28*28])
y_ = tf.placeholder('float', [None, 10])


def weight_variable(shape):  # 使用高斯随机截断赋予权重初始量
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):  # 使用全0.1常量赋予偏置初始量
    return tf.Variable(tf.constant(0.1, shape=shape))


'''
卷积和池化:strides为4维向量[1,a,b,1], a、b分别为宽、高方向步长
'''
def conv2d(X, filter):
    return tf.nn.conv2d(X, filter, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(X):
    return tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


'''
第一层卷积+池化
'''
W_conv1 = weight_variable([5, 5, 1, 32])  # shape = [宽， 高， 通道数， 卷积核数量]
b_conv1 = bias_variable([32])
X_input = tf.reshape(X, [-1, 28, 28, 1])  # [自动匹配， 宽， 高， 通道数]
h_conv1 = tf.nn.relu(conv2d(X_input, W_conv1) + b_conv1)  # 卷积
h_pool1 = max_pool_2x2(h_conv1)  # 池化

'''
第二层卷积+池化
'''
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

'''
全连接层
'''
W_fc1 = weight_variable([7*7*64, 1024])  # 全连接层权重矩阵尺寸为:前层神经元数 * 后层神经元数
b_fc1 = bias_variable([1024])
h_pool2_flatten = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flatten, W_fc1) + b_fc1)

'''
Dropout:必须添加占位符描述保留概率，因为需要在训练中开启dropout，在测试中关闭dropout
'''
keep_prob = tf.placeholder('float')  # 保留该神经元的概率
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

'''
输出层：Softmax输出
'''
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

'''
交叉熵损失函数
'''
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

'''
BP算法 + accuracy评价
'''
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  # 使用Adam梯度下降法
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), 'float'))
sess.run(tf.global_variables_initializer())  # 初始化变量
for i in range(20000):
    batch_x, batch_y = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={X: batch_x, y_: batch_y, keep_prob: 1})  # 计算准确率
        print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={X: batch_x, y_: batch_y, keep_prob: 0.5})
print('test accuracy %g' % (accuracy.eval(feed_dict={X: mnist.test.images, y_: mnist.test.labels, keep_prob: 1})))
