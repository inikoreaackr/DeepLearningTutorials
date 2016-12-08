#-*- coding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf

# Create Session
sess = tf.InteractiveSession()

# Input data
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Variables
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

print(W.get_shape())
print(b.get_shape())

# Session run for Init variables
sess.run(tf.initialize_all_variables())

# Predict
y = tf.matmul(x,W) + b
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

# Training Model
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
for i in range(1000):
  batch = mnist.train.next_batch(100) # 변수 두개로 아웃풋을 받거나 하나로 받은 뒤 0, 1 index로 접근하거나
  train_step.run(feed_dict={x: batch[0], y_: batch[1]}) # 여기선 바로 .run으로 세션을 사용(Interactive session때문)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
print(accuracy)



