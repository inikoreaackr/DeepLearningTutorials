#-*- coding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) # Yann LeCun's website에서 MNIST 다운
import tensorflow as tf




#### 모델 셋팅 시작 ####

x = tf.placeholder(tf.float32, [None, 784])  # 데이터 담을 placeholder 선언

W = tf.Variable(tf.zeros([784, 10])) # 학습할 Weight Matrix
b = tf.Variable(tf.zeros([10])) # 학습할 bias

y = tf.nn.softmax(tf.matmul(x, W) + b) # 예측 Label 값

y_ = tf.placeholder(tf.float32, [None, 10]) # True Label 값

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])) # Loss

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#### 모델 셋팅 끝 ####


init = tf.initialize_all_variables() # 변수 초기화(텐서플로우 필수과정)

sess = tf.Session() # 세션 열기
sess.run(init) # 초기화

for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100) # train, validation, test 에 데이터가 들어가 있음 #  return self._images[start:end], self._labels[start:end]
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# print(batch_xs[0]) # 들어간 데이터 확인
#### 테스팅 모델 셋팅 시작 ####
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) # argmax(input,  dimension of the input Tensor to reduce across)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#### 테스팅 모델 셋팅 끝 ####
print(mnist.test.labels)
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))



