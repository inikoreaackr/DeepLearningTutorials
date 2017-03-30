#-*- coding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) # Yann LeCun's website에서 MNIST 다운
import tensorflow as tf


#tensorboard --logdir=path/to/log-directory
#TensorBoard operates by reading TensorFlow events files,
#  which contain summary data that you can generate when running TensorFlow.

def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev/' + name, stddev)
    tf.summary.scalar('max/' + name, tf.reduce_max(var))
    tf.summary.scalar('min/' + name, tf.reduce_min(var))
    tf.summary.histogram(name, var)

#### 모델 셋팅 시작 ####
with tf.name_scope('input'):
  x = tf.placeholder(tf.float32, [None, 784], name="placeholder_mnist")  # 데이터 담을 placeholder 선언
  y_ = tf.placeholder(tf.float32, [None, 10], name="placeholder_TrueLabel") # True Label 값

# to define a hierarchy on the nodes in the graph
# The better your name scopes, the better your visualization.
with tf.name_scope('FC_Layer') as scope:
  with tf.name_scope('weights'):
    W = tf.Variable(tf.zeros([784, 10]), name="var_Weight") # 학습할 Weight Matrix
    variable_summaries(W, 'weights')
  with tf.name_scope('biases'):
    b = tf.Variable(tf.zeros([10]), name="var_Bias") # 학습할 bias
    variable_summaries(b, 'biases')
  with tf.name_scope('Wx_plus_b'):
    preactivate = tf.matmul(x, W) + b
    tf.summary.histogram('pre_activations', preactivate)
  y = tf.nn.softmax(preactivate) # 예측 Label 값
  tf.summary.histogram('softmax', y)



with tf.name_scope('cross_entropy'):
  cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=[1]), name="Cross_entropy") # Loss
  tf.summary.scalar('cross entropy', cross_entropy)

with tf.name_scope('train'):
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#### 모델 셋팅 끝 ####


with tf.name_scope('accuracy'):
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1), name="correct_prediction")
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
  tf.summary.scalar('accuracy', accuracy)

# Merge all the summaries and write them out to /tmp/mnist_logs (by default) # 저장 위치 지정가능
merged = tf.summary.merge_all()


init = tf.global_variables_initializer() # 변수 초기화(텐서플로우 필수과정)

sess = tf.Session() # 세션 열기
sess.run(init) # 초기화

train_writer = tf.summary.FileWriter('./train',
                                      sess.graph)
test_writer = tf.summary.FileWriter('./test')

for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100) # train, validation, test 에 데이터가 들어가 있음 #  return self._images[start:end], self._labels[start:end]
  #sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y_: batch_ys})
  train_writer.add_summary(summary, i)


#### 테스트 세션 시작 ####
print(mnist.test.labels)
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
#### 테스트 세션 끝 ####



