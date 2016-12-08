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
    tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.scalar_summary('stddev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)

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
    tf.histogram_summary('pre_activations', preactivate)
  y = tf.nn.softmax(preactivate) # 예측 Label 값
  tf.histogram_summary('softmax', y)



with tf.name_scope('cross_entropy'):
  cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]), name="Cross_entropy") # Loss
  tf.scalar_summary('cross entropy', cross_entropy)

with tf.name_scope('train'):
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#### 모델 셋팅 끝 ####


with tf.name_scope('accuracy'):
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1), name="correct_prediction")
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
  tf.scalar_summary('accuracy', accuracy)

# Merge all the summaries and write them out to /tmp/mnist_logs (by default) # 저장 위치 지정가능
merged = tf.merge_all_summaries()


init = tf.initialize_all_variables() # 변수 초기화(텐서플로우 필수과정)

sess = tf.Session() # 세션 열기
sess.run(init) # 초기화

train_writer = tf.train.SummaryWriter('./train',
                                      sess.graph)
test_writer = tf.train.SummaryWriter('./test')

for i in range(1000):

  if i % 10 == 0:  # Record summaries and test-set accuracy
    summary, acc = sess.run([merged, accuracy], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    test_writer.add_summary(summary, i)
    print('Accuracy at step %s: %s' % (i, acc))
  if i % 100 == 99:  # Record execution stats, Starting at 99 and This code will emit runtime statistics for every 100th step
    batch_xs, batch_ys = mnist.train.next_batch(100)  # train, validation, test 에 데이터가 들어가 있음 #  return self._images[start:end], self._labels[start:end]
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y_: batch_ys},
                            options=run_options, run_metadata=run_metadata)
    train_writer.add_run_metadata(run_metadata, 'step%d' % i)
    train_writer.add_summary(summary, i)
    print('Adding run metadata for', i)
  else:
    batch_xs, batch_ys = mnist.train.next_batch(100)
    summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y_: batch_ys})
    train_writer.add_summary(summary, i)


#### 테스트 그래프 셋팅 시작 ####
print(mnist.test.labels)
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
#### 테스트 그래프 셋팅 끝 ####




## Reference
#https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py
