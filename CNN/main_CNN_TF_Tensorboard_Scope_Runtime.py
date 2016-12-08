#-*- coding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

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

# Create Session
sess = tf.InteractiveSession()

# Input data
with tf.name_scope('input'):
  x = tf.placeholder(tf.float32, shape=[None, 784], name="placeholder_MNIST") #x-input
  y_ = tf.placeholder(tf.float32, shape=[None, 10], name="placeholder_TrueLabel") #y-input


# Weight Initialization
def weight_variable(shape, name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)

def bias_variable(shape, name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name)

# Convolution and Pooling
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x, name):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME', name=name)



# Reshape Input
with tf.name_scope('input_reshape'):
  x_image = tf.reshape(x, [-1,28,28,1], name="x_image_reshape") # ?, width, height, Dim of Color   # 1 dim -> 4d tensor
  tf.image_summary('input', x_image, 10)




with tf.name_scope("Conv_Layer1"):
  # First Convolutional Layer
  W_conv1 = weight_variable([5, 5, 1, 32], name="W_conv1")  # 5x5 filter, Num of Input channel == 1, 32 features
  b_conv1 = bias_variable([32], name="b_conv1")
  variable_summaries(W_conv1, 'W_conv1')
  variable_summaries(b_conv1, 'b_conv1')

  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1, name="h_conv1")
  variable_summaries(h_conv1, 'h_conv1')

with tf.name_scope("Pooling_Layer1"):
  h_pool1 = max_pool_2x2(h_conv1, name="h_pool1")
  variable_summaries(h_pool1, 'h_pool1')



with tf.name_scope("Conv_Layer2"):
  W_conv2 = weight_variable([5, 5, 32, 64], name="W_conv2")
  b_conv2 = bias_variable([64], name="b_conv2")
  variable_summaries(W_conv2, 'W_conv2')
  variable_summaries(b_conv2, 'b_conv2')

  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2, name="h_conv2")
  variable_summaries(h_conv2, 'h_conv2')

with tf.name_scope("Pooling_Layer2"):
  h_pool2 = max_pool_2x2(h_conv2, name="h_pool2")
  variable_summaries(h_pool2, 'h_pool2')



# Fully Connected Layer
with tf.name_scope("FC_Layer1"):
  W_fc1 = weight_variable([7 * 7 * 64, 1024], name="W_fc1") # Now that the image size has been reduced to 7x7 (Pooiling 2번!)
  b_fc1 = bias_variable([1024], name="b_fc1")
  variable_summaries(W_fc1, 'W_fc1')
  variable_summaries(b_fc1, 'b_fc1')
  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64], name="h_pool2_flat")
  with tf.name_scope('Wx_plus_b_with_ReLU'):
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name="h_fc1")



# Dropout
with tf.name_scope('dropout'):
  keep_prob = tf.placeholder(tf.float32, name="keep_prob")
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name="h_fc1_drop")
  variable_summaries(keep_prob, 'dropout_keep_probability')
  variable_summaries(h_fc1_drop, 'h_fc1_drop')


# Readout Layer (FC 2)
with tf.name_scope("FC_Layer2"):
  W_fc2 = weight_variable([1024, 10], name="W_fc2")
  b_fc2 = bias_variable([10], name="b_fc2")
  variable_summaries(W_fc2, 'W_fc2')
  variable_summaries(b_fc2, 'b_fc2')
  with tf.name_scope('Wx_plus_b'):
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    tf.histogram_summary('y_conv', y_conv)



# Train and Evaluate the Model
with tf.name_scope("cross_entropy"):
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_), name="cross_entropy")
  tf.scalar_summary('cross entropy', cross_entropy)

with tf.name_scope('train'):
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


with tf.name_scope('accuracy'):
  with tf.name_scope('correct_prediction'):
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1), name="correct_prediction") # argmax(input,  dimension of the input Tensor to reduce across)
    #tf.scalar_summary('correct_prediction', correct_prediction) #에러 발생
  with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
  tf.scalar_summary('accuracy', accuracy)


merged = tf.merge_all_summaries()

sess.run(tf.initialize_all_variables())

train_writer = tf.train.SummaryWriter('./train', sess.graph)
test_writer = tf.train.SummaryWriter('./test')

for i in range(20000):

  if i%100 == 0:
    # Training Accuracy Code
    batch = mnist.train.next_batch(50)
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
    summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    train_writer.add_summary(summary, i)

    #Test Accuracy Code
    #summary, acc = sess.run([merged, accuracy], feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
    #test_writer.add_summary(summary, i)
    #print('step %d, Test Accuracy : %s' % (i, acc))


  if i % 100 == 99:  # Record execution stats, Starting at 99 and This code will emit runtime statistics for every 100th step
    batch = mnist.train.next_batch(50)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5},
                            options=run_options, run_metadata=run_metadata)
    train_writer.add_run_metadata(run_metadata, 'step%d' % i)
    train_writer.add_summary(summary, i)
    print('Adding run metadata for', i)
  else:
    batch = mnist.train.next_batch(50)
    summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    train_writer.add_summary(summary, i)

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
