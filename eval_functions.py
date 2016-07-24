"""This module provides the a softmax cross entropy loss for training FCN.

In order to train VGG first build the model and then feed apply vgg_fcn.up
to the loss. The loss function can be used in combination with any optimizer
(e.g. Adam) to finetune the whole model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import numpy as np

def loss(logits, labels, num_classes):
	
	logits = tf.reshape(logits, [-1, num_classes])
	#epsilon = tf.constant(value=1e-4)
	#logits = logits + epsilon
	
	#CHANGE LABELS TYPE to INT, for sparse_softmax_Cross_...
	# to FLOAT, for softmax_Cross_entropy...
	#labels = tf.to_float(tf.reshape(labels, [-1]))
	labels = tf.to_int64(tf.reshape(labels, [-1]))
	#print (np.unique(labels))
	print ('shape of logits: %s' % str(logits.get_shape()))
	print ('shape of labels: %s' % str(labels.get_shape()))
		
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='Cross_Entropy')
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean')
	tf.add_to_collection('losses', cross_entropy_mean)

	loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
	#loss = tf.add_n(cross_entropy)
	return loss


def training(loss, learning_rate):
	"""
	Sets up the training Ops.
	Creates a summarizer to track the loss over time in TensorBoard.
	Creates an optimizer and applies the gradients to all trainable variables.
	The Op returned by this function is what must be passed to the sess.run()` call to cause the model to train.

	Args:
	loss: Loss tensor, from loss().
	learning_rate: The learning rate to use for gradient descent.

	Returns:
	train_op: The Op for training.
	"""
	# Add a scalar summary for the snapshot loss.
	tf.scalar_summary(loss.op.name, loss)

	# Create the gradient descent optimizer with the given learning rate.
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)

	# Create a variable to track the global step.
	global_step = tf.Variable(0, name='global_step', trainable=False)

	# Use the optimizer to apply the gradients that minimize the loss
	# (and also increment the global step counter) as a single training step.
	train_op = optimizer.minimize(loss, global_step=global_step)
	return train_op

def accuracy_eval(logits, labels):
	"""
	Evaluate the quality of the logits at predicting the label.

	Args:
	logits: Logits tensor, float - [batch_size, NUM_CLASSES].
	labels: Labels tensor, int32 - [batch_size], with values in the range [0, NUM_CLASSES).

	Returns:
	A scalar int32 tensor with the number of examples (out of batch_size) that were predicted correctly.
	"""

	# For a classifier model, we can use the in_top_k Op.
	# It returns a bool tensor with shape [batch_size] that is true for
	# the examples where the label is in the top k (here k=1)
	# of all logits for that example.
	
	#correct = tf.nn.in_top_k(logits, labels, 1)
	#tf.cast() might introduce a break in backpropagation - throwing a ValueError!
	#correct = tf.equal(logits, labels)
	logits_index = tf.argmax(logits,3)
	correct = tf.equal(tf.cast(logits_index, tf.float32), tf.cast(labels, tf.float32))
	accuracy = tf.reduce_mean(tf.cast(correct, tf.int32))

	# Return the number of true entries.
	return accuracy
