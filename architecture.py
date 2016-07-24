from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
from math import ceil
import sys

import numpy as np
import tensorflow as tf

VGG_MEAN = [103.939, 116.779, 123.68]


class FCN8VGG:
	def __init__(self, vgg16_npy_path=None):
		# locate vgg16.npy weights dictionary
		if vgg16_npy_path is None:
			path = sys.modules[self.__class__.__module__].__file__
			# print path
			path = os.path.abspath(os.path.join(path, os.pardir))
			# print path
			path = os.path.join(path, "vgg16.npy")
			print(path)
			vgg16_npy_path = path

		# Load VGG16 pre-trained weights data dictionary
		"""Dictionary keys: 
		conv1_1, conv1_2, 
		conv2_1, conv2_2, 
		conv3_1, conv3_2, conv3_3, 
		conv4_1, conv4_2, conv4_3, 
		conv5_1, conv5_2, conv5_3, 
		fc6, fc7, fc8
		"""        
		self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
		self.wd = 5e-4 # weight decay factor
		print("npy file loaded")

	def build(self, rgb, train=True, num_classes=2, random_init_fc8=True,
			  debug=True):
		"""
		Build the VGG model using loaded weights
		Parameters
		----------
		rgb: image batch tensor
			Image in rgb shap. Scaled to Intervall [0, 255]
		train: bool
			Whether to build train or inference graph
		num_classes: int
			How many classes should be predicted (by fc8)
		random_init_fc8 : bool
			Whether to initialize fc8 layer randomly.
			Finetuning is required in this case.
		debug: bool
			Whether to print additional Debug Information.
		"""
		# Convert RGB to BGR

		with tf.name_scope('Processing'):

			red, green, blue = tf.split(3, 3, rgb)
			bgr = tf.concat(3, [
				blue - VGG_MEAN[0],
				green - VGG_MEAN[1],
				red - VGG_MEAN[2],
			])

			if debug:
				bgr = tf.Print(bgr, [tf.shape(bgr)],
							   message='Shape of input image: ',
							   summarize=4, first_n=1)

		self.conv1_1 = self._conv_layer(bgr, "conv1_1")
		self.conv1_2 = self._conv_layer(self.conv1_1, "conv1_2")
		self.pool1 = self._max_pool(self.conv1_2, 'pool1', debug)

		self.conv2_1 = self._conv_layer(self.pool1, "conv2_1")
		self.conv2_2 = self._conv_layer(self.conv2_1, "conv2_2")
		self.pool2 = self._max_pool(self.conv2_2, 'pool2', debug)

		self.conv3_1 = self._conv_layer(self.pool2, "conv3_1")
		self.conv3_2 = self._conv_layer(self.conv3_1, "conv3_2")
		self.conv3_2 = self._conv_layer(self.conv3_2, "conv3_3")
		self.pool3 = self._max_pool(self.conv3_2, 'pool3', debug)

		self.conv4_1 = self._conv_layer(self.pool3, "conv4_1")
		self.conv4_2 = self._conv_layer(self.conv4_1, "conv4_2")
		self.conv4_3 = self._conv_layer(self.conv4_2, "conv4_3")
		self.pool4 = self._max_pool(self.conv4_3, 'pool4', debug)

		self.conv5_1 = self._conv_layer(self.pool4, "conv5_1")
		self.conv5_2 = self._conv_layer(self.conv5_1, "conv5_2")
		self.conv5_3 = self._conv_layer(self.conv5_2, "conv5_3")
		self.pool5 = self._max_pool(self.conv5_3, 'pool5', debug)

		# Fully connected layers, converted to convolutions. Named fc6 and fc7 to retain nomenclature from VGG16 
		# Fully connected layer as a convolution [7x7x512, 4096 filters]
		self.fc6 = self._fc_layer(self.pool5, "fc6")

		if train:
			self.fc6 = tf.nn.dropout(self.fc6, 0.5)

		# Fully connected layer as a convolution [1x1x4096, 4096 filters]
		self.fc7 = self._fc_layer(self.fc6, "fc7")
		if train:
			self.fc7 = tf.nn.dropout(self.fc7, 0.5)

		self.conv_11 = self._fc_layer(self.fc7, "conv_11", num_classes=2)

		#if random_init_fc8:
		#	self.score_fr = self._score_layer(self.fc7, "score_fr",
		#									  num_classes)
		#else:
		#	self.score_fr = self._fc_layer(self.fc7, "score_fr",
		#								   num_classes=num_classes,
		#								   relu=False)

		#self.pred = tf.argmax(self.score_fr, dimension=3)

		self.conv_10 = self._fc_layer(self.pool4, "conv_10", num_classes=2)

		self.deconv_1 = self._upscore_layer(self.conv_11,
											shape=tf.shape(bgr),
											num_classes=2,
											debug=debug, name='deconv_1',
											ksize=64, stride=32)

		#self.conv_out = self._fc_layer(self.deconv_1, "conv_out", debug=True, num_classes=2)
		
		self.pred_up = tf.argmax(self.deconv_1, dimension=3)

		"""
		self.score_pool4 = self._score_layer(self.pool4, "score_pool4",
											 num_classes=num_classes)
		
		self.fuse_pool4 = tf.add(self.deconv_1, self.score_pool4)

		self.deconv_2 = self._upscore_layer(self.fuse_pool4,
											shape=tf.shape(self.pool3),
											num_classes=num_classes,
											debug=debug, name='deconv_2',
											ksize=4, stride=2)
		self.score_pool3 = self._score_layer(self.pool3, "score_pool3",
											 num_classes=num_classes)
		self.fuse_pool3 = tf.add(self.deconv_2, self.score_pool3)

		self.deconv_3 = self._upscore_layer(self.fuse_pool3,
											 shape=tf.shape(bgr),
											 num_classes=num_classes,
											 debug=debug, name='deconv_3',
											 ksize=16, stride=8)
		"""

		#self.logits = self._softmax(self.deconv_1, 2)

		#self.pred_up = tf.argmax(self.conv_out, dimension=3)


	def _softmax(self, bottom, num_classes):
		# computes softmax
		with tf.variable_scope('logits') as scope:
			n1 = tf.to_float(bottom.get_shape()[1])
			#n1size = 1*750*750*2 
			stddev = (1 / n1)**0.5
			weights = self._variable_with_weight_decay(shape=[n1 , num_classes],
													stddev=stddev, wd=0.0)
			biases = self._bias_variable([num_classes])
			logits = tf.add(tf.matmul(bottom, weights), bias, name=scope.name)
			_activation_summary(logits)

		return logits


	def _max_pool(self, bottom, name, debug):
		pool = tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
							  padding='SAME', name=name)

		if debug:
			pool = tf.Print(pool, [tf.shape(pool)],
							message='Shape of %s' % name,
							summarize=4, first_n=1)
		return pool

	def _deconv_reshape(self, bottom, num_classes=2):
		logits = tf.to_float(tf.reshape(bottom, (-1, num_classes)))
		return logits

	def _conv_layer(self, bottom, name):
		with tf.variable_scope(name) as scope:
			filt = self.get_conv_filter(name)
			conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

			conv_biases = self.get_bias(name)
			bias = tf.nn.bias_add(conv, conv_biases)

			relu = tf.nn.relu(bias)
			# Add summary to Tensorboard
			_activation_summary(relu)
			return relu

	def _fc_layer(self, bottom, name, num_classes=None,
				  relu=True, debug=True):
		with tf.variable_scope(name) as scope:
			shape = bottom.get_shape().as_list()

			if name == 'fc6':
				filt = self.get_fc_weight_reshape(name, [7, 7, 512, 4096])
			elif name == 'score_fr':
				name = 'fc8'  # Name of score_fr layer in VGG Model
				filt = self.get_fc_weight_reshape(name, [1, 1, 4096, 1000],
												  num_classes=num_classes)
			elif name == 'conv_11':
				filt = self.get_fc_weight(bottom, name=name)
			elif name == 'conv_10':
				filt = self.get_fc_weight(bottom, name=name)
			elif name == 'conv_out':
				filt = self.get_fc_weight(bottom, name=name)
			else:
				filt = self.get_fc_weight_reshape(name, [1, 1, 4096, 4096])
			conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

			if name == 'conv_11':
				conv_biases = self.higher_bias(bottom, name=name)
			elif name == 'conv_10':
				conv_biases = self.higher_bias(bottom, name=name)
			elif name == 'conv_out':
				conv_biases = self.higher_bias(bottom, name=name)
			else:
				conv_biases = self.get_bias(name, num_classes=num_classes)
			bias = tf.nn.bias_add(conv, conv_biases)

			if relu:
				bias = tf.nn.relu(bias)
			_activation_summary(bias)

			if debug:
				bias = tf.Print(bias, [tf.shape(bias)],
								message='Shape of %s' % name,
								summarize=4, first_n=1)
			return bias

	def _score_layer(self, bottom, name, num_classes):
		with tf.variable_scope(name) as scope:
			# get number of input channels
			in_features = bottom.get_shape()[3].value
			shape = [1, 1, in_features, num_classes]
			# Initialization scheme
			if name == "score_fr":
				num_input = in_features
				stddev = (2 / num_input)**0.5
			elif name == "score_pool4":
				stddev = 0.001
			elif name == "score_pool3":
				stddev = 0.0001
			# Apply convolution
			w_decay = self.wd
			weights = self._variable_with_weight_decay(shape, stddev, w_decay)
			conv = tf.nn.conv2d(bottom, weights, [1, 1, 1, 1], padding='SAME')
			# Apply bias
			conv_biases = self._bias_variable([num_classes], constant=0.0)
			bias = tf.nn.bias_add(conv, conv_biases)

			_activation_summary(bias)

			return bias

	def _upscore_layer(self, bottom, shape,
					   num_classes, name, debug,
					   ksize=4, stride=2):
		strides = [1, stride, stride, 1]
		with tf.variable_scope(name):
			in_features = bottom.get_shape()[3].value

			if shape is None:
				# Compute shape out of Bottom
				in_shape = tf.shape(bottom)

				h = ((in_shape[1] - 1) * stride) + 1
				w = ((in_shape[2] - 1) * stride) + 1
				new_shape = [in_shape[0], h, w, num_classes]
			else:
				new_shape = [shape[0], shape[1], shape[2], num_classes]
			output_shape = tf.pack(new_shape)

			logging.debug("Layer: %s, Fan-in: %d" % (name, in_features))
			f_shape = [ksize, ksize, num_classes, in_features]

			# create
			num_input = ksize * ksize * in_features / stride
			stddev = (2 / num_input)**0.5

			weights = self.get_deconv_filter(f_shape)
			deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
											strides=strides, padding='SAME')

			if debug:
				deconv = tf.Print(deconv, [tf.shape(deconv)],
								  message='Shape of %s' % name,
								  summarize=4, first_n=1)

		_activation_summary(deconv)
		return deconv

	def get_deconv_filter(self, f_shape):
		width = f_shape[0]
		heigh = f_shape[0]
		f = ceil(width/2.0)
		c = (2 * f - 1 - f % 2) / (2.0 * f)
		bilinear = np.zeros([f_shape[0], f_shape[1]])
		for x in range(width):
			for y in range(heigh):
				value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
				bilinear[x, y] = value
		weights = np.zeros(f_shape)
		for i in range(f_shape[2]):
			weights[:, :, i, i] = bilinear

		init = tf.constant_initializer(value=weights,
									   dtype=tf.float32)
		return tf.get_variable(name="up_filter", initializer=init,
							   shape=weights.shape)

	def get_conv_filter(self, name):
		init = tf.constant_initializer(value=self.data_dict[name][0],
									   dtype=tf.float32)
		shape = self.data_dict[name][0].shape
		print('Layer name: %s' % name)
		print('Layer shape: %s' % str(shape))
		var = tf.get_variable(name="filter", initializer=init, shape=shape)
		if not tf.get_variable_scope().reuse:
			weight_decay = tf.mul(tf.nn.l2_loss(var), self.wd,
								  name='weight_loss')
			tf.add_to_collection('losses', weight_decay)
		return var

	def get_bias(self, name, num_classes=None):
		bias_weights = self.data_dict[name][1]
		shape = self.data_dict[name][1].shape
		if name == 'fc8':
			bias_weights = self._bias_reshape(bias_weights, shape[0],
											 num_classes)
			shape = [num_classes]
		init = tf.constant_initializer(value=bias_weights,
									   dtype=tf.float32)
		return tf.get_variable(name="biases", initializer=init, shape=shape)

	def higher_bias(self,bottom, name):
		if name == 'conv_11':
			shape = [2]
		elif name == 'conv_10':
			shape = [2]
		elif name == 'conv_out':
			shape = [3]
		init = tf.constant_initializer(value=0.1,dtype=tf.float32)
		var = tf.get_variable(name="biases", initializer=init, shape=shape)

		return var

	def get_fc_weight_reshape(self, name, shape, num_classes=None):
		print('Layer name: %s' % name)
		print('Layer shape: %s' % shape)
		weights = self.data_dict[name][0]
		weights = weights.reshape(shape)
		if num_classes is not None:
			weights = self._filter_weights_reshape(weights, shape,
											num_new=num_classes)
		init = tf.constant_initializer(value=weights,
									   dtype=tf.float32)
		return tf.get_variable(name="weights", initializer=init, shape=shape)

	def get_fc_weight(self, bottom, name):
		if name == 'conv_11':
			shape = [1,1,4096,2]
		elif name == 'conv_10':
			shape = [1,1,512,2]
		elif name == 'conv_out':
			shape = [3,3,2,3]
		init = tf.constant_initializer(value=0.1,
									   dtype=tf.float32)
		#shape = self.data_dict[name][0].shape
		var = tf.get_variable(name="weights", initializer=init, shape=shape)
		if not tf.get_variable_scope().reuse:
			weight_decay = tf.mul(tf.nn.l2_loss(var), self.wd,
								  name='weight_loss')
			tf.add_to_collection('losses', weight_decay)
		return var

	def _bias_reshape(self, bweight, num_orig, num_new):
		""" Build bias weights for filter produces with `_filter_weights_reshape`

		"""
		n_averaged_elements = num_orig//num_new
		avg_bweight = np.zeros(num_new)
		for i in range(0, num_orig, n_averaged_elements):
			start_idx = i
			end_idx = start_idx + n_averaged_elements
			avg_idx = start_idx//n_averaged_elements
			if avg_idx == num_new:
				break
			avg_bweight[avg_idx] = np.mean(bweight[start_idx:end_idx])
		return avg_bweight

	def _filter_weights_reshape(self, fweight, shape, num_new):
		""" Produce weights for a reduced fully-connected layer.

		FC8 of VGG produces 1000 classes. Most semantic segmentation
		task require much less classes. This reshapes the original weights
		to be used in a fully-convolutional layer which produces num_new
		classes. To archive this the average (mean) of n adjanced classes is
		taken.

		Consider reordering fweight, to perserve semantic meaning of the
		weights.

		Args:
		  fweight: original weights
		  shape: shape of the desired fully-convolutional layer
		  num_new: number of new classes


		Returns:
		  Filter weights for `num_new` classes.
		"""
		num_orig = shape[3]
		shape[3] = num_new
		assert(num_new < num_orig)
		n_averaged_elements = num_orig//num_new
		avg_fweight = np.zeros(shape)
		for i in range(0, num_orig, n_averaged_elements):
			start_idx = i
			end_idx = start_idx + n_averaged_elements
			avg_idx = start_idx//n_averaged_elements
			if avg_idx == num_new:
				break
			avg_fweight[:, :, :, avg_idx] = np.mean(
				fweight[:, :, :, start_idx:end_idx], axis=3)
		return avg_fweight

	def _variable_with_weight_decay(self, shape, stddev, wd):
		"""Helper to create an initialized Variable with weight decay.

		Note that the Variable is initialized with a truncated normal
		distribution.
		A weight decay is added only if one is specified.

		Args:
		  name: name of the variable
		  shape: list of ints
		  stddev: standard deviation of a truncated Gaussian
		  wd: add L2Loss weight decay multiplied by this float. If None, weight
			  decay is not added for this Variable.

		Returns:
		  Variable Tensor
		"""

		initializer = tf.truncated_normal_initializer(stddev=stddev)
		var = tf.get_variable('weights', shape=shape,
							  initializer=initializer)

		if wd and (not tf.get_variable_scope().reuse):
			weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
			tf.add_to_collection('losses', weight_decay)
		return var

	def _bias_variable(self, shape, constant=0.0):
		initializer = tf.constant_initializer(constant)
		return tf.get_variable(name='biases', shape=shape,
							   initializer=initializer)




def _activation_summary(x):
	"""Helper to create summaries for activations.

	Creates a summary that provides a histogram of activations.
	Creates a summary that measure the sparsity of activations.

	Args:
	  x: Tensor
	Returns:
	  nothing
	"""
	# Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
	# session. This helps the clarity of presentation on tensorboard.
	tensor_name = x.op.name
	# tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
	tf.histogram_summary(tensor_name + '/activations', x)
	tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
