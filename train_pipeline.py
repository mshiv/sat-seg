from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import eval_functions
import data_input
import architecture

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 20, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 1, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('display_step', 2, 'Decides steps of displayed loss/acc values')
flags.DEFINE_string('train_dir', '/tensorboard-event-files/', 'Directory to put the training data.')
#flags.DEFINE_string('train_dir', '/tmp/basi/', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')


def run_training():
	data_sets = data_input.read_data_sets('Dataset')
	vgg_fcn = architecture.FCN8VGG()

	images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)

	with tf.Session() as sess:

		with tf.name_scope('input_images'):
			tf.image_summary('input', images_placeholder, 1)

		# Instantiate a SummaryWriter to output summaries and the Graph.
		#summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

		with tf.name_scope("fcn32s-architecture"):
			vgg_fcn.build(images_placeholder, num_classes=2, debug=True)

		feed_dict = fill_feed_dict(data_sets.train, images_placeholder, labels_placeholder)

		with tf.name_scope('loss'):
			
			with tf.name_scope('logits'):
				logits = vgg_fcn.deconv_1
				#logits = sess.run(decon, feed_dict= feed_dict)
				#tf.scalar_summary('logit_summary', logits)

			loss = eval_functions.loss(logits, labels_placeholder, 2)
			#loss_summary = tf.scalar_summary('cross_entropy', loss)

		with tf.name_scope('Optimizer'):
			train_op = eval_functions.training(loss, FLAGS.learning_rate)

		with tf.name_scope('Accuracy'):
			eval_correct = eval_functions.accuracy_eval(logits, labels_placeholder)
			accuracy_summary = tf.scalar_summary('Accuracy', eval_correct)


		# Build the summary operation based on the TF collection of Summaries.
		summary_op = tf.merge_all_summaries()
		# Create a saver for writing training checkpoints.
		saver = tf.train.Saver()
		# Add the variable initializer Op.
		init = tf.initialize_all_variables()
		# And then after everything is built:
		# Create a session for running Ops on the Graph.
		sess = tf.Session()

		summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)
		# Run the Op to initialize the variables.
		sess.run(init)

		print ("beginning session run:")

		for step in xrange(FLAGS.max_steps):
			start_time = time.time()
			#print (start_time)

			# Fill a feed dictionary with the actual set of images and labels
			# for this particular training step.
			#feed_dict = fill_feed_dict(data_sets.train, images_placeholder, labels_placeholder)

			# Run one step of the model.  The return values are the activations
			# from the `train_op` (which is discarded) and the `loss` Op.  To
			# inspect the values of your Ops or variables, you may include them
			# in the list passed to sess.run() and the value tensors will be
			# returned in the tuple from the call.
			
			_, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

			duration = time.time() - start_time


			# Write the summaries and print an overview fairly often.
			if step % FLAGS.display_step == 0:
				# Print status to stdout.
				print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
				#print ("Step" + str(step) + ": Loss= " + "{:.6f}".format(loss_value))# + "(%.3f sec)".format(duration))

				# Update the events file.
				#merged = tf.merge_summary([loss_summary, accuracy_summary])
				summary_str = sess.run(summary_op, feed_dict=feed_dict)
				summary_writer.add_summary(summary_str, step)
				summary_writer.flush()

			# Save a checkpoint and evaluate the model periodically.
			if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
				saver.save(sess, FLAGS.train_dir, global_step=step)
			
				# Evaluate against the training set.
				print('Training Data Eval:')
				do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.train)
			'''
				# Evaluate against the test set.
				print('Test Data Eval:')
				do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.test)
			'''


def placeholder_inputs(batch_size):
	"""
	Generate placeholder variables to represent the input tensors.
	These placeholders are used as inputs by the rest of the model building
	code and will be fed from the imported data in the .run() loop, below.

	Args:
	batch_size: The batch size will be baked into both placeholders.

	Returns:
	images_placeholder: Images placeholder.
	labels_placeholder: Labels placeholder.
	"""
	images_placeholder = tf.placeholder(tf.float32)#, shape=[None, 750, 750, 3])
	labels_placeholder = tf.placeholder(tf.float32)#, shape=[None, 750, 750])#, 3])
	return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl):
	"""
	Fills the feed_dict for training the given step.
	A feed_dict takes the form of:
	feed_dict = {
	<placeholder>: <tensor of values to be passed for placeholder>,
	....
	}

	Args:
	data_set: The set of images and labels, from input_data.read_data_sets()
	images_pl: The images placeholder, from placeholder_inputs().
	labels_pl: The labels placeholder, from placeholder_inputs().

	Returns:
	feed_dict: The feed dictionary mapping from placeholders to values.
	"""
	images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size, FLAGS.fake_data)
	feed_dict = {images_pl: images_feed, labels_pl: labels_feed}
	return feed_dict

def do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_set):
	"""
	Runs one evaluation against the full epoch of data.

	Args:
	sess: The session in which the model has been trained.
	eval_correct: The Tensor that returns the number of correct predictions.
	images_placeholder: The images placeholder.
	labels_placeholder: The labels placeholder.
	data_set: The set of images and labels to evaluate from input_data.read_data_sets().
	"""
	#And run one epoch of eval.
	true_count = 0 # Counts number of correct predictions
	steps_per_epoch = data_set.num_examples // FLAGS.batch_size
	num_examples = steps_per_epoch * FLAGS.batch_size

	for step in xrange(steps_per_epoch):
		feed_dict = fill_feed_dict(data_set, images_placeholder, labels_placeholder)
		true_count += sess.run(eval_correct, feed_dict=feed_dict)
	precision = true_count / num_examples
	print('Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' % (num_examples, true_count, precision))


def main(_):
  run_training()


if __name__ == '__main__':
  tf.app.run()
