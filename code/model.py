from __future__ import print_function
import numpy as np
import tensorflow as tf
import time
import random
import math
import sys

class Model(object):
	"""docstring for model"""
	def __init__(self, data_name, num_class, embeddings, size = 300, batch_size = 64, dropout = 0.5, max_grad_norm = 5.0, L2reg = 0.000001, 
				 rnn_cell = 'lstm', optimize = 'Adagrad'):
		self.data_name = data_name
		self.num_class = num_class
		self.size = size
		self.batch_size = batch_size
		self.dropout = dropout
		self.max_grad_norm = max_grad_norm
		self.bidirection = rnn_cell.startswith('bi')
		self.rnn_cell = rnn_cell
		self.embeddings = embeddings
		self.optimize = optimize
		self.log_file = '../data/' + self.data_name + '/log' + '_' + rnn_cell + '_' + str(size)

		self.build()
	# def linear(self, input_, output_size, scope=None):
	#     '''
	#     Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]
	#     Args:
	#         args: a tensor or a list of 2D, batch x n, Tensors.
	#     output_size: int, second dimension of W[i].
	#     scope: VariableScope for the created subgraph; defaults to "Linear".
	#     Returns:
	#     A 2D Tensor with shape [batch x output_size] equal to
	#     sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
	#     Raises:
	#     ValueError: if some of the arguments has unspecified or wrong shape.
	#     '''

	#     shape = input_.get_shape().as_list()
	#     if len(shape) != 2:
	#         raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
	#     if not shape[1]:
	#         raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
	#     input_size = shape[1]

	#     # Now the computation.
	#     with tf.variable_scope(scope or "SimpleLinear"):
	#         matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
	#         bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

	#     return tf.matmul(input_, tf.transpose(matrix)) + bias_term

	def linear(self, input_, output_size, scope=None):

	    shape = input_.get_shape().as_list()
	    if len(shape) != 2:
	        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
	    if not shape[1]:
	        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
	    input_size = shape[1]

	    # Now the computation.
	    with tf.variable_scope(scope or "SimpleLinear"):
	        matrix1 = tf.get_variable("Matrix1", [output_size+10, input_size], dtype=input_.dtype)
	        bias_term1 = tf.get_variable("Bias1", [output_size+10], dtype=input_.dtype)
	        matrix2 = tf.get_variable("Matrix2", [output_size, output_size+10], dtype=input_.dtype)
	        bias_term2 = tf.get_variable("Bias2", [output_size], dtype=input_.dtype)
	        hidden1 = tf.nn.relu(tf.matmul(input_, tf.transpose(matrix1)) + bias_term1)
	    return tf.matmul(hidden1, tf.transpose(matrix2)) + bias_term2


	def padding(self, data):
		#print data[0]
		padded_batch_set = []
		max_docs_len = max([len(doc) for doc,_,_,_ in data])
		max_words_len = max([max([len(sentence) for sentence in doc]) for doc,_,_,_ in data])
		for doc, label, _, _ in data:
			docs_len = len(doc)
			doc_pad = doc + [[0]] * (max_docs_len - docs_len)
			words_len = [len(sentence) for sentence in doc_pad]
			doc_pad = [sentence + [0] * (max_words_len - len(sentence)) for sentence in doc_pad]
			padded_batch_set.append([doc_pad, label, docs_len, words_len])

		return padded_batch_set

	def get_batch_set(self, data, batch_size, shuffle = True):

		data_size = len(data)
		num_batches = int(data_size/batch_size) if data_size % batch_size == 0 else int(data_size/batch_size) + 1
		# Shuffle the data at each epoch
		if shuffle:
			random.shuffle(data)

		for batch_num in range(num_batches):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			
			yield self.padding(data[start_index:end_index])

	def fit(self, train_set, test_set, epoch = 40):
		config = tf.ConfigProto(allow_soft_placement=True)
		config.gpu_options.allow_growth=True
		# config.gpu_options.per_process_gpu_memory_fraction = 0.4
		with tf.Session(config=config) as sess:
			sess.run(tf.global_variables_initializer())
			f_log = open(self.log_file, 'w')
			for j in range(epoch):
			 	batch_set = self.get_batch_set(train_set, self.batch_size, True)
			 	loss, start_time = 0.0, time.time()
			 	num_batches = int(len(train_set)/self.batch_size) if len(train_set) % self.batch_size == 0 else int(len(train_set)/self.batch_size) + 1
			 	for i, batch_sample in enumerate(batch_set):
			 		docs, labels, doc_len, word_len = zip(*batch_sample)
			 		#print X, '\n', Y, '\n', length, '\n', mask, '\n', self.dropout, '\n'
			 		#return
			 		batch_loss, _ = sess.run([self.mean_loss, self.train_op],
			 								 {self.docs: docs, self.labels: labels, self.doc_len: doc_len, 
			 								 self.word_len: word_len, self.dropout_keep_prob: self.dropout})
			 		loss += batch_loss
			 		print ('training %.2f ...' % ((i+1) * 100.0 / num_batches))
			 		sys.stdout.write("\033[F")
			 	print("%d : loss = %.3f, time = %.3f" % (j+1, loss, time.time() - start_time), end='')
			 	
			 	f_log.write("%d,%.3f,%.3f" % (j+1, loss, time.time() - start_time))
			 	results = self.evaluate(test_set, sess)
			 	for (key, value) in results.items():
			 		print (", %s = %.3f" % (key, value), end='')
			 		f_log.write(",%.3f" % value)
			 	print ('\n')
			 	f_log.write('\n')
			 	
			f_log.close()

	
	def evaluate(self, test_set, sess):
		batch_set = self.get_batch_set(test_set, self.batch_size, False)
		total_correct = []
		for batch_sample in batch_set:
			docs, labels, doc_len, word_len = zip(*batch_sample)
			#print(np.array(docs).shape)
			#print(np.array(labels).shape)
			#print(np.array(doc_len).shape)
			#print(np.array(word_len).shape)
			correct = sess.run(self.correct,
								  {self.docs: docs, self.labels: labels, self.doc_len: doc_len, 
			 					   self.word_len: word_len, self.dropout_keep_prob: 1.0})
			#print (correct)
			#return
			total_correct.append(correct)
		sum_correct = np.sum(np.array(total_correct))*1.0
		total = len(test_set)*1.0
		accuracy = np.sum(np.array(total_correct)) * 100.0 / len(test_set)
		'''
		print(total_correct)
		print(sum_correct)
		print(total)
		print(accuracy)
		'''
		return {'accuracy' : accuracy, 'correct' : sum_correct, 'total' : total}
	
	

	def get_cell(self, output_dim, dropout_keep_prob = 1.0):
		if self.rnn_cell.endswith('lstm'):
			# if 'reuse' in inspect.signature(tf.contrib.rnn.LSTMCell.__init__).parameters:
			# 	cell = tf.contrib.rnn.LSTMCell(self.size, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
			# else:
			cell = tf.contrib.rnn.LSTMCell(self.size, state_is_tuple=True)
		if self.rnn_cell.endswith('gru'):
			cell = tf.contrib.rnn.GRUCell(self.size)
		if self.rnn_cell.endswith('rnn'):
			cell = tf.contrib.rnn.BasicRNNCell(self.size)
		cell = tf.contrib.rnn.DropoutWrapper(cell, dropout_keep_prob, dropout_keep_prob)
		cell = tf.contrib.rnn.OutputProjectionWrapper(cell, output_dim)
		return cell

	def build(self):
		self.docs = tf.placeholder(tf.int32, [None, None, None])
		self.labels = tf.placeholder(tf.int32, [None])
		self.word_len = tf.placeholder(tf.int32, [None, None])
		self.doc_len = tf.placeholder(tf.int32, [None])
		self.dropout_keep_prob = tf.placeholder(tf.float32)
		batch_size, doc_size, word_size = tf.shape(self.docs)[0], tf.shape(self.docs)[1], tf.shape(self.docs)[2]
		docs = tf.reshape(self.docs, [-1, word_size])
		word_len = tf.reshape(self.word_len, [-1])

		with tf.device('/cpu:0'):
			word_embedding = tf.Variable(self.embeddings.word_embeddings, dtype = tf.float32)
			docs_embed = tf.nn.embedding_lookup(word_embedding, docs)

		with tf.variable_scope("sen_rnn"):
			if self.bidirection == False:
				cell = self.get_cell(self.size, self.dropout_keep_prob)
				outputs, state = tf.nn.dynamic_rnn(cell, inputs = docs_embed, sequence_length = word_len, dtype = tf.float32)
			else:
				cell_fw = self.get_cell(self.size, self.dropout_keep_prob)
				cell_bw = self.get_cell(self.size, self.dropout_keep_prob)
				outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_bw, cell_fw, inputs = docs_embed, sequence_length = tf.to_int64(word_len), dtype = tf.float32)
				outputs = outputs[0] + outputs[1]

		docs_embed = tf.reduce_mean(outputs, 1)
		docs_embed = tf.reshape(docs_embed, [batch_size, doc_size, self.size])

		with tf.variable_scope("doc_rnn"):
			if self.bidirection == False:
				cell = self.get_cell(self.size, self.dropout_keep_prob)
				outputs, state = tf.nn.dynamic_rnn(cell, inputs = docs_embed, sequence_length = self.doc_len, dtype = tf.float32)
			else:
				cell_fw = self.get_cell(self.size, self.dropout_keep_prob)
				cell_bw = self.get_cell(self.size, self.dropout_keep_prob)
				outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_bw, cell_fw, inputs = docs_embed, sequence_length = tf.to_int64(self.doc_len), dtype = tf.float32)
				outputs = outputs[0] + outputs[1]

		docs_embed = tf.reduce_mean(outputs, 1)
		docs_embed = self.linear(docs_embed, self.num_class)

		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = docs_embed, labels = self.labels)
		self.mean_loss = tf.reduce_mean(loss)
		self.predictions = tf.cast(tf.argmax(docs_embed, 1), tf.int32)
		self.correct = tf.reduce_sum(tf.cast(tf.equal(self.predictions, self.labels), tf.int32))

		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(self.mean_loss, tvars), self.max_grad_norm)
		if self.optimize == 'Adagrad':
			optimizer = tf.train.AdagradOptimizer(0.1)
		else:
			optimizer = tf.train.AdamOptimizer()
		self.train_op = optimizer.apply_gradients(zip(grads, tvars))
